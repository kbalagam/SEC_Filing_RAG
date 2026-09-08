"""
Hybrid search: combines BM25 (keyword/lexical) search with existing
vector (semantic) search via Reciprocal Rank Fusion (RRF).

WHY: pure vector search can underperform on queries containing specific,
exact terms (e.g. "NHTSA", "GDPR", a specific dollar figure) — embeddings
capture MEANING, not exact tokens, so a query for one specific regulatory
body name might retrieve chunks about various regulatory bodies in
general, rather than the exact one asked about. BM25 excels at exactly
this: exact term matching via an inverted index.

WHY RRF, not naive score averaging: BM25 scores and vector similarity/
distance scores are on completely different, incompatible scales.
Naively averaging or min-max normalizing them is fragile — a single
BM25 outlier (a chunk that repeats a query term many times) can distort
the whole scale. RRF instead combines RANKS, not raw scores:

    RRF_score(doc) = sum over each result list of: 1 / (k + rank_in_that_list)

This rewards documents that rank well in BOTH methods (consistency),
regardless of the two methods' incompatible raw score scales. k=60 is
the standard constant from the original RRF paper.

BM25 index is built once from all documents already stored in Chroma
(collection.get() with no filter returns everything), tokenized with
simple lowercase + whitespace splitting — a crude but standard starting
baseline per rank_bm25's own documented example. The SAME tokenization
must be applied to queries at search time, or corpus/query preprocessing
would be mismatched (same principle as matching RETRIEVAL_DOCUMENT /
RETRIEVAL_QUERY task types for embeddings).

KNOWN LIMITATION (documented, not chased): BM25 indices are not
automatically updated when the underlying document set changes — per
industry sources, index rebuilding must be treated as part of the
ingestion pipeline, not a one-time setup step. This module's
build_bm25_index() would need to be re-run any time new chunks are
added via scripts/run_pipeline.py.
"""

import re
from rank_bm25 import BM25Okapi

RRF_K = 60  # standard constant from the original RRF paper


def tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation + whitespace split. MUST be applied
    identically to corpus (at index time) and queries (at search time).

    Improved from a naive .lower().split() after finding a real bug:
    dense tabular/boilerplate text (e.g. exhibit lists with codes like
    "101.INS") produced unusual tokens like '101.ins' that BM25's
    rarity-based scoring (IDF) rewarded highly, even for chunks
    completely unrelated to the actual query. Stripping punctuation
    before splitting turns "101.ins" into "101" and "ins" — both far
    more common, less artificially "rare" tokens."""
    cleaned = re.sub(r'[^\w\s]', '', text)
    return cleaned.lower().split()


def build_bm25_index(collection) -> dict:
    """Pull every stored chunk out of Chroma and build a BM25 index
    from it. Returns a dict bundling the index with the parallel
    ids/metadatas lists needed to map BM25 results back to real chunks."""
    all_data = collection.get(include=["documents", "metadatas"])

    documents = all_data["documents"]
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]

    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "bm25": bm25,
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }


def bm25_search(bm25_index: dict, query: str, top_k: int) -> list[dict]:
    """Search the BM25 index, returning results in the same shape as
    vector retrieve() — {"text", "metadata", "id"} — plus a "rank" field
    (1-indexed) needed for RRF fusion."""
    tokenized_query = tokenize(query)
    scores = bm25_index["bm25"].get_scores(tokenized_query)

    # pair each score with its index, sort descending, take top_k
    scored = list(enumerate(scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = scored[:top_k]

    results = []
    for rank, (idx, score) in enumerate(top, start=1):
        results.append({
            "id": bm25_index["ids"][idx],
            "text": bm25_index["documents"][idx],
            "metadata": bm25_index["metadatas"][idx],
            "rank": rank,
            "bm25_score": score,
        })
    return results


def reciprocal_rank_fusion(vector_results: list[dict], keyword_results: list[dict],
                            top_k: int, k: int = RRF_K) -> list[dict]:
    """Combine two ranked result lists into one, using RRF.

    vector_results: from retrieve() — already has "id" and "rank" fields.
    keyword_results: from bm25_search() — already has "id" and "rank".
    """
    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for result_list in (vector_results, keyword_results):
        for hit in result_list:
            chunk_id = hit["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + hit["rank"])
            chunk_lookup[chunk_id] = hit  # keep one copy of the actual chunk data

    ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused = []
    for chunk_id in ranked_ids[:top_k]:
        hit = chunk_lookup[chunk_id]
        fused.append({
            "id": chunk_id,
            "text": hit["text"],
            "metadata": hit["metadata"],
            "rrf_score": rrf_scores[chunk_id],
        })
    return fused


def hybrid_retrieve(client, collection, bm25_index: dict, query: str, top_k: int = 3) -> list[dict]:
    """Full hybrid search: run vector search and BM25 search independently,
    then fuse their results with RRF. This is the function real code
    (e.g. generate_answer) should call — same pattern as Week 1's
    retrieve_and_answer() combining independently-tested pieces."""
    from src.rag.retrieve import retrieve  # import here to avoid a hard
                                             # dependency for callers that
                                             # only need bm25_search/RRF

    # Retrieve MORE than top_k from each individual method before fusion —
    # a chunk might rank #8 by vector search but #1 by keyword search;
    # fusing only each method's top-3 would lose that chunk entirely.
    fetch_k = max(top_k * 3, 10)

    vector_results = retrieve(client, collection, query, top_k=fetch_k)
    keyword_results = bm25_search(bm25_index, query, top_k=fetch_k)

    return reciprocal_rank_fusion(vector_results, keyword_results, top_k=top_k)