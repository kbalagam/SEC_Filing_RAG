"""
Given a natural-language question, embed it and return the top_k most
relevant stored chunks, with their source metadata attached.

UPDATED for hybrid search: now includes "id" and "rank" (1-indexed) on
each hit, needed to fuse these results with BM25 keyword search results
via Reciprocal Rank Fusion. Chroma's .query() already returns results
sorted by relevance, so "rank" is simply the position in that order —
enumerate() gives both the item and its position in one pass.
"""

from src.llm_client import embed_with_retry
from src.config import DEFAULT_TOP_K


def retrieve(client, collection, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Embed a query (using RETRIEVAL_QUERY task type, matching the
    RETRIEVAL_DOCUMENT type used when chunks were stored) and return
    the top_k most relevant chunks with metadata, distance, id, and rank."""
    query_embedding = embed_with_retry(client, query, task_type="RETRIEVAL_QUERY")
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for rank, (chunk_id, doc, meta, dist) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        hits.append({
            "id": chunk_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "rank": rank,
        })
    return hits