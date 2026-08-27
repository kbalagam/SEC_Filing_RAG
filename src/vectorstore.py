"""
Embed filing chunks with Gemini and store them in Chroma.

Design decisions made deliberately:
  - task_type="RETRIEVAL_DOCUMENT" for chunks being stored (vs.
    "RETRIEVAL_QUERY" used later for user questions — matching task
    types on each side improves retrieval accuracy).
  - RESUMABLE: before embedding a chunk, check if its ID already exists
    in the Chroma collection. If yes, skip it. This means a crash or
    quota exhaustion mid-run doesn't waste already-completed API calls —
    just rerun the pipeline and it picks up where it left off. This was
    tested for real under actual rate-limit and network-error crashes
    during development, not just in a demo scenario.
  - A small delay (RATE_LIMIT_DELAY) after each successful embedding
    call, added after hitting the free tier's 100-requests-per-minute
    embedding limit in practice. Proactively staying under the limit
    beats reactively recovering from 429s on every run.
"""

import time
import chromadb

from src.llm_client import embed_with_retry

RATE_LIMIT_DELAY_SECONDS = 0.7  # stays under Gemini free tier's 100 req/min embedding limit


def get_chroma_collection(db_path: str, collection_name: str):
    chroma_client = chromadb.PersistentClient(path=db_path)
    return chroma_client.get_or_create_collection(name=collection_name)


def make_chunk_id(company: str, section_title: str, index: int) -> str:
    """Build a stable, unique ID per chunk. Used both to store the chunk
    and to check whether it's already been stored (resumability)."""
    return f"{company}_{section_title}_{index}".replace(" ", "_").replace(".", "")


def embed_and_store_chunks(client, collection, company: str, section_title: str, chunks: list[str]) -> None:
    """Embed and store each chunk from one section, skipping any chunk
    whose ID already exists in the collection (resumability)."""
    for i, chunk_text in enumerate(chunks):
        chunk_id = make_chunk_id(company, section_title, i)

        existing = collection.get(ids=[chunk_id])
        if existing["ids"]:
            print(f"    [skip] {chunk_id} already embedded.")
            continue

        embedding = embed_with_retry(client, chunk_text, task_type="RETRIEVAL_DOCUMENT")

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk_text],
            metadatas=[{
                "company": company,
                "section_title": section_title,
                "chunk_index": i,
            }],
        )
        print(f"    [added] {chunk_id}")
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
