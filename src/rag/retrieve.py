"""
Given a natural-language question, embed it and return the top_k most
relevant stored chunks, with their source metadata attached.
"""

from src.llm_client import embed_with_retry
from src.config import DEFAULT_TOP_K


def retrieve(client, collection, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Embed a query (using RETRIEVAL_QUERY task type, matching the
    RETRIEVAL_DOCUMENT type used when chunks were stored) and return
    the top_k most relevant chunks with metadata and distance."""
    query_embedding = embed_with_retry(client, query, task_type="RETRIEVAL_QUERY")
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits
