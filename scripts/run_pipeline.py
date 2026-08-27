"""
Main pipeline orchestrator: for each configured company, fetch (or load
from cache) its parsed sections, chunk each section, and embed + store
every chunk. Run this once to build the vector database; safe to rerun
(resumable) if interrupted.

Run: GEMINI_API_KEY=your_key python3 scripts/run_pipeline.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bs4 import BeautifulSoup

from src.config import SEC_COMPANIES, SECTIONS_CACHE_DIR, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
from src.llm_client import get_client
from src.ingest.sec_downloader import get_ticker_to_cik_map, get_recent_10k, build_document_url, fetch_filing_html
from src.ingest.section_parser import split_into_sections
from src.chunking import chunk_section
from src.vectorstore import get_chroma_collection, embed_and_store_chunks


def get_cache_path(ticker: str) -> str:
    return os.path.join(SECTIONS_CACHE_DIR, f"{ticker}_sections.json")


def load_or_fetch_sections(ticker: str, cik: str) -> dict:
    """Return {section_title: [paragraphs]} for this company — from cache
    if available, otherwise fetch fresh from SEC and cache the result."""
    cache_path = get_cache_path(ticker)

    if os.path.exists(cache_path):
        print(f"[{ticker}] Loading sections from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[{ticker}] No cache found — fetching from SEC...")
    filing = get_recent_10k(cik)
    if not filing:
        print(f"[{ticker}] No 10-K found.")
        return {}

    doc_url = build_document_url(cik, filing["accession_number"], filing["primary_document"])
    html_content = fetch_filing_html(doc_url)
    soup = BeautifulSoup(html_content, "html.parser")

    sections = split_into_sections(soup)

    os.makedirs(SECTIONS_CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    print(f"[{ticker}] Cached {len(sections)} sections to {cache_path}")

    return sections


def main():
    client = get_client()
    collection = get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)

    print("Fetching ticker -> CIK map...")
    ticker_map = get_ticker_to_cik_map()

    for ticker in SEC_COMPANIES:
        cik = ticker_map.get(ticker)
        if not cik:
            print(f"[{ticker}] CIK not found, skipping.")
            continue

        sections = load_or_fetch_sections(ticker, cik)

        for section_title, paragraphs in sections.items():
            if not paragraphs:
                continue
            chunks = chunk_section(paragraphs)
            print(f"[{ticker}] {section_title}: {len(paragraphs)} paragraphs -> {len(chunks)} chunks")
            embed_and_store_chunks(client, collection, ticker, section_title, chunks)

    print(f"\nTotal chunks stored: {collection.count()}")


if __name__ == "__main__":
    main()
