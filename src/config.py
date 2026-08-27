"""
Centralized configuration. Every constant used across the pipeline lives
here — model names, chunking thresholds, storage paths, SEC headers.

Before this file existed, these values were duplicated (and slightly
inconsistent) across multiple scripts — e.g. MIN_WORDS was 400 in one
place, 100 in another, before being unified as MIN_TOKENS=150 here.
One source of truth prevents that class of bug.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini models ---
# NOTE: model names/versions move fast (we hit a deprecated model on Day 1
# and had to update it based on the error message's own suggestion).
# Verify these are still current if you see a 404 NOT_FOUND error.
CHAT_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-001"

# --- SEC EDGAR ---
# SEC requires a real identifying User-Agent or it will block requests.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your_email@example.com")
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}
SEC_COMPANIES = {"AAPL": None, "TSLA": None, "MSFT": None}

# --- Chunking (tokens, not words — see chunking.py for reasoning) ---
MIN_TOKENS = 150
MAX_TOKENS = 512
OVERLAP_TOKENS = 50

# --- Storage ---
SECTIONS_CACHE_DIR = "sections_cache"
CHROMA_DB_PATH = "./chroma_store"
CHROMA_COLLECTION_NAME = "filings_real"

# --- Retrieval ---
DEFAULT_TOP_K = 3

# --- Retry behavior (used by llm_client.py) ---
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = (429, 500, 503)
