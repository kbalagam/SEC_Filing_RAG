"""
Single source of truth for talking to the Gemini API — client setup and
retry-with-backoff logic.

Before this file existed, call_with_retry() was copy-pasted across
day1_api_basics.py, day4_embed_and_store.py, and generate_answer.py.
That meant a bug fix (like broadening the except clause for network
errors, or reading the server's retryDelay instead of guessing) would
need to be applied in multiple places. One copy, imported everywhere.
"""

import os
import sys
import time

from google import genai
from google.genai import errors, types

from src.config import CHAT_MODEL, EMBED_MODEL, MAX_RETRIES, RETRYABLE_STATUS_CODES


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        print("  export GEMINI_API_KEY=your_key_here  (or put it in .env)")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def call_with_retry(client: genai.Client, prompt: str, system_instruction: str = None,
                     max_retries: int = MAX_RETRIES):
    """Chat completion call with retry-with-backoff on transient errors.
    Returns the FULL response object (not just .text) — callers extract
    what they need (text, usage_metadata, etc.)."""
    config = None
    if system_instruction:
        config = types.GenerateContentConfig(system_instruction=system_instruction)

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(model=CHAT_MODEL, contents=prompt, config=config)
        except errors.APIError as e:
            if getattr(e, "code", None) in RETRYABLE_STATUS_CODES and attempt < max_retries:
                wait = 3 ** attempt
                print(f"  [retry {attempt}] API error {e.code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [FAILED] Non-retryable error: {e}")
            raise
    raise RuntimeError("Exceeded max retries")


def embed_with_retry(client: genai.Client, text: str, task_type: str,
                      max_retries: int = MAX_RETRIES) -> list[float]:
    """Embedding call with retry-with-backoff on transient errors.
    task_type should be "RETRIEVAL_DOCUMENT" for chunks being stored,
    or "RETRIEVAL_QUERY" for a user's search question."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return response.embeddings[0].values
        except errors.APIError as e:
            if getattr(e, "code", None) in RETRYABLE_STATUS_CODES and attempt < max_retries:
                wait = 3 ** attempt
                print(f"    [retry {attempt}] API error {e.code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    [FAILED] Non-retryable error: {e}")
            raise
    raise RuntimeError("Exceeded max retries")
