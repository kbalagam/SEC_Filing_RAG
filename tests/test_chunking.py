"""
Tests for src/chunking.py — promoted from the original __main__ sanity
check (which printed output for manual eyeballing) into real assertions.

Run: pytest tests/test_chunking.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunking import chunk_section, count_tokens
from src.config import MIN_TOKENS, MAX_TOKENS


FAKE_PARAGRAPHS = [
    "Short intro sentence.",  # too short alone
    "Another short one here too.",  # still short
    " ".join([f"This is filler sentence number {i} about business risk." for i in range(120)]),  # very long
    "A reasonably sized closing paragraph that stands fine on its own without needing to merge with anything else nearby.",
]


def test_chunk_section_produces_chunks():
    chunks = chunk_section(FAKE_PARAGRAPHS)
    assert len(chunks) > 0


def test_short_paragraphs_get_merged():
    """The two short intro paragraphs should not survive as standalone
    chunks — they should be merged into something larger."""
    chunks = chunk_section(FAKE_PARAGRAPHS)
    # neither "Short intro sentence." nor "Another short one here too."
    # should appear as an ENTIRE chunk on its own
    assert "Short intro sentence." not in chunks
    assert "Another short one here too." not in chunks


def test_long_paragraph_gets_split():
    """The ~1080-token filler paragraph should be split into multiple
    chunks, each roughly within MAX_TOKENS."""
    chunks = chunk_section(FAKE_PARAGRAPHS)
    long_chunks = [c for c in chunks if "filler sentence number" in c]
    assert len(long_chunks) > 1  # confirms it was actually split

    for c in long_chunks:
        # allow some slack above MAX_TOKENS for overlap, but it shouldn't
        # be wildly over (e.g. the whole 1080-token blob in one chunk)
        assert count_tokens(c) <= MAX_TOKENS + 100


def test_known_limitation_trailing_short_chunk():
    """Documented, accepted limitation: the last paragraph in a section
    may form its own standalone chunk even if under MIN_TOKENS, since
    there's nothing after it to merge with. This test documents that
    behavior explicitly rather than letting it silently exist."""
    chunks = chunk_section(FAKE_PARAGRAPHS)
    last_chunk = chunks[-1]
    assert "reasonably sized closing paragraph" in last_chunk
    # confirmed known limitation: this chunk is allowed to be under
    # MIN_TOKENS since it's the trailing paragraph with nothing to merge into
