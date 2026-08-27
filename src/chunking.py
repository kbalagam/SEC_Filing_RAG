"""
Chunk section text for embedding.

Takes a section's paragraphs and produces chunks in a target size range,
measured in TOKENS (not words) — tokens are the actual unit LLMs and
embedding models operate on.

  - Paragraphs under MIN_TOKENS get merged with the next paragraph
    (too short to carry standalone meaning on its own).
  - Paragraphs over MAX_TOKENS get split further, sentence by sentence,
    with a small overlap (snapped to sentence boundaries, not raw
    token cuts) so context isn't sharply lost at a forced split point.

Thresholds (150 / 512 / 50, see config.py) follow common 2026 industry
defaults for recursive/structure-aware chunking: ~512 tokens as the
upper bound, a low floor to only catch genuinely thin fragments (not
force-merge every normal-sized paragraph), and ~10% overlap. Overlap's
benefit is contested in some 2026 retrieval studies — kept here as a
reasonable default, not treated as a guaranteed win.

Token counting uses tiktoken's cl100k_base encoding — a fast, local
approximation. Not exact for Gemini specifically, but standard practice
for chunking decisions (calling a live LLM API just to count tokens
during preprocessing would be slow and wasteful).

Known limitation: thresholds are heuristics, not perfect. A chunk could
still end up slightly under/over target size, and the very last
paragraph in a section may form a standalone chunk smaller than
MIN_TOKENS if there's nothing left to merge it with. Documented, not
chased further — a deliberate tradeoff, not an oversight (see README).
"""

import re
import tiktoken

from src.config import MIN_TOKENS, MAX_TOKENS, OVERLAP_TOKENS

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using OpenAI's cl100k_base encoding — a widely used,
    fast, local approximation for general English text."""
    return len(_ENC.encode(text))


def get_overlap_sentences(chunk_sentences: list[str], overlap_tokens: int) -> list[str]:
    """Given a finalized chunk's sentences, return the trailing sentences
    (in correct reading order) to seed the next chunk with, targeting
    roughly `overlap_tokens` tokens."""
    overlap = []
    token_count = 0
    for sentence in reversed(chunk_sentences):
        overlap.append(sentence)
        token_count += count_tokens(sentence)
        if token_count >= overlap_tokens:
            break
    return overlap[::-1]


def split_long_paragraph(text: str, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """Split a long paragraph into ~max_tokens chunks, splitting only at
    sentence boundaries, with sentence-level overlap between chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]  # drop empty fragments

    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_token_count = count_tokens(sentence)

        if current_token_count + sentence_token_count > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_sentences = get_overlap_sentences(current_chunk, overlap_tokens)
            current_chunk = overlap_sentences
            current_token_count = sum(count_tokens(s) for s in overlap_sentences)

        current_chunk.append(sentence)
        current_token_count += sentence_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def merge_short_paragraphs(paragraphs: list[str], min_tokens: int = MIN_TOKENS) -> list[str]:
    """Combine consecutive paragraphs so no merged unit is under min_tokens,
    unless it's the very last paragraph in the section (nothing left to
    merge it with)."""
    merged = []
    buffer = ""
    buffer_token_count = 0

    for para in paragraphs:
        para_token_count = count_tokens(para)

        if buffer:
            buffer = buffer + " " + para
        else:
            buffer = para
        buffer_token_count += para_token_count

        if buffer_token_count >= min_tokens:
            merged.append(buffer)
            buffer = ""
            buffer_token_count = 0

    # leftover buffer (didn't reach min_tokens) — still append it,
    # better to keep a slightly-short final chunk than lose the text
    if buffer:
        merged.append(buffer)

    return merged


def chunk_section(paragraphs: list[str]) -> list[str]:
    """Full pipeline for one section's paragraphs -> embedding-ready chunks.

    1. Merge short paragraphs together (avoid thin, low-signal chunks).
    2. Split any resulting chunk that's still too long (avoid diluted,
       multi-topic chunks).
    """
    merged = merge_short_paragraphs(paragraphs, MIN_TOKENS)

    final_chunks = []
    for chunk in merged:
        token_count = count_tokens(chunk)
        if token_count > MAX_TOKENS:
            final_chunks.extend(split_long_paragraph(chunk, MAX_TOKENS, OVERLAP_TOKENS))
        else:
            final_chunks.append(chunk)

    return final_chunks
