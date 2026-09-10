"""
Generate a grounded, cited answer from retrieved chunks.

System prompt explicitly instructs the model to:
  - answer ONLY from the provided context (prevents hallucination —
    without this instruction, the model blends its own training
    knowledge with the retrieved text, and you can't tell which is which)
  - cite which source(s) it used (possible because build_context labels
    every chunk with its company + section before it reaches the model)
  - explicitly say "I don't know" if the context doesn't answer the
    question, rather than guessing — verified working with deliberately
    off-topic test questions
  - NOT invent additional source detail (page numbers, dates) beyond
    what's in the [Source: ...] label. This rule was added after a real
    bug was found in testing: the model initially invented plausible-
    looking page numbers that didn't exist anywhere in the metadata —
    a subtle hallucination riding inside an otherwise well-grounded
    answer. Fixed by making the citation boundary explicit and concrete
    rather than a vague "don't fabricate" instruction; verified fixed
    via direct before/after comparison on the same real question.
"""

from src.llm_client import call_with_retry

SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer the user's question "
    "using ONLY the information in the provided context below. "
    "Do not use any outside knowledge, even if you know it. "
    "When you use information from the context, mention which source "
    "it came from (e.g. 'According to AAPL's Item 1A. Risk Factors...'). "
    "Only cite the company and section title exactly as given in each "
    "[Source: ...] label — do not invent page numbers, dates, or any "
    "other source detail that isn't explicitly present in the label. "
    "If the context does not contain enough information to answer the "
    "question, say clearly: 'I don't have enough information in the "
    "provided filings to answer that.' Do not guess or make anything up."
)


def build_context(hits: list[dict]) -> str:
    """Turn retrieved hits into a labeled context block, so the model
    can see and cite which source each piece of text came from."""
    context = ""
    for hit in hits:
        text = hit["text"]
        company = hit["metadata"]["company"]
        section_title = hit["metadata"]["section_title"]
        context += f"[Source: {company}, {section_title}]\n{text}\n\n"
    return context


def generate_answer(client, question: str, hits: list[dict]) -> str:
    """Given a question and pre-retrieved hits, build a grounded prompt
    and generate a cited answer.

    Deliberately takes hits as a parameter rather than calling retrieve()
    internally — this keeps generation testable with fake hits, without
    spending embedding API calls, separate from retrieve_and_answer()
    below which wires both together for actual use."""
    context = build_context(hits)
    prompt = f"""Context:
{context}
Question: {question}"""

    response = call_with_retry(client, prompt, system_instruction=SYSTEM_PROMPT)
    return response.text


def retrieve_and_answer(client, collection, bm25_index, question: str, top_k: int = None) -> str:
    from hybrid_search import hybrid_retrieve
    from src.config import DEFAULT_TOP_K

    k = top_k if top_k is not None else DEFAULT_TOP_K
    hits = hybrid_retrieve(client, collection, bm25_index, question, k)
    return generate_answer(client, question, hits)
