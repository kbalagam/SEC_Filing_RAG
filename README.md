# SEC Filing RAG

Ask natural-language questions about real SEC 10-K filings (Apple, Tesla, Microsoft) and get grounded, cited answers — built from scratch, not a framework quickstart.

## What this is

A retrieval-augmented generation (RAG) system that:
1. Downloads real 10-K filings directly from SEC EDGAR's public API
2. Parses the raw HTML into labeled sections (Item 1A. Risk Factors, Item 7. MD&A, etc.)
3. Chunks each section into token-sized pieces suitable for embedding
4. Embeds every chunk with Gemini and stores it in a local vector database (Chroma)
5. Given a question, retrieves the most relevant chunks and generates a cited answer — using only the retrieved text, never outside knowledge

## Why this project

SEC filings are real, messy, unstructured documents — not clean Wikipedia articles. Different filers format their HTML completely differently (see [Known limitations](#known-limitations)), which makes this a genuinely representative RAG engineering problem rather than a tutorial clone.

## Architecture

```
SEC EDGAR (raw filing HTML)
        │
        ▼
src/ingest/sec_downloader.py    — ticker → CIK lookup, fetch filing HTML
        │
        ▼
src/ingest/section_parser.py    — parse HTML into {section_title: [paragraphs]}
        │                          (dual-method header/paragraph detection
        │                           with automatic fallback — see below)
        ▼
src/chunking.py                 — merge short / split long paragraphs into
        │                          ~512-token chunks, sentence-boundary overlap
        ▼
src/vectorstore.py              — embed each chunk (Gemini), store in Chroma
        │                          (resumable — safe to interrupt and rerun)
        ▼
src/rag/retrieve.py             — embed a question, return top-k relevant chunks
        │
        ▼
src/rag/generate.py             — build a labeled, cited prompt; generate an
                                    answer grounded only in retrieved context
```

`scripts/run_pipeline.py` orchestrates ingestion → chunking → embedding for all configured companies. `app.py` is a Streamlit UI on top of the same, already-tested `retrieve_and_answer()` function.

## Setup

```bash
git clone <this-repo>
cd sec-filing-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: add your GEMINI_API_KEY and SEC_USER_AGENT (SEC requires a
# real identifying User-Agent or it will block requests)
```

Build the vector database (safe to interrupt — rerunning skips already-embedded chunks):
```bash
python3 scripts/run_pipeline.py
```

Run the UI:
```bash
streamlit run app.py
```

Run tests:
```bash
pytest tests/ -v
```

## Known limitations

These were found through real debugging, not theoretical — each was investigated with direct evidence before being accepted as a deliberate tradeoff rather than chased indefinitely.

- **Different filers use different HTML conventions.** AAPL/TSLA use clean `<b>`/`<strong>` header tags and `<div>`-based paragraphs (no `<p>` tags at all in the document). MSFT splits header text across multiple sibling `<span>` tags and uses real `<p>` tags for body text. `section_parser.py` handles both via dual-method detection with automatic fallback (tries one approach, falls back to the other if results look unreliable — e.g., suspiciously short header titles).
- **Multi-span header titles may contain a stray space at the split point** (e.g. "ITEM 1. B USINESS" instead of "BUSINESS"). This is cosmetic — it affects the citation *label* only. The underlying paragraph *content* in that section is complete and correctly retrieved; verified via direct inspection.
- **One isolated section-boundary miss was found** (a TSLA "ITEM 1A. RISK FACTORS" header wasn't detected at one specific occurrence, so it appeared as trailing text in the prior section instead of starting a new one). Confirmed via targeted follow-up queries (legal proceedings, properties, cybersecurity sections) to be an isolated occurrence, not a systemic pattern.
- **The last paragraph in a section may form its own chunk under the minimum token threshold**, since there's nothing after it to merge with. Documented and accepted rather than adding backward-merge logic, which would risk pushing the previous chunk over the maximum size instead.
- **Table data is not specially parsed.** Financial tables lose their row/column structure when flattened to text. This is a known, industry-recognized RAG failure mode (fixed-size or naive splitters commonly separate table headers from their data), not unique to this project. A production version would parse `<table>` elements separately.
- **Citation fabrication was found and fixed during testing.** An early version of the generation prompt allowed the model to invent plausible-looking page numbers not present in the actual metadata. Fixed by making the system prompt explicitly forbid citing anything beyond what's in the `[Source: ...]` label — verified fixed via direct before/after comparison on the same question.

## What I'd improve with more time

- Parse `<table>` elements separately from prose, so financial statement data isn't lost
- Fix the multi-span header-splitting cosmetic issue with word-boundary-aware joining
- Add a real evaluation set (20+ question/expected-answer pairs) and track retrieval accuracy over time, rather than manual spot-checking
- Support incremental updates (new filings) without re-running the full pipeline
- Add response streaming in the UI for better perceived latency

## Tech stack

- **LLM & embeddings:** Google Gemini (`gemini-3.6-flash` for generation, `gemini-embedding-001` for embeddings)
- **Vector store:** ChromaDB (local, persistent)
- **Data source:** SEC EDGAR public API
- **UI:** Streamlit
- **Chunking:** custom token-aware chunker (tiktoken for counting) with sentence-boundary-safe overlap
