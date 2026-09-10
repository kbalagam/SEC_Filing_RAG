"""
Streamlit UI for the SEC Filing RAG system.

Thin wrapper around the already-tested pipeline (src/rag/generate.py's
retrieve_and_answer). This file contains no business logic of its own —
all retrieval/generation/citation logic lives in src/, already verified
independently. This keeps the UI layer simple and means UI bugs can't
hide pipeline bugs, or vice versa.

Run: streamlit run app.py
"""

import streamlit as st

from src.llm_client import get_client
from src.vectorstore import get_chroma_collection
from src.rag.generate import retrieve_and_answer
from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, SEC_COMPANIES, DEFAULT_TOP_K

st.set_page_config(page_title="SEC Filing RAG", page_icon="📄", layout="centered")

st.title("📄 SEC Filing Q&A")
st.caption(
    f"Ask questions about the 10-K filings of: {', '.join(SEC_COMPANIES.keys())}. "
    "Answers are grounded in the actual filing text and cite their source section."
)


@st.cache_resource
def load_client_and_collection():
    """Cached across reruns — avoids reconnecting to the API/DB on every
    keystroke or interaction, which Streamlit would otherwise trigger."""
    client = get_client()
    collection = get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
    return client, collection

@st.cache_resource
def load_bm25_index(_collection):
    from hybrid_search import build_bm25_index
    return build_bm25_index(_collection)

try:
    client, collection = load_client_and_collection()
    chunk_count = collection.count()
except SystemExit:
    st.error(
        "GEMINI_API_KEY is not set. Add it to your .env file before running this app."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not connect to the vector database: {e}")
    st.info("Have you run `python3 scripts/run_pipeline.py` yet to build the database?")
    st.stop()

if chunk_count == 0:
    st.warning(
        "The vector database is empty. Run `python3 scripts/run_pipeline.py` "
        "first to fetch, chunk, and embed the filings."
    )
    st.stop()

bm25_index = load_bm25_index(collection)

st.caption(f"Database contains {chunk_count} indexed chunks.")

with st.expander("Example questions"):
    st.markdown(
        "- What supply chain risks does Apple face?\n"
        "- What are Tesla's main risk factors?\n"
        "- What business segments does Microsoft operate?"
    )

question = st.text_input("Ask a question about these filings:", placeholder="e.g. What are Apple's biggest risks?")
top_k = st.slider("Number of source chunks to retrieve", min_value=1, max_value=10, value=DEFAULT_TOP_K)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving relevant filing sections and generating an answer..."):
        try:
            answer = retrieve_and_answer(client, collection, bm25_index, question, top_k=top_k)
            st.markdown("### Answer")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Something went wrong generating the answer: {e}")

st.divider()
st.caption(
    "Known limitations: some section titles for certain filers may contain minor "
    "formatting artifacts (documented in the README) — this affects citation "
    "labels only, not the accuracy of the retrieved content itself."
)
