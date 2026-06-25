"""
app.py
Streamlit UI: upload PDFs -> ingest (text + images) -> chat with them.
Run with: streamlit run app.py
"""

import os
from pathlib import Path

_hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
_model_cached = any(_hf_cache.glob("models--sentence-transformers--all-MiniLM-L6-v2*"))
if _model_cached:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

load_dotenv()

STORE_PATH = "data/store"

st.set_page_config(page_title="MultiModal RAG - Chat with PDFs", page_icon="📄", layout="wide")

# ---------- Session state ----------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_pipeline() -> RAGPipeline:
    if st.session_state.pipeline is None:
        api_key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key")
        if not api_key:
            st.error("No Groq API key found. Enter it in the sidebar or add it to the .env file.")
            st.stop()
        try:
            st.session_state.pipeline = RAGPipeline(api_key)
        except Exception as exc:
            st.error(f"Failed to initialize the pipeline: {exc}")
            st.stop()
        # load any previously persisted index
        if os.path.exists(os.path.join(STORE_PATH, "index.faiss")):
            st.session_state.pipeline.load(STORE_PATH)
    return st.session_state.pipeline


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📄 MultiModal RAG")
    st.caption("Chat with PDFs — text + images")

    if not os.getenv("GROQ_API_KEY"):
        key_input = st.text_input("Groq API Key", type="password", value=st.session_state.get("groq_api_key", ""))
        if key_input:
            st.session_state.groq_api_key = key_input

    st.divider()
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDFs", type="primary", use_container_width=True):
        pipeline = get_pipeline()
        status = st.empty()

        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.ingested_files:
                status.info(f"Skipping {uploaded_file.name} (already ingested).")
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            def progress(msg, _name=uploaded_file.name):
                status.info(f"**{_name}**: {msg}")

            n_text, n_img = pipeline.ingest_pdf(tmp_path, uploaded_file.name, progress_callback=progress)
            os.remove(tmp_path)
            st.session_state.ingested_files.append(uploaded_file.name)
            status.success(f"✅ {uploaded_file.name}: {n_text} text chunks, {n_img} images indexed.")

        pipeline.save(STORE_PATH)

    st.divider()
    if st.session_state.ingested_files:
        st.subheader("Indexed documents")
        for f in st.session_state.ingested_files:
            st.write(f"📄 {f}")

        if st.button("Clear all documents", use_container_width=True):
            st.session_state.pipeline = None
            st.session_state.ingested_files = []
            st.session_state.messages = []
            import shutil
            if os.path.exists(STORE_PATH):
                shutil.rmtree(STORE_PATH)
            st.rerun()
    else:
        st.caption("No documents indexed yet.")

# ---------- Main chat area ----------
st.header("Chat with your PDFs")

if not st.session_state.ingested_files:
    st.info("Upload and process at least one PDF from the sidebar to start chatting.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    icon = "🖼️" if s["type"] == "image" else "📝"
                    st.markdown(f"{icon} **{s['source_file']}**, page {s['page']} (score: {s['score']:.2f})")
                    st.caption(s["content"][:300] + ("..." if len(s["content"]) > 300 else ""))

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    pipeline = get_pipeline()
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = pipeline.answer(prompt)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources used"):
                for s in result["sources"]:
                    icon = "🖼️" if s["type"] == "image" else "📝"
                    st.markdown(f"{icon} **{s['source_file']}**, page {s['page']} (score: {s['score']:.2f})")
                    st.caption(s["content"][:300] + ("..." if len(s["content"]) > 300 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
