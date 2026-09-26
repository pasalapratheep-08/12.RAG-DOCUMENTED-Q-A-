"""
🤖 RAG Document Q&A — Premium Streamlit App
============================================
A polished Retrieval-Augmented Generation assistant powered by:
  • Groq (openai/gpt-oss-20b)   → LLM generation
  • FastEmbed (BAAI/bge-small-en-v1.5) → embeddings
  • ChromaDB                     → vector store
  • PyPDF / python-docx          → document loading
"""

import os
import re
import io
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Document loaders
from pypdf import PdfReader
from docx import Document as DocxDocument

# Vector DB
import chromadb

# Embeddings
from fastembed import TextEmbedding

# LLM
from groq import Groq

# ════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════
load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
LLM_MODEL       = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
TOP_K           = 5
DATA_DIR        = "data"
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "documents"

# ════════════════════════════════════════════════════════════
# CUSTOM CSS  — Premium dark-gradient look
# ════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: radial-gradient(circle at 0% 0%, #1a1f3a 0%, #0e1225 40%, #070a17 100%);
    color: #e6e9f0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 24px;
    padding: 2.2rem 2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 20px 60px rgba(102,126,234,0.35);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    position: relative; z-index: 1;
}
.hero p {
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
    color: rgba(255,255,255,0.9);
    font-weight: 400;
    position: relative; z-index: 1;
}
.hero .badges {
    margin-top: 1rem;
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    position: relative; z-index: 1;
}
.hero .badge {
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    color: #fff;
    font-weight: 500;
}

/* ---------- Cards ---------- */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
}
.card:hover {
    border-color: rgba(102,126,234,0.5);
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(102,126,234,0.15);
}

/* ---------- Source pill ---------- */
.source-pill {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}
.distance-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    color: #a5b4fc;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
}

/* ---------- Chat messages ---------- */
.chat-user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    padding: 1rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.6rem 0 0.6rem auto;
    max-width: 78%;
    box-shadow: 0 8px 24px rgba(102,126,234,0.3);
    font-size: 0.95rem;
    line-height: 1.5;
}
.chat-bot {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.09);
    color: #e6e9f0;
    padding: 1rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.6rem auto 0.6rem 0;
    max-width: 82%;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(102,126,234,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(102,126,234,0.55);
    color: #fff;
}
.stButton > button:active {transform: translateY(0);}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1225 0%, #131829 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5b4fc;
    font-weight: 700;
    font-size: 1rem;
    margin-top: 1.2rem;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: rgba(255,255,255,0.03);
    padding: 0.4rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: #9aa3b8;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(102,126,234,0.4);
}

/* ---------- Inputs ---------- */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e6e9f0 !important;
    font-size: 0.95rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important;
}

/* ---------- Expander ---------- */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #c7d0e0 !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 0 0 10px 10px !important;
}

/* ---------- Metric ---------- */
[data-testid="stMetricValue"] {
    color: #a5b4fc !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #9aa3b8 !important;
    font-size: 0.8rem !important;
}

/* ---------- Divider ---------- */
hr {border-color: rgba(255,255,255,0.08);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# CORE FUNCTIONS  (ported from notebook)
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_embedder():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def embed_texts(texts: List[str]) -> np.ndarray:
    embedder = get_embedder()
    embeddings = list(embedder.embed(texts))
    return np.array(embeddings, dtype=np.float32)


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks, start = [], 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


# ---------- Document loaders ----------
def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text.strip():
            pages.append({"text": text, "source": Path(file_path).name, "page": i})
    return pages


def load_txt(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = clean_text(f.read())
    return [{"text": text, "source": Path(file_path).name, "page": 1}]


def load_docx(file_path: str) -> List[Dict[str, Any]]:
    doc = DocxDocument(file_path)
    text = clean_text("\n".join(p.text for p in doc.paragraphs))
    return [{"text": text, "source": Path(file_path).name, "page": 1}]


def load_documents(folder: str) -> List[Dict[str, Any]]:
    docs = []
    folder_path = Path(folder)
    if not folder_path.exists():
        return docs
    for file in sorted(folder_path.iterdir()):
        ext = file.suffix.lower()
        try:
            if ext == ".pdf":
                docs.extend(load_pdf(str(file)))
            elif ext == ".txt":
                docs.extend(load_txt(str(file)))
            elif ext == ".docx":
                docs.extend(load_docx(str(file)))
        except Exception as e:
            st.warning(f"⚠️ Could not load {file.name}: {e}")
    return docs


def build_chunks(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append({
                "text":     chunk,
                "source":   doc["source"],
                "page":     doc["page"],
                "chunk_id": f"{doc['source']}_p{doc['page']}_c{i}",
            })
    return all_chunks


# ---------- Indexing ----------
def index_chunks(chunks: List[Dict[str, Any]], reset: bool = True):
    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    if not texts:
        return collection

    embeddings = embed_texts(texts)
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return collection


# ---------- Retrieval ----------
def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []
    if collection.count() == 0:
        return []

    q_emb = embed_texts([query])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        dist = dists[i] if i < len(dists) else None
        out.append({
            "text":     doc,
            "source":   meta.get("source", "Unknown"),
            "page":     meta.get("page", "Unknown"),
            "distance": dist,
        })
    return out


# ---------- Prompt ----------
def build_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return (
            "You are a document question-answering assistant.\n\n"
            "The document database returned no relevant information.\n\n"
            f"User question:\n{query}\n\n"
            "Answer exactly:\n"
            "I don't have enough information in the provided documents to answer this.\n"
        )

    blocks = []
    for i, item in enumerate(retrieved, start=1):
        blocks.append(
            f"[{i}]\n"
            f"Source: {item.get('source','Unknown')}\n"
            f"Page: {item.get('page','Unknown')}\n\n"
            f"{item.get('text','')}\n"
        )
    context = "\n".join(blocks)

    return f"""You are a grounded document Q&A assistant.

Your job is to answer the user's question using ONLY the information contained in the provided context.

IMPORTANT RULES:
1. Do NOT use outside knowledge.
2. Do NOT invent information.
3. Do NOT make assumptions.
4. If the answer is not present in the context, respond exactly:
   I don't have enough information in the provided documents to answer this.
5. Every factual claim must include a citation.
6. Citation format: [Source: filename, page X]
7. Keep the answer concise and factual.
8. If multiple sources support an answer, cite each relevant source.
9. Preserve important numbers, names, dates and technical terms.

--------------------------------------------------
CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{query}

--------------------------------------------------
ANSWER
--------------------------------------------------
"""


# ---------- Answer generation ----------
def generate_answer(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    client = get_groq_client()
    if client is None:
        return {"answer": "❌ GROQ_API_KEY not configured.", "sources": []}

    retrieved = retrieve(query=query, top_k=top_k)
    if not retrieved:
        return {
            "answer": "I don't have enough information in the provided documents to answer this.",
            "sources": [],
        }

    prompt = build_prompt(query=query, retrieved=retrieved)
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a factual RAG assistant. Only answer using the supplied document context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        return {"answer": f"❌ Groq API error ({type(e).__name__}): {e}", "sources": retrieved}

    return {"answer": answer, "sources": retrieved}


# ---------- In-memory PDF agent ----------
def load_pdf_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text.strip():
            pages.append({"text": text, "source": filename, "page": i})

    chunks = []
    for p in pages:
        for j, ch in enumerate(chunk_text(p["text"])):
            chunks.append({
                "text":     ch,
                "source":   p["source"],
                "page":     p["page"],
                "chunk_id": f"{p['source']}_p{p['page']}_c{j}",
            })

    embs = embed_texts([c["text"] for c in chunks]) if chunks else np.zeros((0, 384), dtype=np.float32)
    return {"chunks": chunks, "embeddings": embs, "name": filename}


def retrieve_in_pdf(query: str, state: Dict[str, Any], top_k: int = 4) -> List[Dict[str, Any]]:
    embs = state.get("embeddings")
    chunks = state.get("chunks", [])
    if embs is None or len(chunks) == 0:
        return []

    q_vec = embed_texts([query])[0]
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    d_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    sims = d_norm @ q_norm

    top_idx = np.argsort(-sims)[:top_k]
    out = []
    for idx in top_idx:
        c = chunks[int(idx)]
        out.append({
            "text":     c["text"],
            "source":   c["source"],
            "page":     c["page"],
            "distance": float(1.0 - sims[int(idx)]),
        })
    return out


# ════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════
def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🤖 RAG Document Q&A</h1>
        <p>Ask questions across your document library — grounded, cited, and hallucination-free.</p>
        <div class="badges">
            <span class="badge">⚡ Groq · gpt-oss-20b</span>
            <span class="badge">🧠 FastEmbed · bge-small-en-v1.5</span>
            <span class="badge">🗂️ ChromaDB</span>
            <span class="badge">🔒 Grounded answers only</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources: List[Dict[str, Any]], expanded: bool = False):
    if not sources:
        return
    with st.expander(f"📚 Retrieved Sources ({len(sources)})", expanded=expanded):
        for i, s in enumerate(sources, start=1):
            d = s.get("distance")
            d_text = f"{d:.4f}" if isinstance(d, (int, float)) else "N/A"
            st.markdown(
                f'<span class="source-pill">[{i}] {s["source"]}</span>'
                f'<span class="source-pill">page {s["page"]}</span>'
                f'<span class="distance-pill">distance {d_text}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='card'>{s['text'][:600]}{'...' if len(s['text']) > 600 else ''}</div>",
                unsafe_allow_html=True,
            )


def render_chat_message(role: str, content: str):
    cls = "chat-user" if role == "user" else "chat-bot"
    st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)


def sidebar_stats():
    with st.sidebar:
        st.markdown("## 📊 Knowledge Base")
        try:
            client = get_chroma_client()
            collection = client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            st.metric("Indexed chunks", count)
        except Exception:
            count = 0
            st.metric("Indexed chunks", 0)

        data_path = Path(DATA_DIR)
        files = []
        if data_path.exists():
            files = [f for f in data_path.iterdir()
                     if f.suffix.lower() in {".pdf", ".txt", ".docx"}]
        st.metric("Source documents", len(files))

        if files:
            st.markdown("### 📁 Files")
            for f in sorted(files):
                st.markdown(f"- `{f.name}`")

        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.markdown(f"- **LLM:** `{LLM_MODEL}`")
        st.markdown(f"- **Embeddings:** `{EMBEDDING_MODEL}`")
        st.markdown(f"- **Chunk size:** `{CHUNK_SIZE}`")
        st.markdown(f"- **Top-K:** `{TOP_K}`")

        if not GROQ_API_KEY:
            st.error("🔑 GROQ_API_KEY missing in .env")

        return count


# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_agent_state" not in st.session_state:
    st.session_state.pdf_agent_state = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
render_hero()
chunk_count = sidebar_stats()

tab_chat, tab_pdf, tab_manage = st.tabs(
    ["💬 Chat with Documents", "📄 PDF Agent", "🛠️ Manage Index"]
)

# ────────────────────────────────────────────────────────────
# TAB 1 — MAIN RAG CHAT
# ────────────────────────────────────────────────────────────
with tab_chat:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 💬 Ask anything about your indexed documents")
    with col2:
        top_k = st.slider("Top-K", 1, 8, TOP_K, key="chat_topk")

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            render_chat_message(msg["role"], msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    # Input row
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([8, 1, 1])
        with cols[0]:
            query = st.text_input(
                "Your question",
                placeholder="e.g. What is the main topic of the documents?",
                label_visibility="collapsed",
            )
        with cols[1]:
            submit = st.form_submit_button("🚀 Ask", use_container_width=True)
        with cols[2]:
            clear = st.form_submit_button("🧹 Clear", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if submit and query.strip():
        if chunk_count == 0:
            st.warning("⚠️ No documents indexed yet. Go to **Manage Index** tab and click *Build Index*.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": query})

            with st.spinner("🔍 Retrieving context and generating answer..."):
                result = generate_answer(query, top_k=top_k)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
            st.rerun()


# ────────────────────────────────────────────────────────────
# TAB 2 — PDF AGENT
# ────────────────────────────────────────────────────────────
with tab_pdf:
    st.markdown("### 📄 Isolated PDF Q&A")
    st.caption("Upload or pick a PDF — it's embedded in-memory, isolated from the main index.")

    col_a, col_b = st.columns([2, 2])
    with col_a:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_upload")
    with col_b:
        data_path = Path(DATA_DIR)
        existing = [f.name for f in data_path.glob("*.pdf")] if data_path.exists() else []
        picked = st.selectbox("Or pick from `data/`", ["— select —"] + existing, key="pdf_pick")

    col_load, col_clear = st.columns([1, 1])
    with col_load:
        load_clicked = st.button("📥 Load PDF", use_container_width=True, key="pdf_load")
    with col_clear:
        if st.button("🗑️ Unload", use_container_width=True, key="pdf_unload"):
            st.session_state.pdf_agent_state = None
            st.rerun()

    if load_clicked:
        try:
            if uploaded is not None:
                state = load_pdf_bytes(uploaded.read(), uploaded.name)
            elif picked and picked != "— select —":
                with open(data_path / picked, "rb") as f:
                    state = load_pdf_bytes(f.read(), picked)
            else:
                st.warning("⚠️ Upload a PDF or pick one from `data/`.")
                state = None

            if state:
                st.session_state.pdf_agent_state = state
                st.success(f"✅ Loaded **{state['name']}** — {len(state['chunks'])} chunks indexed.")
        except Exception as e:
            st.error(f"❌ Failed to load PDF: {e}")

    state = st.session_state.pdf_agent_state
    if state:
        st.info(f"📄 Active PDF: **{state['name']}** — {len(state['chunks'])} chunks")

        with st.form("pdf_form", clear_on_submit=True):
            cols = st.columns([6, 1, 1])
            with cols[0]:
                pdf_q = st.text_input(
                    "Ask about this PDF",
                    placeholder="e.g. What is the main argument?",
                    label_visibility="collapsed",
                )
            with cols[1]:
                pdf_submit = st.form_submit_button("🚀 Ask", use_container_width=True)
            with cols[2]:
                pdf_clear = st.form_submit_button("🧹 Clear", use_container_width=True)

        if pdf_submit and pdf_q.strip():
            with st.spinner("🔍 Searching PDF..."):
                retrieved = retrieve_in_pdf(pdf_q, state, top_k=4)
                prompt = build_prompt(pdf_q, retrieved)
                client = get_groq_client()
                try:
                    completion = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system",
                             "content": "You are a factual RAG assistant that only answers from provided context."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=600,
                    )
                    answer = completion.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"❌ Groq error: {e}"

            st.markdown("#### 💡 Answer")
            st.markdown(f"<div class='card'>{answer}</div>", unsafe_allow_html=True)
            render_sources(retrieved, expanded=True)
    else:
        st.info("Upload or select a PDF to begin.")


# ────────────────────────────────────────────────────────────
# TAB 3 — MANAGE INDEX
# ────────────────────────────────────────────────────────────
with tab_manage:
    st.markdown("### 🛠️ Manage the Knowledge Base")
    st.caption("Load documents from `data/`, chunk, embed, and index them into ChromaDB.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⚡ Build / Rebuild Index", use_container_width=True, key="build_idx"):
            with st.spinner("📚 Loading documents..."):
                docs = load_documents(DATA_DIR)
            if not docs:
                st.error(f"❌ No documents found in `{DATA_DIR}/`. Add PDFs/TXTs/DOCX files.")
            else:
                with st.spinner(f"✂️ Chunking {len(docs)} pages..."):
                    chunks = build_chunks(docs)
                with st.spinner(f"🧠 Embedding {len(chunks)} chunks..."):
                    index_chunks(chunks, reset=True)
                st.session_state.indexed = True
                st.success(f"✅ Indexed {len(chunks)} chunks from {len(docs)} pages.")
                st.rerun()

    with col2:
        if st.button("🗑️ Delete Index", use_container_width=True, key="del_idx"):
            try:
                client = get_chroma_client()
                client.delete_collection(COLLECTION_NAME)
                st.success("✅ Index deleted.")
                st.rerun()
            except Exception as e:
                st.warning(f"Nothing to delete: {e}")

    st.markdown("---")
    st.markdown("#### 📊 Current Index")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Chunks", chunk_count)
    data_path = Path(DATA_DIR)
    n_files = len([f for f in data_path.iterdir()
                   if f.suffix.lower() in {".pdf", ".txt", ".docx"}]) if data_path.exists() else 0
    col_b.metric("Documents", n_files)
    col_c.metric("Embed dim", 384)

    st.markdown("---")
    st.markdown("#### 📁 Files in `data/`")
    if data_path.exists():
        files = sorted([f for f in data_path.iterdir()
                        if f.suffix.lower() in {".pdf", ".txt", ".docx"}])
        if files:
            for f in files:
                size_kb = f.stat().st_size / 1024
                st.markdown(f"- 📄 `{f.name}` — {size_kb:.1f} KB")
        else:
            st.info("No documents yet. Drop PDFs/TXTs/DOCX into the `data/` folder.")
    else:
        st.warning(f"`{DATA_DIR}/` folder doesn't exist. Create it and add documents.")


# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown(
    "<div style='text-align:center; color:#5a6a85; font-size:0.8rem; margin-top:3rem;'>"
    "Built with ❤️ using Streamlit · Groq · FastEmbed · ChromaDB"
    "</div>",
    unsafe_allow_html=True,
)