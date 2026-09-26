# 🤖 RAG Document Q&A

> A polished **Retrieval-Augmented Generation (RAG)** assistant that answers questions across your document library — **grounded, cited, and hallucination-free**.

Built with **Streamlit · Groq · FastEmbed · ChromaDB**, this project lets you index PDFs, TXT, and DOCX files, then ask natural-language questions and receive answers that **only** come from your documents, with citations back to the source page.

---

## ✨ Features

- 💬 **Chat with your documents** — ask anything about your indexed library
- 📄 **Isolated PDF Agent** — upload a PDF in-memory and query it without touching the main index
- 🛠️ **Manage Index** — build, rebuild, or delete the vector index from the UI
- 🧠 **Local embeddings** — FastEmbed (`BAAI/bge-small-en-v1.5`), no API cost for embeddings
- ⚡ **Groq-powered LLM** — blazing-fast inference with `openai/gpt-oss-20b`
- 🗂️ **Persistent vector store** — ChromaDB survives restarts
- 🔒 **Grounded answers only** — every factual claim is cited as `[Source: filename, page X]`
- 🚫 **No hallucinations** — if the answer isn't in your docs, the model says so
- 🎨 **Premium dark UI** — gradient hero, glassmorphism cards, animated pills
- 🧪 **Notebook + Web UI** — full pipeline available as a Jupyter notebook and Streamlit app

---

## 🖼️ Architecture

```
                ┌──────────────────────┐
                │   Source Documents   │
                │  PDF · TXT · DOCX    │
                └──────────┬───────────┘
                           │
                    [Ingestion & Cleaning]
                           │
                    [Chunking: 500/100]
                           │
                ┌──────────▼───────────┐
                │  FastEmbed (bge-small)│
                │   384-d embeddings   │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │      ChromaDB        │
                │  (cosine similarity) │
                └──────────┬───────────┘
                           │
       User Query ──► [Embed query] ──► [Top-K Retrieval]
                           │
                    [Build Grounded Prompt]
                           │
                ┌──────────▼───────────┐
                │   Groq · gpt-oss-20b │
                └──────────┬───────────┘
                           │
                 ✅ Cited Answer + Sources
```

---

## 📁 Project Structure

```
rag-document-qa/
│
├── app2.py                    # 🎨 Streamlit web app (premium UI)
├── RAG DOC.Q&A.ipynb          # 📓 Full notebook: pipeline + ipywidgets UI
├── requirements.txt           # 📦 Python dependencies
├── .env                       # 🔑 GROQ_API_KEY (NOT committed — see below)
├── .env.example               # 🧪 Template for environment variables
├── .gitignore                 # 🚫 Ignore .env, chroma_db/, __pycache__/, etc.
├── README.md                  # 📖 This file
│
├── data/                      # 📚 Put your PDFs / TXT / DOCX here
│   ├── Artificial Intelligence.pdf
│   ├── BUSINESS.pdf
│   ├── LAW.pdf
│   ├── SCIENCE.pdf
│   └── The MCU.pdf
│
├── chroma_db/                 # 🗄️ Persistent ChromaDB (gitignored)
│
└── rag_artifacts.pkl          # 💾 Exported chunks + embeddings (optional)
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key

1. Go to **[console.groq.com/keys](https://console.groq.com/keys)**
2. Sign up (free) and create an API key
3. Copy the key (starts with `gsk_...`)

### 5. Configure environment

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> 💡 Tip: copy `.env.example` and rename it to `.env`.

### 6. Add your documents

Drop your `.pdf`, `.txt`, or `.docx` files into the `data/` folder:

```bash
mkdir -p data
# copy your files here
```

### 7. Run the app

```bash
streamlit run app2.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser. 🎉

---

## 🧭 Using the App

The Streamlit app has **three tabs**:

### 💬 Tab 1 — Chat with Documents
- Ask natural-language questions about your whole indexed library
- Adjust **Top-K** to control how many chunks are retrieved
- Every answer comes with a **Sources** expander showing the exact chunks used

### 📄 Tab 2 — PDF Agent
- Upload a **new PDF** (in-memory only) or pick one from `data/`
- Query it in isolation — **does not** touch the main Chroma index
- Great for one-off document analysis

### 🛠️ Tab 3 — Manage Index
- **Build / Rebuild Index** — re-ingest everything in `data/`
- **Delete Index** — wipe the Chroma collection
- View file sizes and current index stats

---

## 📓 Running the Notebook

The Jupyter notebook `RAG DOC.Q&A.ipynb` walks through the **entire pipeline** step by step:

| Step | Description |
|------|-------------|
| 0 | Install dependencies |
| 1 | Imports & configuration |
| 2 | Sanity-check the Groq API |
| 3 | Document ingestion (PDF/TXT/DOCX) |
| 4 | Text chunking (500 chars / 100 overlap) |
| 5 | Embedding model (FastEmbed) |
| 6 | Vector DB indexing (ChromaDB) |
| 7 | Retrieval (top-K similarity search) |
| 8 | Prompt construction (grounded + cited) |
| 9 | Answer generation (full RAG pipeline) |
| 10 | End-to-end test |
| 11 | CLI loop (optional) |
| 12 | **Bonus:** ipywidgets two-tab UI |
| 13 | Export artifacts to `rag_artifacts.pkl` |

Run it with:

```bash
jupyter notebook "RAG DOC.Q&A.ipynb"
```

---

## ⚙️ Configuration

All settings live at the top of `app2.py` (and the notebook). Tweak as needed:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `openai/gpt-oss-20b` | Groq model for answer generation |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed embedding model (384-d) |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `DATA_DIR` | `data` | Source document folder |
| `CHROMA_DIR` | `chroma_db` | Persistent vector DB folder |
| `COLLECTION_NAME` | `documents` | Chroma collection name |

---

## 🔒 How Grounding Works

This project is designed to **never hallucinate**:

1. **Retrieval-first** — if no chunks are retrieved, the app returns the fixed fallback:
   > *"I don't have enough information in the provided documents to answer this."*
2. **Strict prompt** — the LLM is instructed to:
   - ✅ Answer **only** from the provided context
   - ❌ Never use outside knowledge
   - 📌 Cite every factual claim as `[Source: filename, page X]`
   - 🚫 Never invent information
3. **Low temperature** — `temperature=0.1` keeps outputs deterministic and factual.
4. **Transparent sources** — every answer exposes the retrieved chunks with similarity distances.

---

## 🐳 Docker (Optional)

Want to containerize it? Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app2.py", "--server.address=0.0.0.0"]
```

Build & run:

```bash
docker build -t rag-document-qa .
docker run -p 8501:8501 --env-file .env rag-document-qa
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **New app** → select your repo → set main file to `app2.py`
4. In **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

5. Click **Deploy** 🚀

> ⚠️ **Note:** Streamlit Cloud has an ephemeral filesystem — `chroma_db/` won't persist between restarts. For production, use a hosted vector DB (Pinecone, Weaviate, Qdrant Cloud).

---

## 🧪 Example Queries

Once the sample docs are indexed, try:

- *"What is the main topic of the documents?"*
- *"Summarize the key challenges of digital transformation."*
- *"What does the fair use doctrine permit?"*
- *"How much has atmospheric CO₂ increased since the Industrial Revolution?"*
- *"Which phase of the MCU introduced the multiverse?"*
- *"What are the ethical concerns around generative AI?"*

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not found` | Create `.env` with your key (step 5) |
| `No documents found in data/` | Add PDFs/TXTs/DOCX to `data/` |
| `No documents indexed yet` | Go to **Manage Index** tab → click **Build / Rebuild Index** |
| ChromaDB errors on rebuild | Delete `chroma_db/` folder and rebuild |
| Slow first run | First embedding run downloads the model (~130 MB) — subsequent runs are cached |
| `streamlit: command not found` | Activate your virtualenv, or `pip install streamlit` |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit, ipywidgets |
| **LLM** | [Groq](https://groq.com) — `openai/gpt-oss-20b` |
| **Embeddings** | [FastEmbed](https://github.com/qdrant/fastembed) — `BAAI/bge-small-en-v1.5` |
| **Vector DB** | [ChromaDB](https://www.trychroma.com) |
| **Document loaders** | pypdf, python-docx |
| **Config** | python-dotenv |
| **Numerics** | NumPy |

---

## 🔐 Security Notes

- **Never commit your `.env` file** — it's already in `.gitignore`
- Use `.env.example` as a safe template
- If you accidentally leak your Groq key, **revoke it immediately** at [console.groq.com/keys](https://console.groq.com/keys)
- Uploaded PDFs in the **PDF Agent** tab are processed **in-memory only** — nothing is written to disk

---

## 📌 `.gitignore` (recommended)

```gitignore
# Environment
.env
.env.local
*.env

# Vector DB (regenerate locally)
chroma_db/
*.sqlite3

# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.venv/
env/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Model cache
.cache/
fastembed_cache/
```

---

## 📌 `.env.example`

```bash
# Get a free key at https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🗺️ Roadmap

- [ ] Multi-collection support (per-user knowledge bases)
- [ ] Hybrid search (BM25 + dense) — `rank-bm25` already in requirements
- [ ] Streaming answers (token-by-token)
- [ ] Conversation memory across turns
- [ ] Reranking with a cross-encoder
- [ ] Support for `.md`, `.csv`, `.html`
- [ ] Cloud vector DB adapter (Pinecone / Qdrant)
- [ ] Auth + per-user document isolation

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) for blazing-fast LLM inference
- [FastEmbed](https://github.com/qdrant/fastembed) for local embeddings
- [ChromaDB](https://www.trychroma.com) for the vector store
- [Streamlit](https://streamlit.io) for the UI framework

---

## 📬 Contact:9390499617

**Your Name** — [Pasala Pratheep](https://github.com/your-handle)

Project link: [https://github.com/your-username/rag-document-qa]([https://github.com/your-username/rag-document-qa](https://github.com/pasalapratheep-08/12.RAG-DOCUMENTED-Q-A-))

---

<div align="center">

**Built with ❤️ using Streamlit · Groq · FastEmbed · ChromaDB**

⭐ If this project helped you, consider giving it a star!

</div>
