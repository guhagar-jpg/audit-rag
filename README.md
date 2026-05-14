# OAGO Audit Intelligence — Pilot v1.0

AI-powered search, Q&A, and summarization across OAG Ontario audit reports.

**Stack:** Streamlit · PyMuPDF · FAISS · sentence-transformers · Claude Haiku API  
**Cost:** Free (Streamlit Community Cloud hosting + minimal Claude API costs)

---

## Features

| Feature | Description |
|---|---|
| 📂 Upload | Upload OAG Ontario audit PDFs; auto-parsed and indexed |
| 💬 Q&A | Ask natural language questions across all loaded reports |
| 📋 Summarize | Generate structured summaries with key findings & recommendations |
| 📎 Citations | Every answer cites the source report and page number |

---

## Deploy to Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: OAGO Audit Intelligence"
git remote add origin https://github.com/YOUR_USERNAME/oago-rag.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app** → select your repo → set main file: `app.py`
4. Click **Deploy**

### Step 3 — Set your API Key (Secrets)
In Streamlit Cloud → your app → **Settings → Secrets**, add:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```
Or users can enter it directly in the app sidebar.

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Getting Reports

Download audit reports from:  
👉 https://www.auditor.on.ca/en/content/annualreports_en.html

---

## Architecture

```
PDF Upload
    └── PyMuPDF (text extraction, page-by-page)
        └── LangChain TextSplitter (600-token chunks, 100 overlap)
            └── sentence-transformers (all-MiniLM-L6-v2 embeddings)
                └── FAISS (cosine similarity search, in-memory)
                    └── Claude Haiku (answer generation with citations)
```

---

## Notes

- The embedding model (`all-MiniLM-L6-v2`) downloads ~90MB on first startup — subsequent loads are cached.
- All data is session-scoped (refreshing the page clears loaded reports). For persistence, consider adding a SQLite or Chroma backend.
- To upgrade answer quality, change `CLAUDE_MODEL` in `rag_engine.py` to `claude-sonnet-4-6`.
