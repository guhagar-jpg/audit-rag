"""
OAGO Audit Intelligence — Streamlit App
A RAG-powered tool for searching and querying OAG Ontario audit reports.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from rag_engine import RAGEngine

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OAGO Audit Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Background */
    [data-testid="stAppViewContainer"] { background-color: #f4f6f9; }
    [data-testid="stSidebar"] { background-color: #1a3a6b; color: white; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] input { color: #1a1a1a !important; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a3a6b 0%, #2a5298 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .app-header p { color: #c8d8f0; margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* Cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        text-align: center;
        border-top: 3px solid #2a5298;
    }
    .stat-card .number { font-size: 1.8rem; font-weight: 700; color: #1a3a6b; }
    .stat-card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Source cards */
    .source-card {
        background: white;
        border-left: 4px solid #2a5298;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        font-size: 0.85rem;
    }
    .source-card .source-label {
        font-weight: 600;
        color: #1a3a6b;
        margin-bottom: 0.3rem;
    }
    .source-card .source-text { color: #444; line-height: 1.5; }
    .badge {
        background: #e8f0fb;
        color: #2a5298;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Answer box */
    .answer-box {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 5px solid #27ae60;
    }

    /* Search result rows */
    .search-result {
        background: white;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Buttons */
    .stButton > button {
        background-color: #1a3a6b !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #2a5298 !important;
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(26,58,107,0.3);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #1a3a6b !important;
        color: #1a3a6b !important;
    }

    hr { border-color: #e0e7ef; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embed_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your free key at console.anthropic.com",
    )

    if api_key:
        st.success("✓ API key set")
    else:
        st.warning("Enter your API key to enable Q&A and summaries")

    st.markdown("---")
    st.markdown("### 📚 Loaded Reports")

    if "rag" in st.session_state and st.session_state.rag.loaded_files():
        files = st.session_state.rag.loaded_files()
        st.info(f"{len(files)} report(s) • {st.session_state.rag.total_chunks()} chunks indexed")
        for f in files:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {f[:28]}{'...' if len(f) > 28 else ''}")
            if col2.button("✕", key=f"del_{f}", help=f"Remove {f}"):
                st.session_state.rag.remove_file(f)
                if f in st.session_state.get("loaded_files", set()):
                    st.session_state.loaded_files.discard(f)
                st.rerun()
    else:
        st.markdown("*No reports loaded yet*")

    st.markdown("---")
    st.markdown(
        "<small>Built with Streamlit + FAISS + Claude Haiku<br>OAG Ontario Audit Intelligence Pilot</small>",
        unsafe_allow_html=True,
    )

# ── Initialize RAG engine ─────────────────────────────────────────────────────

if api_key:
    if "rag" not in st.session_state or st.session_state.get("current_key") != api_key:
        st.session_state.rag = RAGEngine(api_key, embed_model)
        st.session_state.current_key = api_key

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>🔍 OAGO Audit Intelligence</h1>
        <p>AI-powered search, Q&A, and summarization across OAG Ontario audit reports — Pilot v1.0</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    n_reports = len(st.session_state.rag.loaded_files()) if "rag" in st.session_state else 0
    st.markdown(
        f'<div class="stat-card"><div class="number">{n_reports}</div><div class="label">Reports Loaded</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    n_chunks = st.session_state.rag.total_chunks() if "rag" in st.session_state else 0
    st.markdown(
        f'<div class="stat-card"><div class="number">{n_chunks:,}</div><div class="label">Text Chunks Indexed</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="stat-card"><div class="number">Free</div><div class="label">Hosting Cost</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📂  Upload Reports", "💬  Ask Questions", "📋  Summarize Report"])

# ─── Tab 1: Upload ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Upload Audit Reports")
    st.markdown(
        "Upload PDF audit reports from the [OAG Ontario website](https://www.auditor.on.ca). "
        "Reports will be automatically parsed, chunked, and indexed for search."
    )

    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple reports at once. Each PDF is processed page by page.",
    )

    if uploaded_files:
        if not api_key:
            st.error("⚠️ Please enter your Anthropic API key in the sidebar first.")
        else:
            for file in uploaded_files:
                if file.name not in st.session_state.get("loaded_files", set()):
                    with st.spinner(f"Indexing **{file.name}**..."):
                        n = st.session_state.rag.add_pdf(file.read(), file.name)
                        if "loaded_files" not in st.session_state:
                            st.session_state.loaded_files = set()
                        st.session_state.loaded_files.add(file.name)
                    st.success(f"✅ **{file.name}** — {n} chunks indexed")
                else:
                    st.info(f"ℹ️ **{file.name}** is already loaded.")

    st.markdown("---")
    st.markdown(
        "**Tip:** Download reports directly from "
        "[auditor.on.ca/en/content/annualreports](https://www.auditor.on.ca/en/content/annualreports_en.html) "
        "and upload them here."
    )

# ─── Tab 2: Q&A ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Ask Questions Across Reports")

    if "rag" not in st.session_state or not st.session_state.rag.loaded_files():
        st.info("📂 Upload reports in the **Upload Reports** tab first.")
    else:
        st.markdown(
            f"Searching across **{len(st.session_state.rag.loaded_files())} report(s)** "
            f"with **{st.session_state.rag.total_chunks():,} indexed chunks**."
        )

        # Sample questions
        st.markdown("**Try these questions:**")
        sample_questions = [
            "What were the key IT general control deficiencies identified?",
            "What recommendations were made regarding access controls?",
            "What findings related to change management were reported?",
            "Were there any issues with data integrity or completeness?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.prefill_question = q

        st.markdown("<br>", unsafe_allow_html=True)

        # Query input
        default_q = st.session_state.pop("prefill_question", "")
        question = st.text_area(
            "Your question",
            value=default_q,
            placeholder="e.g. What were the main findings related to privileged access management?",
            height=80,
        )

        col_a, col_b = st.columns([1, 4])
        ask_btn = col_a.button("🔍 Ask", use_container_width=True)

        if ask_btn and question.strip():
            with st.spinner("Retrieving relevant passages and generating answer..."):
                answer, sources = st.session_state.rag.answer(question)

            st.markdown("#### 💡 Answer")
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )

            if sources:
                with st.expander(f"📎 View {len(sources)} source passages used"):
                    for s in sources:
                        relevance_pct = int(s["score"] * 100)
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-label">
                                    📄 {s['metadata']['filename']} — Page {s['metadata']['page']}
                                    <span class="badge">{relevance_pct}% match</span>
                                </div>
                                <div class="source-text">{s['chunk'][:350]}{'...' if len(s['chunk']) > 350 else ''}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ─── Tab 3: Summarize ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Summarize a Report")

    if "rag" not in st.session_state or not st.session_state.rag.loaded_files():
        st.info("📂 Upload reports in the **Upload Reports** tab first.")
    else:
        files = st.session_state.rag.loaded_files()
        selected_file = st.selectbox(
            "Select a report",
            files,
            help="Choose a loaded report to generate a structured AI summary.",
        )

        col1, col2 = st.columns([1, 4])
        summarize_btn = col1.button("📋 Summarize", use_container_width=True)

        if summarize_btn and selected_file:
            with st.spinner(f"Generating summary for **{selected_file}**..."):
                summary = st.session_state.rag.summarize(selected_file)

            st.markdown(f"#### Summary: {selected_file}")
            st.markdown(summary)

            # Download button
            st.download_button(
                label="⬇️ Download Summary (txt)",
                data=f"SUMMARY: {selected_file}\n\n{summary}",
                file_name=f"summary_{selected_file.replace('.pdf', '')}.txt",
                mime="text/plain",
            )
