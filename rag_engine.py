"""
RAG Engine for OAGO Audit Reports
Handles: PDF ingestion → chunking → embedding → vector search → LLM generation
"""

import fitz  # PyMuPDF
import numpy as np
import faiss
import anthropic
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGEngine:
    """
    Core RAG engine. All state lives in memory (session-scoped).
    Embeddings: sentence-transformers/all-MiniLM-L6-v2 (free, local)
    Generation: Claude Haiku via Anthropic API (low cost)
    Vector store: FAISS (in-memory)
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    CLAUDE_MODEL = "claude-haiku-4-5-20251001"
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
    TOP_K = 6

    def __init__(self, api_key: str, embed_model: SentenceTransformer):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.embed_model = embed_model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
        )
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.index = None

    # ── Ingestion ────────────────────────────────────────────────────────────

    def add_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        """
        Extract text from a PDF, chunk it, embed it, and add to FAISS index.
        Returns the number of new chunks added.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        raw_chunks, raw_meta = [], []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                continue
            page_chunks = self.splitter.split_text(text)
            for chunk in page_chunks:
                raw_chunks.append(chunk)
                raw_meta.append({"filename": filename, "page": page_num})

        if not raw_chunks:
            return 0

        self.chunks.extend(raw_chunks)
        self.metadata.extend(raw_meta)
        self._rebuild_index()
        return len(raw_chunks)

    def _rebuild_index(self):
        if not self.chunks:
            return
        embeddings = self.embed_model.encode(self.chunks, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Return top-k relevant chunks with metadata and similarity score."""
        if self.index is None or not self.chunks:
            return []
        k = top_k or self.TOP_K
        vec = self.embed_model.encode([query]).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "score": float(score),
                    }
                )
        return results

    # ── Q&A ──────────────────────────────────────────────────────────────────

    def answer(self, query: str) -> tuple[str, list[dict]]:
        """Answer a question using retrieved context. Returns (answer, sources)."""
        sources = self.search(query)
        if not sources:
            return (
                "⚠️ No reports are loaded. Please upload PDFs in the Upload tab first.",
                [],
            )

        context_blocks = []
        for r in sources:
            context_blocks.append(
                f"[Source: {r['metadata']['filename']}, Page {r['metadata']['page']}]\n{r['chunk']}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a professional IT and performance auditor assisting colleagues at the Office of the Auditor General of Ontario (OAG Ontario).

Answer the following question using ONLY the context provided below from audit reports.

Guidelines:
- Be precise, concise, and professional
- Always cite your sources by report name and page number (e.g., "According to [Report Name], Page X...")
- Use bullet points for multiple findings
- If the context does not contain enough information to answer, say so clearly

---
CONTEXT:
{context}
---

QUESTION: {query}

ANSWER:"""

        response = self.client.messages.create(
            model=self.CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text, sources

    # ── Summarize ────────────────────────────────────────────────────────────

    def summarize(self, filename: str) -> str:
        """Generate a structured summary of a specific report."""
        file_chunks = [
            c
            for c, m in zip(self.chunks, self.metadata)
            if m["filename"] == filename
        ]
        if not file_chunks:
            return "Report content not found."

        # Use first ~20 chunks (representative of full report)
        combined = "\n\n".join(file_chunks[:20])

        prompt = f"""You are an expert auditor. Produce a structured professional summary of the following audit report content.

Format your response with these sections:
## Audit Objective
## Scope & Period
## Key Findings
(bullet points — include risk ratings if mentioned)
## Recommendations
(numbered list)
## Management Response Summary
(if mentioned)
## Overall Assessment

Report content:
{combined}

Summary:"""

        response = self.client.messages.create(
            model=self.CLAUDE_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    # ── Helpers ──────────────────────────────────────────────────────────────

    def loaded_files(self) -> list[str]:
        return sorted(set(m["filename"] for m in self.metadata))

    def total_chunks(self) -> int:
        return len(self.chunks)

    def remove_file(self, filename: str):
        """Remove a specific file from the index."""
        keep = [(c, m) for c, m in zip(self.chunks, self.metadata) if m["filename"] != filename]
        if keep:
            self.chunks, self.metadata = zip(*keep)
            self.chunks = list(self.chunks)
            self.metadata = list(self.metadata)
        else:
            self.chunks, self.metadata = [], []
        self._rebuild_index()
