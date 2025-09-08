
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
import io, hashlib, csv
import pandas as pd
import re
from typing import List, Dict, Any

# ---------------------- Utility functions ----------------------
@st.cache_resource
def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the sentence-transformers model."""
    return SentenceTransformer(model_name)


def file_md5(file_bytes: bytes) -> str:
    m = hashlib.md5()
    m.update(file_bytes)
    return m.hexdigest()


def extract_text_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Return list of dicts: [{'page': 1, 'text': '...'}, ...]"""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append({"page": i + 1, "text": txt.strip()})
    return pages


def split_text_into_chunks(pages: List[Dict[str, Any]], max_chunk_chars: int = 900) -> List[Dict[str, Any]]:
    """Split text into chunks that roughly represent clauses/paragraphs.

    Strategy:
    - Split by double-newline paragraphs first
    - If a paragraph is longer than max_chunk_chars, split into sentence windows
    - Keep page number for context
    """
    chunks = []
    sentence_split_re = re.compile(r'(?<=[\.!?])\s+')

    for p in pages:
        text = p.get("text", "")
        if not text:
            continue
        # preliminary paragraph split
        paras = re.split(r'\n\s*\n', text)
        for para in paras:
            para = para.strip()
            if not para:
                continue
            if len(para) <= max_chunk_chars:
                chunks.append({"page": p["page"], "text": para})
            else:
                # split to sentences and create sliding windows
                sents = sentence_split_re.split(para)
                cur = ""
                for s in sents:
                    s = s.strip()
                    if not s:
                        continue
                    if len(cur) + len(s) + 1 <= max_chunk_chars:
                        cur = (cur + " " + s).strip()
                    else:
                        if cur:
                            chunks.append({"page": p["page"], "text": cur})
                        cur = s
                if cur:
                    chunks.append({"page": p["page"], "text": cur})
    return chunks


@st.cache_data(show_spinner=False)
def embed_texts_numpy(model_name: str, texts: List[str]) -> np.ndarray:
    """Return numpy array of normalized embeddings for a list of texts.

    We cache embeddings by model_name + texts hash inside Streamlit cache.
    """
    model = load_model(model_name)
    # model.encode returns numpy when convert_to_tensor=False
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)
    # Ensure float32 numpy array
    return np.array(embeddings, dtype=np.float32)


def compute_cosine_scores(query_emb: np.ndarray, chunk_embs: np.ndarray) -> np.ndarray:
    """Assumes both query_emb and chunk_embs are L2-normalized. Returns 1D array of scores."""
    # If shapes: chunk_embs (n, d), query_emb (d,) -> dot product
    return np.dot(chunk_embs, query_emb)


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="AI Legal Document Analyzer", layout="wide")
st.title("AI-powered Legal Document Analyzer")
st.write(
    "Upload a PDF contract and search it. The app returns clauses (chunks) from your contract it thinks are relevant to your query."
)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Embedding model", options=["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1"], index=0)
    max_chunk_chars = st.number_input("Max chunk characters", min_value=200, max_value=4000, value=900, step=50)
    default_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.38, step=0.01)
    top_k = st.number_input("Max results to show", min_value=1, max_value=200, value=20)
    st.markdown("---")
    st.markdown(
        "Tips:\n- Raise threshold to show fewer, more relevant results.\n- If PDF is scanned, run OCR first (not included here)."
    )

# File uploader
uploaded_file = st.file_uploader("Upload a PDF contract", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_hash = file_md5(file_bytes)
    st.write(f"File: **{uploaded_file.name}** — size {len(file_bytes)/1024:.1f} KB")

    # Display small preview and page count
    try:
        pages = extract_text_pages_from_pdf_bytes(file_bytes)
        total_pages = len(pages)
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        pages = []
        total_pages = 0

    st.write(f"Pages detected: **{total_pages}**")

    if total_pages == 0 or all(len(p.get("text",""))==0 for p in pages):
        st.warning(
            "This PDF appears to contain little or no extractable text. If this is a scanned document, you'll need to OCR it before using this app."
        )
    else:
        # Chunk and embed (cache by file_hash + model_name + max_chunk_chars)
        cache_key = f"chunks_{file_hash}_{model_name}_{max_chunk_chars}"
        if cache_key in st.session_state:
            chunks = st.session_state[cache_key]["chunks"]
            chunk_embs = st.session_state[cache_key]["embeddings"]
        else:
            with st.spinner("Splitting document into chunks..."):
                chunks = split_text_into_chunks(pages, max_chunk_chars=max_chunk_chars)
            if len(chunks) == 0:
                st.warning("No textual chunks were produced from this document.")
                chunks = []
                chunk_embs = np.zeros((0, 384), dtype=np.float32)
            else:
                with st.spinner("Embedding document chunks (this may take a moment)..."):
                    texts = [c["text"] for c in chunks]
                    chunk_embs = embed_texts_numpy(model_name, texts)
            # store in session state
            st.session_state[cache_key] = {"chunks": chunks, "embeddings": chunk_embs}

        st.success(f"Document processed into {len(chunks)} chunks.")

        # Query input
        st.subheader("Search the contract")
        query = st.text_area("Enter search query or question (example: 'payment terms', 'termination clause', 'liability')", height=120)
        threshold = st.slider("Filter threshold (cosine)", 0.0, 1.0, default_threshold, step=0.01)
        k = int(top_k)

        if st.button("Find matching clauses"):
            if not query or len(query.strip()) == 0:
                st.warning("Please enter a query first.")
            elif len(chunks) == 0:
                st.warning("No chunks to search. Make sure the PDF contains text.")
            else:
                with st.spinner("Computing query embedding and searching..."):
                    model = load_model(model_name)
                    q_emb = model.encode(query, convert_to_tensor=False, normalize_embeddings=True)
                    # ensure numpy
                    q_emb = np.array(q_emb, dtype=np.float32)
                    # ensure chunk_embs available
                    chunk_embs = st.session_state[cache_key]["embeddings"]
                    if chunk_embs is None or chunk_embs.shape[0] == 0:
                        st.warning("No embeddings available to search.")
                    else:
                        scores = compute_cosine_scores(q_emb, chunk_embs)  # dot product; embeddings are normalized
                        # gather indices and filter
                        idxs = np.where(scores >= threshold)[0]
                        sorted_idxs = idxs[np.argsort(scores[idxs])[::-1]]
                        sorted_idxs = sorted_idxs[:k]

                        results = []
                        for i in sorted_idxs:
                            results.append(
                                {
                                    "score": float(scores[i]),
                                    "page": chunks[i]["page"],
                                    "text": chunks[i]["text"],
                                }
                            )

                        st.write(f"Found **{len(results)}** clauses with cosine >= **{threshold:.2f}**")

                        if len(results) == 0:
                            st.info("No matching clauses found at this threshold. Try lowering the threshold or rephrasing your query.")
                        else:
                            # Show results in two-column layout
                            for r in results:
                                with st.expander(f"Page {r['page']} — score {r['score']:.3f}"):
                                    st.write(r['text'])

                            # allow download as CSV
                            df = pd.DataFrame(results)
                            csv_bytes = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download matches as CSV",
                                data=csv_bytes,
                                file_name=f"matches_{uploaded_file.name}.csv",
                                mime="text/csv",
                            )

        # Also allow browsing raw pages
        st.subheader("Document pages (preview)")
        for p in pages:
            st.markdown(f"**Page {p['page']}**")
            preview = p.get('text','')[:800]
            if preview:
                st.write(preview + ("..." if len(p.get('text',''))>800 else ""))
            else:
                st.write("(no extractable text on this page)")

# End of app
