"""EthicsRAG: lightweight, local TF-IDF retrieval demo (no ML model)."""
from collections import Counter
from math import log, sqrt
from pathlib import Path
import re

import streamlit as st

DOCUMENT_PATH = Path(__file__).resolve().parent / "documents" / "lausuntopyynto.txt"
CHUNK_SIZE = 500
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def split_into_chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        while end < len(text) and not text[end].isspace():
            end += 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


@st.cache_resource(show_spinner="Loading the document and building a small search index…")
def load_index():
    if not DOCUMENT_PATH.is_file():
        raise FileNotFoundError(f"Source document not found: {DOCUMENT_PATH}")
    chunks = split_into_chunks(DOCUMENT_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("Source document is empty")

    counts = [Counter(tokenize(chunk)) for chunk in chunks]
    doc_freq = Counter()
    for count in counts:
        doc_freq.update(count.keys())
    n = len(chunks)
    idf = {term: log((1 + n) / (1 + freq)) + 1 for term, freq in doc_freq.items()}
    vectors = []
    for count in counts:
        weights = {term: (1 + log(freq)) * idf[term] for term, freq in count.items()}
        norm = sqrt(sum(weight * weight for weight in weights.values())) or 1.0
        vectors.append({term: weight / norm for term, weight in weights.items()})
    return chunks, idf, vectors


def search(question: str, chunks, idf, vectors, limit=3):
    count = Counter(tokenize(question))
    query = {term: (1 + log(freq)) * idf[term] for term, freq in count.items() if term in idf}
    norm = sqrt(sum(weight * weight for weight in query.values()))
    if norm == 0:
        return []
    query = {term: weight / norm for term, weight in query.items()}
    scored = [
        (sum(weight * vector.get(term, 0.0) for term, weight in query.items()), index)
        for index, vector in enumerate(vectors)
    ]
    best = sorted((item for item in scored if item[0] > 0), reverse=True)[:limit]
    return [(index, chunks[index], score) for score, index in best]


st.title("EthicsRAG")
st.caption("Research Ethics Guidance — Lightweight Retrieval Demo (TF-IDF)")
st.info(
    "This prototype retrieves passages from research ethics guidance documents. "
    "It does not make ethical decisions. Search uses lexical TF-IDF matching, not a language model."
)
question = st.text_input(
    "Ask a question about research ethics:",
    "Milloin tutkimus tarvitsee eettisen ennakkoarvioinnin?",
)
if st.button("Search"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            chunks, idf, vectors = load_index()
            results = search(question, chunks, idf, vectors)
            if not results:
                st.warning("No matching passages found. Try using words from the source document (Finnish).")
            for rank, (chunk_id, passage, score) in enumerate(results, start=1):
                st.subheader(f"Result {rank}")
                st.write(passage)
                st.caption(f"Source: {DOCUMENT_PATH.name} | Chunk: {chunk_id} | TF-IDF similarity: {score:.3f}")
        except (OSError, ValueError) as exc:
            st.error(f"Search could not start: {exc}")
