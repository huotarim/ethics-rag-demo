"""EthicsRAG Streamlit application.

Initializes the document collection in-process when empty, using the same
embedding model for indexing and retrieval.
"""
from pathlib import Path
import logging

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ethicsrag")

COLLECTION_NAME = "ethics_documents_multilingual"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCUMENT_PATH = Path(__file__).resolve().parent / "documents" / "lausuntopyynto.txt"
DATABASE_PATH = Path(__file__).resolve().parent / "chroma_db"
CHUNK_SIZE = 500


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


@st.cache_resource(show_spinner="Preparing the multilingual search model and database…")
def load_database():
    # The same embedding function is used for indexing and querying.
    # Crucially, no separate Python subprocess loads a second model.
    logger.info("Opening Chroma database")
    client = chromadb.PersistentClient(path=str(DATABASE_PATH))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_function
    )

    if collection.count() == 0:
        if not DOCUMENT_PATH.is_file():
            raise FileNotFoundError(
                f"Required source document missing: {DOCUMENT_PATH}. "
                "Ensure documents/lausuntopyynto.txt exists in the project."
            )
        logger.info("Indexing document: %s", DOCUMENT_PATH.name)
        chunks = split_into_chunks(DOCUMENT_PATH.read_text(encoding="utf-8"))
        if not chunks:
            raise ValueError(f"Source document is empty: {DOCUMENT_PATH}")
        document_id = DOCUMENT_PATH.stem
        collection.upsert(
            ids=[f"{document_id}_chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[
                {"document_id": document_id, "source": DOCUMENT_PATH.name, "chunk_id": i}
                for i in range(len(chunks))
            ],
        )
        logger.info("Indexed %s chunks", len(chunks))
    return collection


st.title("EthicsRAG")
st.caption("Research Ethics Guidance — Retrieval Demo")
st.info(
    "This prototype retrieves relevant passages from research ethics guidance "
    "documents. It does not make ethical decisions."
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
            with st.spinner("Searching the document… First search may take longer."):
                collection = load_database()
                count = collection.count()
                if count == 0:
                    st.warning("The database contains no passages.")
                    st.stop()
                results = collection.query(query_texts=[question], n_results=min(3, count))
            for index, document in enumerate(results["documents"][0], start=1):
                metadata = results["metadatas"][0][index - 1] or {}
                st.subheader(f"Result {index}")
                st.write(document)
                st.caption(
                    f"Source: {metadata.get('source', 'unknown')} | "
                    f"Chunk: {metadata.get('chunk_id', '?')}"
                )
        except Exception as exc:
            logger.exception("EthicsRAG search failed")
            st.error(f"Search could not start: {exc}")
            st.caption("Check the terminal logs for details. Loading the embedding model may require substantial memory.")
