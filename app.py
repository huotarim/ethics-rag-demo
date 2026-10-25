import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
import subprocess
import sys


# --------------------------------------------------
# Database
# --------------------------------------------------

@st.cache_resource
def load_database():

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path="./chroma_db")

    # Initialize database if the collection does not exist
    if "ethics_documents_multilingual" not in [
        c.name for c in client.list_collections()
    ]:
        subprocess.run(
            [sys.executable, "chunk_documents.py"],
            check=True
        )

    return client.get_collection(
        name="ethics_documents_multilingual",
        embedding_function=embedding_function
    )


# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("EthicsRAG")
st.caption("Research Ethics Guidance — Retrieval Demo")

st.info(
    "This prototype retrieves relevant passages from research "
    "ethics guidance documents. It does not make ethical decisions."
)

question = st.text_input(
    "Ask a question about research ethics:",
    "Milloin tutkimus tarvitsee eettisen ennakkoarvioinnin?"
)

if st.button("Search"):

    collection = load_database()

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    for index, document in enumerate(results["documents"][0], start=1):

        metadata = results["metadatas"][0][index - 1]

        st.subheader(f"Result {index}")
        st.write(document)

        st.caption(
            f"Source: {metadata['source']} | "
            f"Chunk: {metadata['chunk_id']}"
        )