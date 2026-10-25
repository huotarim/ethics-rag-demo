import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ============================================================
# 1. TEST QUESTIONS
# ============================================================

questions = [
    "Milloin tutkimus tarvitsee eettisen ennakkoarvioinnin?",
    "Milloin tutkimukseen osallistuminen edellyttää huoltajan suostumusta?",
    "Mitä tietoon perustuva suostumus tarkoittaa?",
    "Milloin tutkimuksessa käsitellään henkilötietoja?",
    "Mitä erityisiin henkilötietoryhmiin kuuluvat tiedot ovat?",
    "Mitä tutkimuksen tietosuojasta pitää kertoa?",
    "Millaisia riskejä tutkimuksesta voi aiheutua?",
    "Milloin tutkimuksessa voidaan poiketa tietoon perustuvasta suostumuksesta?",
    "Mitä tutkimusaineistolle tapahtuu tutkimuksen päätyttyä?",
    "Mitä liitteitä eettisen ennakkoarvioinnin hakemukseen tarvitaan?"
]


# ============================================================
# 2. DATABASE CONNECTION
# ============================================================

client = chromadb.PersistentClient(path="./chroma_db")

vector_collection = client.get_collection(
    name="ethics_documents"
)

multilingual_ef = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

multilingual_collection = client.get_collection(
    name="ethics_documents_multilingual",
    embedding_function=multilingual_ef
)


# ============================================================
# 3. RETRIEVAL
# ============================================================

def search(collection, question, top_k=5):

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    return [
        metadata["chunk_id"]
        for metadata in results["metadatas"][0]
    ]


# ============================================================
# 4. BENCHMARK
# ============================================================

print("\nRAG RETRIEVAL BENCHMARK")
print("=" * 60)

for index, question in enumerate(questions, start=1):

    vector_results = search(vector_collection, question)
    multilingual_results = search(multilingual_collection, question)

    print(f"\n[{index}/{len(questions)}] {question}")

    print(f"Default:      {vector_results}")
    print(f"Multilingual: {multilingual_results}")

    # ============================================================
# 5. INSPECT CHUNKS
# ============================================================

def show_chunk(chunk_id):
    result = multilingual_collection.get(
        where={"chunk_id": chunk_id}
    )

    for document in result["documents"]:
        print(f"\n--- CHUNK {chunk_id} ---")
        print(document)