import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Open existing database
client = chromadb.PersistentClient(path="./chroma_db")

multilingual_ef = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_collection(
    name="ethics_documents_multilingual",
    embedding_function=multilingual_ef
)

# Search question
question = "Milloin tutkimus tarvitsee eettisen ennakkoarvioinnin?"

results = collection.query(
    query_texts=[question],
    n_results=collection.count() #n_results=5
)

# Display results with sources
for index, document in enumerate(results["documents"][0], start=1):
    metadata = results["metadatas"][0][index - 1]

    print(f"\n--- RESULT {index} ---")
    print(f"Source: {metadata['source']}")
    print(f"Chunk: {metadata['chunk_id']}")
    distance = results["distances"][0][index - 1]
    print(f"Distance: {distance:.4f}")
    print(document)

# Find the ranking of chunk 4
for rank, metadata in enumerate(results["metadatas"][0], start=1):
    if metadata["chunk_id"] == 4:
        print(f"\nChunk 4 ranking: {rank}")
        break