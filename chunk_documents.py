import chromadb
from pathlib import Path

# Read extracted text
file_path = Path("documents/lausuntopyynto.txt")
text = file_path.read_text(encoding="utf-8")
document_id = file_path.stem

# Split text into chunks
CHUNK_SIZE = 500

chunks = []

# Keep full words
start = 0

while start < len(text):
    end = min(start + CHUNK_SIZE, len(text))

    # Extend chunk to the next whitespace
    if end < len(text):
        while end < len(text) and not text[end].isspace():
            end += 1

    chunk = text[start:end].strip()
    chunks.append(chunk)

    start = end

# Show results
print(f"Document length: {len(text)} characters")
print(f"Number of chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {index} ---")
    print(chunk)

# Create persistent vector database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="ethics_documents"
)

# Remove old chunks for the document in process
old_chunks = collection.get(
    where={"document_id": document_id}
)

if old_chunks["ids"]:
    collection.delete(ids=old_chunks["ids"])

# Store chunks with document identifiers
collection.upsert(
    ids=[f"{document_id}_chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    metadatas=[
        {
            "document_id": document_id,
            "source": file_path.name,
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]
)

print(f"\nStored {len(chunks)} chunks in ChromaDB.")
print(f"Database contains {collection.count()} chunks.")
#print(collection.get(limit=3)["ids"])