from chunk_data import get_chunks
from sentence_transformers import SentenceTransformer
import chromadb

# Get chunks and their source files from chunk_data.py
chunk_texts, chunk_sources = get_chunks()
print(f"Chunks created: {len(chunk_texts)}")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert chunks into embeddings
embeddings = model.encode(chunk_texts)

# Create a Chroma client that saves data to disk
client = chromadb.PersistentClient(path="chroma_db")

# Create (or get) a collection
collection = client.get_or_create_collection(name="college_data")

# Build metadata for each chunk (its source file)
metadatas = [{"source": source} for source in chunk_sources]

# Add chunks, embeddings, and metadata to the collection
ids = [str(i) for i in range(len(chunk_texts))]
collection.add(
    documents=chunk_texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)

print(f"Stored {len(chunk_texts)} chunks in Chroma.")