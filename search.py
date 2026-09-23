import chromadb
from sentence_transformers import SentenceTransformer

# Connect to the same Chroma database you already saved
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="college_data")

# Load the same embedding model used before
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ask a question (typed by you)
question = input("Ask a question: ")

# Convert the question into a vector
question_embedding = model.encode([question])

# Search Chroma for the closest matching chunks
results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=8
)

# Print the top matching chunks along with their source file
for i, doc in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i]["source"]
    print(f"\n---------- Match {i+1} (source: {source}) ----------")
    print(doc)
    