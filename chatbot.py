import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Connect to the same Chroma database you already saved
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="college_data")

# Load the same embedding model used before
model = SentenceTransformer('all-MiniLM-L6-v2')

print("NHCE College Chatbot (type 'exit' to quit)\n")

# Keep asking questions until the user types 'exit'
while True:
    question = input("Ask a question: ")

    # Let the user quit the loop
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Convert the question into a vector
    question_embedding = model.encode([question])

    # Search Chroma for the closest matching chunks
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3
    )

    # Combine the matching chunks into one block of context
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    # Build the prompt
    prompt = f"""You are a helpful assistant for New Horizon College of Engineering (NHCE).

If the user's message is a greeting or casual talk (like "hi", "hello", "thanks"), reply naturally and briefly.

If the user asks a real question about the college, answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

    # Send the prompt to the local LLM through Ollama
    response = ollama.generate(model="llama3.2", prompt=prompt)

    # Print the answer
    print("\n---------- Answer ----------")
    print(response["response"])
    print()