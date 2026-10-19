import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Page settings (title, icon, layout)
st.set_page_config(
    page_title="NHCE Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Connect to the same Chroma database you already saved
client = chromadb.Client()
collection = client.get_or_create_collection(name="college_data")

# Load the embedding model (cached so it doesn't reload on every question)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("🎓 NHCE College Chatbot")
st.caption("Ask me anything about New Horizon College of Engineering — admissions, fees, hostel, placements, and more.")

# Sidebar with basic info
with st.sidebar:
    st.header("About")
    st.write("This chatbot answers questions using NHCE's official college data.")
    st.write("It runs fully offline using a local AI model.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Keep track of chat history across the page
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box at the bottom of the page
question = st.chat_input("Ask a question about NHCE...")

if question:
    # Show the user's question
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Convert the question into a vector
    question_embedding = model.encode([question])

    # Search Chroma for the closest matching chunks
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=5
    )
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    # Build conversation history as text (excluding the current question)
    history_text = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Build the prompt
    prompt = f"""You are a helpful assistant for New Horizon College of Engineering (NHCE).

If the user's message is a greeting or casual talk (like "hi", "hello", "thanks"), reply naturally and briefly.

Use the conversation history ONLY to understand what topic the user is asking about (like "it", "that", "what about X"). Do NOT use the conversation history as a source of facts.

For any factual answer, use ONLY the information in the Context section below.
The Context may contain chunks that are only loosely related to the question, not actual answers to it. Carefully check whether the Context actually answers the SPECIFIC question asked.
If the question asks about "fees" in general, and the Context only has hostel fees (not tuition fees), that does NOT count as an answer — say "I don't have that information."
If the specific fact requested is not clearly and directly in the Context, say "I don't have that information" — even if something related or similar-sounding appears in the Context.

Conversation history (for understanding topic only, not facts):
{history_text}

Context (the ONLY source of facts you may use):
{context}

Question: {question}

Answer:"""

    # Get the answer from Ollama
    response = ollama.generate(model="llama3.2", prompt=prompt)
    answer = response["response"]

    # Show the answer
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})