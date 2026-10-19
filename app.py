
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from chunk_data import get_chunks
from groq import Groq

# Page settings
st.set_page_config(
    page_title="NHCE Chatbot",
    page_icon="🎓",
    layout="centered"
)


# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Build the knowledge base
@st.cache_resource
def load_knowledge_base():
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="college_data"
    )

    if collection.count() == 0:
        chunk_texts, chunk_sources = get_chunks()

        if not chunk_texts:
            raise ValueError(
                "No document chunks found. Check College_data."
            )

        model = load_model()
        embeddings = model.encode(
            chunk_texts,
            show_progress_bar=False
        ).tolist()

        collection.add(
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=[
                {"source": source}
                for source in chunk_sources
            ],
            ids=[
                str(i)
                for i in range(len(chunk_texts))
            ]
        )

    return collection


model = load_model()
collection = load_knowledge_base()


# Page heading
st.title("🎓 NHCE College Chatbot")

st.caption(
    "Ask me about NHCE admissions, fees, hostel, "
    "placements, and more."
)


# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot answers questions using "
        "NHCE college documents and RAG."
    )
    st.write(
        "Document retrieval uses ChromaDB. "
        "AI answers use a hosted language model."
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# User question
question = st.chat_input("Ask a question about NHCE...")


if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    # Handle greetings without requiring college facts
    if question.strip().lower() in [
        "hi", "hello", "hey", "thanks", "thank you"
    ]:
        answer = "Hello! How can I help you with NHCE today?"

    else:
        # Retrieve relevant document chunks
        question_embedding = model.encode(
            [question]
        ).tolist()

        total_chunks = collection.count()

        if total_chunks == 0:
            answer = (
                "The college knowledge base is empty. "
                "Please check the College_data folder."
            )

        else:
            results = collection.query(
                query_embeddings=question_embedding,
                n_results=min(5, total_chunks)
            )

            retrieved_chunks = (
                results["documents"][0] or []
            )

            context = "\n\n".join(retrieved_chunks)

            # Conversation history for context
            history_text = ""

            for msg in st.session_state.messages[:-1]:
                role = (
                    "User"
                    if msg["role"] == "user"
                    else "Assistant"
                )
                history_text += (
                    f"{role}: {msg['content']}\n"
                )

            # RAG prompt
            prompt = f"""
You are a helpful assistant for New Horizon
College of Engineering (NHCE).

Use conversation history only to understand
the topic of the current question, not as a
source of factual information.

For factual questions, use ONLY the Context below.
If the context does not clearly answer the specific
question, reply exactly:
"I don't have that information."

Do not invent fees, dates, policies, placement
statistics, or other college facts.

Conversation history:
{history_text}

Context from college documents:
{context}

Question:
{question}

Answer:
"""

            # Generate answer using Groq
            try:
                if "GROQ_API_KEY" not in st.secrets:
                    answer = (
                        "AI generation is not configured yet. "
                        "Please add GROQ_API_KEY in the "
                        "Streamlit app's Secrets settings."
                    )
                else:
                    groq_client = Groq(
                        api_key=st.secrets["GROQ_API_KEY"]
                    )

                    response = (
                        groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            temperature=0.2,
                            max_tokens=500
                        )
                    )

                    answer = (
                        response.choices[0].message.content
                        or "I don't have that information."
                    )

            except Exception as e:
                st.error(
                    "The AI service could not generate "
                    "a response. Check your API key and "
                    "the deployment logs."
                )
                answer = "Please try again in a moment."


    # Display answer
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
