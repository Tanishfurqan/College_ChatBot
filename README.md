# NHCE College Chatbot

A chatbot that answers questions about **New Horizon College of Engineering (NHCE), Bangalore** using only the college's own data. It uses Retrieval-Augmented Generation (RAG) and runs fully offline with free, open-source models. No paid AI API is used.

## Features

- Answers only from the college data (admissions, departments, hostel, placements, library, scholarships, and more)
- Says "I don't have that information" instead of guessing
- Remembers earlier messages in the chat, so follow-up questions work
- Reads both `.txt` files and PDFs
- Stores the source file name for every chunk, which helps with debugging
- Simple chat interface built with Streamlit

## How it works

1. **Load data:** Read all `.txt` files (and PDFs) from the `College_data` folder and its subfolders.
2. **Chunk:** Split the text into small pieces using `RecursiveCharacterTextSplitter` (chunk size 800, overlap 150).
3. **Embed:** Turn each chunk into a vector with `all-MiniLM-L6-v2`.
4. **Store:** Save the chunks, vectors, and source file names in a local ChromaDB database.
5. **Retrieve:** For each question, find the closest chunks.
6. **Generate:** Send the chunks and the question to a local Llama 3.2 model (via Ollama), which writes the answer from those chunks only.

## Tech stack

- Python
- LangChain text splitters
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- Ollama (Llama 3.2)
- pypdf
- Streamlit

## Project structure

```
clg_chatbot/
├── chunk_data.py        # Reads files and splits them into chunks
├── embed_store.py       # Creates embeddings and saves them in ChromaDB
├── search.py            # Tests retrieval (shows matching chunks and sources)
├── generate_answer.py   # Single question, full answer in the terminal
├── chatbot.py           # Chat loop in the terminal
├── app.py               # Streamlit web interface
├── College_data/        # Your college data (.txt and .pdf files)
└── chroma_db/           # Vector database (created automatically)
```

## Setup

1. Install [Ollama](https://ollama.com/download) and download the model:
   ```
   ollama pull llama3.2
   ```
2. Install the Python packages:
   ```
   pip install langchain-text-splitters sentence-transformers chromadb pypdf ollama streamlit
   ```
3. Put your data files in the `College_data` folder.
4. Build the database:
   ```
   python embed_store.py
   ```

## Run

Make sure the Ollama app is running, then:

```
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Updating the data

`embed_store.py` only adds data and never replaces old data. After you change any file in `College_data`:

1. Close all running terminals and the browser tab.
2. Delete the `chroma_db` folder.
3. Run `python embed_store.py` again.

## Challenges and how I solved them

- **Hallucination:** The bot invented facts that were not in the data. I rewrote the prompt so the model may use only the retrieved context. I also kept chat history separate, so it is used only to understand the topic and never as a source of facts.
- **Missed answers:** Some chunks split a heading away from its content. I increased the chunk size and overlap, and added Q&A-style entries for broad questions.
- **Vague questions:** A question like "what is fees?" returned hostel fees. I made the prompt check that the context answers the specific question asked.

## Limitations

- Answers are only as good as the data in `College_data`. Missing data means "I don't have that information."
- Broad counting questions (for example, "how many departments?") may not always work, because the answer is spread across several chunks.
- It runs on your own computer and is not deployed online.
- Fee details are not included. The bot tells students to contact the admissions office.

## Author

Tanish Furqan
