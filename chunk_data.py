import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

def get_chunks():
    # Main folder that contains all your college data
    DATA_FOLDER = "College_data"

    # List to store (text, source_file) pairs
    documents = []

    # Walk through College_data and all subfolders
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)

            # Handle .txt files
            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append((text, file))

            # Handle .pdf files
            elif file.endswith(".pdf"):
                reader = PdfReader(file_path)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text()
                documents.append((pdf_text, file))

    # Create the text splitter
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    # Split each document, keeping track of which file it came from
    chunk_texts = []
    chunk_sources = []

    for text, source_file in documents:
        pieces = splitter.split_text(text)
        for piece in pieces:
            chunk_texts.append(piece)
            chunk_sources.append(source_file)

    return chunk_texts, chunk_sources

# This only runs if you execute chunk_data.py directly
if __name__ == "__main__":
    chunk_texts, chunk_sources = get_chunks()
    print(f"Chunks created: {len(chunk_texts)}")