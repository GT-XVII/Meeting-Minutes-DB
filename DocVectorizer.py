import os
import shutil
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# === Config ===
SOURCE_DIR = "docs"
PROCESSED_DIR = "vectorized_docs"
FAISS_INDEX_DIR = "faiss_index"

# === Ensure processed dir exists ===
os.makedirs(PROCESSED_DIR, exist_ok=True)

# === Load all .txt files ===
all_docs = []
for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(SOURCE_DIR, filename)
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        all_docs.extend(chunks)

        # Move file to processed dir
        shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
        print(f"Vectorized and moved: {filename}")

# === Create embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Create or update FAISS index ===
try:
    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
    print("Loaded existing FAISS index.")
    vectorstore.add_documents(all_docs)
except:
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    print("Created new FAISS index.")

vectorstore.save_local(FAISS_INDEX_DIR)
print("Saved FAISS index.")
