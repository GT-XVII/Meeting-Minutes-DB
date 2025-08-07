import os
import shutil
import json
from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# === Config ===
SOURCE_DIR = "Docs/Fresh Docs"
PROCESSED_DIR = "Docs/Vectorized Docs"
QDRANT_PATH = "qdrant_local"
COLLECTION_NAME = "document_vectors"

# === Ensure processed dir exists ===
os.makedirs(PROCESSED_DIR, exist_ok=True)

# === Load all .txt files ===
all_docs = []
for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".txt"):
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(SOURCE_DIR, filename)
        json_path = os.path.join(SOURCE_DIR, base_name + ".json")

        # Load .txt content
        loader = TextLoader(txt_path)
        documents = loader.load()

        # Load .json metadata if it exists
        metadata = {"source": filename}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    loaded_meta = json.load(f)

                    # Convert "date" to a timestamp (float)
                    if "date" in loaded_meta:
                        try:
                            loaded_meta["date"] = datetime.strptime(loaded_meta["date"], "%d/%m/%Y").timestamp()
                        except ValueError:
                            print(f"[WARN] Invalid date format in {json_path}. Expected dd/mm/yyyy.")

                    metadata.update(loaded_meta)

                except json.JSONDecodeError:
                    print(f"[WARN] Failed to parse metadata for {filename}")

        # Attach metadata to each document
        for doc in documents:
            doc.metadata.update(metadata)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        all_docs.extend(chunks)

        # Move processed files
        shutil.move(txt_path, os.path.join(PROCESSED_DIR, filename))
        if os.path.exists(json_path):
            shutil.move(json_path, os.path.join(PROCESSED_DIR, os.path.basename(json_path)))
        print(f"Vectorized and moved: {filename}")

# === Create embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Create Qdrant client and collection ===
client = QdrantClient(path=QDRANT_PATH)

if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("Created new Qdrant collection.")
else:
    print("Using existing Qdrant collection.")

# === Add documents to Qdrant ===
vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
vectorstore.add_documents(all_docs)
print("Documents added to Qdrant.")
