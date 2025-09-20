import os
import shutil
import json
from datetime import datetime, timezone
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant  # pip install -U langchain-qdrant
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

def to_ts_utc(date_val):
    """
    Normalize date to a UTC Unix timestamp (float).
    Accepts:
      - 'DD/MM/YYYY' strings (preferred)
      - numeric strings (e.g., '1753999200'/'1753999200.0')
      - ints/floats (assumed epoch seconds)
    Returns float or raises ValueError.
    """
    # Already numeric?
    if isinstance(date_val, (int, float)):
        return float(date_val)

    if isinstance(date_val, str):
        s = date_val.strip()
        # Numeric string?
        if s.replace(".", "", 1).isdigit():
            return float(s)

        # DD/MM/YYYY
        try:
            dt = datetime.strptime(s, "%d/%m/%Y").replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            pass

    raise ValueError(f"Unsupported date format: {date_val!r}")

# === Load all .txt files ===
all_docs = []
for filename in os.listdir(SOURCE_DIR):
    if not filename.endswith(".txt"):
        continue

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

                # Normalize date → float UTC epoch
                if "date" in loaded_meta:
                    try:
                        # keep human-readable version if present
                        if isinstance(loaded_meta["date"], str):
                            metadata["date_str"] = loaded_meta["date"].strip()
                        loaded_meta["date"] = to_ts_utc(loaded_meta["date"])
                        # Debug
                        print(f"[DEBUG] date→UTC timestamp for {filename}: {loaded_meta['date']}")
                    except ValueError as e:
                        print(f"[WARN] Invalid date in {json_path}: {e}")
                        loaded_meta.pop("date", None)  # Remove invalid date

                metadata.update(loaded_meta)

            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse metadata for {filename}")

    # Attach metadata to each document
    for doc in documents:
        doc.metadata.update(metadata)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Debug: verify date is float if present
    for doc in chunks:
        if "date" in doc.metadata and not isinstance(doc.metadata["date"], float):
            try:
                doc.metadata["date"] = to_ts_utc(doc.metadata["date"])
            except ValueError:
                print(f"[WARN] Removing invalid date in chunk from {doc.metadata.get('source')}")
                doc.metadata.pop("date", None)

    all_docs.extend(chunks)

    # Move processed files AFTER successful chunking
    shutil.move(txt_path, os.path.join(PROCESSED_DIR, filename))
    if os.path.exists(json_path):
        shutil.move(json_path, os.path.join(PROCESSED_DIR, os.path.basename(json_path)))
    print(f"Vectorized and moved: {filename}")

# === Final global check (optional) ===
for doc in all_docs:
    if "date" in doc.metadata and not isinstance(doc.metadata["date"], float):
        print(f"[FINAL ERROR] Non-float date in {doc.metadata.get('source')}: {doc.metadata['date']}")

# === Create embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Create Qdrant client and collection ===
client = QdrantClient(path=QDRANT_PATH)

existing = [col.name for col in client.get_collections().collections]
if COLLECTION_NAME not in existing:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("Created new Qdrant collection.")
else:
    print("Using existing Qdrant collection.")

# === Add documents to Qdrant ===
if all_docs:
    vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    vectorstore.add_documents(all_docs)
    print("Documents added to Qdrant.")
else:
    print("[WARN] No documents to add.")
