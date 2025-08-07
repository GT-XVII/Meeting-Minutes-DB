import os
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

# === Config ===
QDRANT_PATH = "qdrant_local"
COLLECTION_NAME = "document_vectors"
MODEL_FILENAME = "llama-2-7b-chat-hf-q4_k_m.gguf"
MODEL_PATH = os.path.join("Models", MODEL_FILENAME)

# === RAG Filter Settings ===
USE_FILTERS = True
MEETING_TYPE = "Brownbag Session"
DATE_FROM = "01/08/2022"
DATE_TO = "01/10/2022"

# === Helper: Convert DD/MM/YYYY to timestamp ===
def to_timestamp(date_str: str) -> float:
    return datetime.strptime(date_str, "%d/%m/%Y").timestamp()

# === Load Embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Load Qdrant Client ===
if not os.path.exists(QDRANT_PATH):
    print("[ERROR] Qdrant index not found. Run vectorize_docs.py first.")
    exit(1)

client = QdrantClient(path=QDRANT_PATH)

vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)
print("[INFO] Qdrant vectorstore loaded successfully.")

# === Optional: Build Metadata Filter ===
filter_conditions = []

if USE_FILTERS:
    if MEETING_TYPE:
        filter_conditions.append(
            FieldCondition(
                key="Meeting type",
                match=MatchValue(value=MEETING_TYPE)
            )
        )

    if DATE_FROM and DATE_TO:
        filter_conditions.append(
            FieldCondition(
                key="date",
                range=Range(
                    gte=to_timestamp(DATE_FROM),
                    lte=to_timestamp(DATE_TO)
                )
            )
        )

retriever = vectorstore.as_retriever(
    search_kwargs={"filter": Filter(must=filter_conditions)} if USE_FILTERS else {}
)

# === Load LlamaCpp model ===
if not os.path.isfile(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    exit(1)

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=35,
    temperature=0.7,
    verbose=True,
)

# === Create Retrieval-Augmented Generation Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# === Query ===
query = "Summarize what was discussed in the Brownbag Sessions this week"
result = qa_chain.invoke(query)

# === Output ===
print(f"\nQuery: {query}\nResponse: {result['result']}")

print("\n=== Retrieved Metadata ===")
for i, doc in enumerate(result['source_documents']):
    print(f"\n[Document {i + 1}]")
    for key, value in doc.metadata.items():
        print(f"{key}: {value}")
