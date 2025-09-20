import os
from datetime import datetime, timezone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# === Config ===
QDRANT_PATH = "qdrant_local"
COLLECTION_NAME = "document_vectors"
MODEL_FILENAME = "llama-2-7b-chat-hf-q4_k_m.gguf"
MODEL_PATH = os.path.join("Models", MODEL_FILENAME)

# === RAG Filter Toggles ===
USE_FILTERS = True
USE_MEETING_TYPE_FILTER = True
USE_DATE_RANGE_FILTER = False

# === Filter Values ===
MEETING_TYPE = "Brownbag Session"
DATE_FROM = "30/07/2025"
DATE_TO = "07/08/2025"

# === Helper: Convert DD/MM/YYYY to UTC timestamp ===
def to_timestamp_utc(date_str: str) -> float:
    return datetime.strptime(date_str, "%d/%m/%Y").replace(tzinfo=timezone.utc).timestamp()

# === Embeddings ===
embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Qdrant client / vector store ===
if not os.path.exists(QDRANT_PATH):
    print("[ERROR] Qdrant index not found. Run vectorize_docs.py first.")
    raise SystemExit(1)

client = QdrantClient(path=QDRANT_PATH)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_fn,   # <-- singular param name
)

# === Debug: peek payloads ===
print("\n=== DEBUG: Print ALL docs and their Meeting_type/date values ===")
docs = vectorstore.similarity_search("", k=100)
print("All Meeting_type values in DB:")
for doc in docs:
    if doc.metadata.get("Meeting_type") == "Brownbag Session":
        print("FOUND: ", doc.metadata)

# === Filter (object style) ===
payload_filter = None
if USE_FILTERS:
    must_filters = []

    if USE_MEETING_TYPE_FILTER and MEETING_TYPE:
        must_filters.append(
            FieldCondition(
                key="Meeting_type",
                match=MatchValue(value=MEETING_TYPE)
            )
        )

    if USE_DATE_RANGE_FILTER and DATE_FROM and DATE_TO:
        must_filters.append(
            FieldCondition(
                key="date",
                range=Range(
                    gte=int(to_timestamp_utc(DATE_FROM)),
                    lte=int(to_timestamp_utc(DATE_TO)),
                ),
            )
        )

    if must_filters:
        payload_filter = Filter(must=must_filters)

print("\n[DEBUG] Payload filter about to be used:")
print(payload_filter if payload_filter else "(no filters)")

retriever = vectorstore.as_retriever(
    search_kwargs={"filter": payload_filter} if payload_filter else {}
)

# === LLM ===
if not os.path.isfile(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    raise SystemExit(1)

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=35,
    temperature=0.3,
    verbose=False,
)

# === RAG chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# === Query ===
query = "What was discussed in the brownbag session?"
result = qa_chain.invoke(query)

# === Output ===
print(f"\nQuery: {query}\nResponse: {result['result']}")

print("\n=== Retrieved Metadata ===")
for i, doc in enumerate(result["source_documents"]):
    print(f"\n[Document {i + 1}]")
    for key, value in doc.metadata.items():
        print(f"{key}: {value}")

# Cleanup
import gc
del llm
gc.collect()
