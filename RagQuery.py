import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# === Config ===
FAISS_INDEX_DIR = "faiss_index"
MODEL_FILENAME = "llama-2-7b-chat.gguf"
MODEL_PATH = os.path.join("models", MODEL_FILENAME)

# === Load embeddings and vectorstore ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists(FAISS_INDEX_DIR):
    print("[ERROR] FAISS index not found. Run vectorize_docs.py first.")
    exit(1)

vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
retriever = vectorstore.as_retriever()

# === Check for model ===
if not os.path.isfile(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    exit(1)

# === Load Llama.cpp model ===
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=35,
    temperature=0.7,
    verbose=True,
)

# === Create RAG chain and query ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

query = "What is the document about?"
response = qa_chain.run(query)
print(f"\nQuery: {query}\nResponse: {response}")
