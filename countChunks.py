
from qdrant_client import QdrantClient

def main():
    # Connect to local Qdrant (adjust path or URL if needed)
    client = QdrantClient(path="qdrant_local")

    # Count vectors in collection
    count = client.count(collection_name="document_vectors").count
    print(f"Total vectors (chunks) in DB: {count}")

if __name__ == "__main__":
    main()
