import os
import chromadb
from sentence_transformers import SentenceTransformer

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_store")

# ========================
# LOAD MODEL
# =========================
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# CONNECT TO CHROMADB
# =========================
client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(
    name="medical_knowledge"
)

# =========================
# RETRIEVAL FUNCTION
# =========================
def retrieve_medical_context(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    final_results = []

    for i in range(len(docs)):
        final_results.append({
            "text": docs[i],
            "topic": metadatas[i].get("topic", "N/A"),
            "source": metadatas[i].get("source", "N/A"),
            "score": distances[i]
        })

    return final_results


# =========================
# TEST
# =========================
if __name__ == "__main__":
    query = "pneumonia fever cough chest infection"

    results = retrieve_medical_context(query, top_k=3)

    print("\n===== RETRIEVAL RESULTS =====\n")

    for i, item in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Topic:", item["topic"])
        print("Source:", item["source"])
        print("Score:", item["score"])
        print("Text:")
        print(item["text"][:700])
        print("-" * 70)