from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

# =========================
# DOWNLOAD NLTK DATA
# =========================
nltk.download("punkt")
nltk.download("punkt_tab")

# =========================
# LOAD MODEL
# ========================= 
print("Loading verification model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# VERIFICATION FUNCTION
# =========================
def verify_report(report, retrieved_docs, threshold=0.45):

    sentences = sent_tokenize(report)

    verified_results = []

    for sentence in sentences:

        sentence_embedding = model.encode(
            sentence,
            convert_to_tensor=True
        )

        best_score = 0
        best_doc = ""

        for doc in retrieved_docs:

            doc_embedding = model.encode(
                doc,
                convert_to_tensor=True
            )

            similarity = util.cos_sim(
                sentence_embedding,
                doc_embedding
            ).item()

            if similarity > best_score:
                best_score = similarity
                best_doc = doc

        verified_results.append({
            "sentence": sentence,
            "score": best_score,
            "supported": best_score >= threshold,
            "evidence": best_doc[:300]
        })

    return verified_results


# =========================
# TESTING
# =========================
if __name__ == "__main__":

    report = """
    Findings:
    Patient shows signs of pneumonia with fever and cough.

    Impression:
    Likely pulmonary infection.

    Recommendation:
    Antibiotic therapy and medical consultation are recommended.
    """

    retrieved_docs = [
        "Pneumonia is a lung infection that can cause fever, cough, and difficulty breathing.",
        "Treatment for pneumonia may include antibiotics and supportive medical care."
    ]

    results = verify_report(report, retrieved_docs)

    print("\n===== VERIFICATION RESULTS =====\n")

    for r in results:

        print("\nSentence:")
        print(r["sentence"])

        print("\nSimilarity Score:", round(r["score"], 3))

        print("Supported:", r["supported"])

        print("\nEvidence:")
        print(r["evidence"])

        print("\n" + "=" * 60)