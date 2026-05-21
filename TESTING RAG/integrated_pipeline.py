from vision import load_model, preprocess_image, run_inference
from cnote import load_model as load_bert_model, process_note

from agents.retrieval_agent import retrieve_medical_context
from agents.report_agent import generate_report
from agents.verifier_agent import verify_report
from agents.triage_agent import get_triage_level


# =========================
# INPUTS
# =========================
image_path = "xray.jpg"

clinical_note = """
45M with fever, cough, chest pain and shortness of breath.
History of COPD.
"""


# =========================
# 1. VISION AGENT
# =========================
print("\n========== VISION AGENT ==========")

vision_model = load_model()
img_tensor = preprocess_image(image_path)

scores = run_inference(
    vision_model,
    img_tensor,
    threshold=0.30
)

print("\nDisease Predictions:")
for disease, score in scores.items():
    print(f"{disease}: {score:.4f}")


# =========================
# 2. CONTEXT AGENT
# =========================
print("\n========== CONTEXT AGENT ==========")

tokenizer, bert_model = load_bert_model()

note_result = process_note(
    tokenizer,
    bert_model,
    clinical_note
)

entities = note_result["entities"]

print("\nClinical Entities:")
print(entities)


# =========================
# 3. QUERY BUILDER
# =========================
print("\n========== QUERY BUILDER ==========")

query_parts = []

# Add disease predictions
for disease, score in scores.items():
    if score >= 0.30:
        query_parts.append(disease)

# Add clinical complaint
chief_complaint = entities.get("chief_complaint")
if chief_complaint:
    query_parts.append(chief_complaint)

# Add comorbidities
for c in entities.get("comorbidities", []):
    query_parts.append(c)

# Add age/sex if present
if entities.get("age"):
    query_parts.append(f"age {entities['age']}")

if entities.get("sex"):
    query_parts.append(entities["sex"])

query = " ".join(query_parts)

print("\nGenerated Query:")
print(query)


# =========================
# 4. RETRIEVAL AGENT
# =========================
print("\n========== RETRIEVAL AGENT ==========")

retrieved = retrieve_medical_context(query, top_k=5)
retrieved_docs = [r["text"] for r in retrieved]

for i, r in enumerate(retrieved, start=1):
    print(f"\nResult {i}")
    print("Topic:", r["topic"])
    print("Source:", r["source"])
    print("Score:", r["score"])
    print("Text:", r["text"][:400])


# =========================
# 5. REPORT AGENT
# =========================
print("\n========== REPORT AGENT ==========")

report = generate_report(query, retrieved_docs)

print("\nGenerated Report:\n")
print(report)


# =========================
# 6. VERIFICATION AGENT
# =========================
print("\n========== VERIFICATION AGENT ==========")

verification_results = verify_report(report, retrieved_docs)

for i, v in enumerate(verification_results, start=1):
    print(f"\nClaim {i}")
    print("Sentence:", v["sentence"])
    print("Supported:", v["supported"])
    print("Score:", round(v["score"], 3))
    print("Evidence:", v["evidence"][:250])


# =========================
# 7. TRIAGE AGENT
# =========================
print("\n========== TRIAGE AGENT ==========")

disease_predictions = [
    {
        "disease": disease,
        "probability": score
    }
    for disease, score in scores.items()
]

triage = get_triage_level(disease_predictions)

print("\nTriage Level:", triage["level"])
print("Reason:", triage["reason"])


# =========================
# FINAL OUTPUT
# =========================
print("\n========== FINAL OUTPUT ==========")

final_output = {
    "query": query,
    "disease_predictions": disease_predictions,
    "clinical_entities": entities,
    "retrieved_evidence": retrieved,
    "report": report,
    "verification": verification_results,
    "triage": triage
}

print("\nPipeline completed successfully.")