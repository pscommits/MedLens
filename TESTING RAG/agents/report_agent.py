from groq import Groq

# =========================
# GROQ CLIENT
# =========================
client = Groq(
    api_key=""
)

# =========================
# REPORT GENERATOR
# =========================
def generate_report(query, retrieved_docs):

    # Combine retrieved passages
    context = "\n\n".join(retrieved_docs)

    # Prompt
    prompt = f""" 
You are an expert radiology AI assistant.

Patient Query:
{query}

Retrieved Medical Context:
{context}

Generate a structured medical report with:

1. Findings
2. Impression
3. Recommendation

Keep it concise, medically accurate, and easy to understand.
Do not hallucinate.
Only use information supported by the provided context.
"""

    # =========================
    # GROQ API CALL
    # =========================
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================
# TESTING
# =========================
if __name__ == "__main__":

    query = "pneumonia fever cough chest pain"

    retrieved_docs = [
        "Pneumonia is a lung infection causing inflammation in the air sacs.",
        "Symptoms include fever, cough, chest pain, and difficulty breathing.",
        "Treatment typically includes antibiotics and supportive care."
    ]

    report = generate_report(query, retrieved_docs)

    print("\n===== GENERATED REPORT =====\n")
    print(report)