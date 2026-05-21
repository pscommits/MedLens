# app/agents/report_agent.py

# PURPOSE:
# Takes pathology scores + retrieved evidence
# Sends them to Groq LLM
# Returns a structured radiology report


import json
from groq import Groq
from dotenv import load_dotenv

from app.schemas import (
    ReportOutput,
    FindingItem,
    CitationItem
)

# ============================================
# LOAD ENVIRONMENT VARIABLES
# ============================================

load_dotenv()

# Create Groq client
client = Groq()

# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are a radiology reporting assistant.

Generate ONLY valid JSON.
No markdown.
No explanations.

Output structure:

{
  "impression": "overall summary",
  "findings": [
    {
      "pathology": "condition",
      "description": "description",
      "citation_ids": [1]
    }
  ],
  "recommendations": ["recommendation"],
  "citations": [
    {
      "id": 1,
      "passage": "evidence text",
      "source": "Radiopaedia"
    }
  ]
}

Rules:
1. Every finding must cite evidence.
2. Do not invent diseases.
3. Output ONLY JSON.
"""

# ============================================
# MAIN FUNCTION
# ============================================


def run(
    pathology_scores,
    clinical_entities,
    retrieved_passages
) -> ReportOutput:

    # ----------------------------------------
    # FILTER ACTIVE PATHOLOGIES
    # ----------------------------------------

    active = [
        p for p in pathology_scores
        if p.above_threshold
    ]

    # ----------------------------------------
    # HANDLE NORMAL XRAY
    # ----------------------------------------

    if not active:

        return ReportOutput(
            impression="No acute cardiopulmonary abnormality detected.",
            findings=[],
            recommendations=[
                "Routine clinical follow-up if needed."
            ],
            citations=[]
        )

    # ----------------------------------------
    # BUILD PASSAGE TEXT
    # ----------------------------------------

    passage_text = ""

    for i, p in enumerate(retrieved_passages, start=1):

        passage_text += (
            f"[{i}] {p.text} "
            f"(Source: {p.source})\n\n"
        )

    # ----------------------------------------
    # BUILD FINDINGS TEXT
    # ----------------------------------------

    findings_text = ""

    for p in active:

        findings_text += (
            f"- {p.name}: "
            f"{p.probability:.0%} confidence\n"
        )

    # ----------------------------------------
    # CLINICAL CONTEXT
    # ----------------------------------------

    age = clinical_entities.get("age", "unknown")
    sex = clinical_entities.get("sex", "unknown")
    symptoms = clinical_entities.get(
        "chief_complaint",
        "not specified"
    )

    # ----------------------------------------
    # USER MESSAGE
    # ----------------------------------------

    user_message = f"""
Patient context:
{age}-year-old {sex}

Chief complaint:
{symptoms}

Detected pathologies:
{findings_text}

Reference passages:
{passage_text}

Generate the radiology report JSON now.
"""

    # ========================================
    # CALL GROQ API
    # ========================================

    last_error = None

    for attempt in range(2):

        try:

            response = client.chat.completions.create(

                model="llama3-70b-8192",

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],

                temperature=0.2
            )

            # --------------------------------
            # EXTRACT LLM TEXT
            # --------------------------------

            content = response.choices[0].message.content

            # --------------------------------
            # PARSE JSON
            # --------------------------------

            data = json.loads(content)

            # --------------------------------
            # BUILD PYDANTIC OBJECTS
            # --------------------------------

            findings = [

                FindingItem(
                    pathology=f["pathology"],
                    description=f["description"],
                    citation_ids=f["citation_ids"]
                )

                for f in data["findings"]
            ]

            citations = [

                CitationItem(
                    id=c["id"],
                    passage=c["passage"],
                    source=c["source"]
                )

                for c in data["citations"]
            ]

            # --------------------------------
            # RETURN FINAL REPORT
            # --------------------------------

            return ReportOutput(

                impression=data["impression"],

                findings=findings,

                recommendations=data["recommendations"],

                citations=citations
            )

        except Exception as e:

            last_error = str(e)

            # Retry once with correction hint
            user_message += f"""

Previous output was invalid.

Error:
{last_error}

Please return ONLY valid JSON.
"""

    # ========================================
    # FALLBACK RESPONSE
    # ========================================

    return ReportOutput(

        impression="Unable to generate verified report.",

        findings=[],

        recommendations=[
            "Manual radiologist review recommended."
        ],

        citations=[]
    )
