# MedLens
### The Verifiable Multimodal Radiology Co-Pilot

> “Every finding cited. Every region highlighted. Every decision auditable.”

MedLens is a research-inspired, multi-agent AI system for chest X-ray triage and report generation. It combines computer vision, retrieval-augmented generation (RAG), explainable AI (XAI), and verification pipelines into one clinically grounded workflow.

Unlike traditional black-box medical AI systems, MedLens provides:
- Pathology probability predictions
- Visual explanations using Grad-CAM
- Evidence-grounded report generation with citations
- Verification against retrieved medical literature
- Rule-based + LLM-assisted triage
- Secure patient-doctor communication using Stellar asymmetric encryption

Built for hackathons, designed for real-world deployment.

---

# 🚀 Core Idea

MedLens accepts:
- A chest X-ray image
- Optional clinical notes

And produces:
1. **AI pathology predictions**
2. **Grad-CAM explainability heatmaps**
3. **Evidence-grounded radiology reports**
4. **Claim verification against retrieved sources**
5. **Urgency triage classification**
6. **Encrypted medical report sharing between doctor and patient**

The entire pipeline is modular, lightweight, and runs using pretrained models — no training required.

---

# 🧠 Why MedLens?

Medical AI systems often fail because they:
- hallucinate findings,
- provide no explanations,
- cannot cite evidence,
- are difficult to trust clinically.

MedLens solves this by combining:

| Capability | How MedLens Solves It |
|---|---|
| Explainability | Grad-CAM visual overlays |
| Hallucination Reduction | Retrieval-Augmented Generation (RAG) |
| Trust & Auditability | Verification agent validates every claim |
| Multimodal Understanding | Combines X-rays + clinical text |
| Workflow Integration | Fast triage-focused UI |
| Secure Sharing | Stellar-based asymmetric encryption |

---

# 🏗️ System Architecture

```text
                 ┌────────────────────┐
                 │     Frontend UI     │
                 │ Streamlit / Next.js │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ FastAPI Orchestrator│
                 └─────────┬──────────┘
                           │
 ┌────────────┬────────────┼────────────┬─────────────┐
 ▼            ▼            ▼            ▼             ▼
Vision      XAI        Context       Retrieval      Report
Agent       Agent       Agent          Agent         Agent
 │            │            │              │             │
 └────────────┴────────────┴──────┬───────┴─────────────┘
                                  ▼
                         Verification Agent
                                  │
                                  ▼
                         Triage + Final Report
                                  │
                                  ▼
                     Stellar Encryption Layer
                                  │
                                  ▼
                  Secure Doctor ↔ Patient Sharing
