"""
main.py — MedLens FastAPI orchestrator

Endpoints:
    POST /api/v1/analyze            — AI pipeline + encrypt for patient
    POST /api/v1/patient/reports    — return ALL reports for a patient (by keypair)
    GET  /health
"""

import time
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

from app.schemas import (
    AnalysisResponse, PatientReportsResponse,
    Citation, VerificationItem, LatencyBreakdown, StructuredReport,
)
from app.agents.vision_agent       import run_vision_and_gradcam
from app.agents.context_agent      import run_biomedbert
from app.agents.retrieval_agent    import query_chromadb
from app.agents.report_agent       import generate_llm_report
from app.agents.verification_agent import verify_and_triage
from app.agents.vault_agent        import encrypt_and_store, get_patient_reports


app = FastAPI(title="MedLens API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "MedLens API", "status": "online", "version": "3.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}


# ===========================================================================
# ANALYZE — full pipeline + encrypt for a specific patient
# ===========================================================================

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_chest_xray(
    image:                  UploadFile = File(...),
    clinical_note:          str        = Form(""),
    session_id:             str        = Form("default-session"),
    patient_stellar_pubkey: str        = Form(""),
):
    start_total = time.time()

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(400, "Empty image upload.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to read image: {e}")

    # Phase 1 — Vision + Context (parallel)
    t0 = time.time()
    try:
        (pathologies, heatmap_b64), (_, entities) = await asyncio.gather(
            run_vision_and_gradcam(image_bytes),
            run_biomedbert(clinical_note),
        )
    except Exception as e:
        raise HTTPException(500, f"Phase 1 failed: {e}")
    t_parallel = time.time() - t0

    # Phase 2 — Retrieval
    t0 = time.time()
    try:
        retrieval_query, passages = await query_chromadb(entities, pathologies, top_k=5)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, f"Phase 2 failed: {e}")
    t_ret = time.time() - t0

    # Phase 3 — Report
    t0 = time.time()
    try:
        raw_report, short_citations = await generate_llm_report(pathologies, entities, passages)
    except Exception as e:
        raise HTTPException(500, f"Phase 3 failed: {e}")
    t_rep = time.time() - t0

    # Phase 4 — Verification + Triage
    t0 = time.time()
    try:
        triage_level, triage_just, verification = await verify_and_triage(
            raw_report, passages, pathologies
        )
    except Exception as e:
        raise HTTPException(500, f"Phase 4 failed: {e}")
    t_ver = time.time() - t0

    passage_map = {p["passage_id"]: p for p in passages}
    compiled_citations = [
        {"marker": c["marker"], "passage_id": p["passage_id"],
         "source": p["source"], "passage": p["passage"]}
        for c in short_citations
        if (p := passage_map.get(c["passage_id"]))
    ]

    response_dict = {
        "pathologies":          pathologies,
        "heatmap_base64":       heatmap_b64,
        "structured_report":    raw_report,
        "citations":            compiled_citations,
        "verification":         verification,
        "triage_level":         triage_level,
        "triage_justification": triage_just,
        "clinical_entities":    entities,
        "retrieval_query":      retrieval_query,
        "latency_breakdown": {
            "vision": round(time.time() - start_total - t_ret - t_rep - t_ver, 2),
            "context": round(t_parallel, 2),
            "retrieval": round(t_ret, 2),
            "report": round(t_rep, 2),
            "verification": round(t_ver, 2),
            "total": round(time.time() - start_total, 2),
        },
    }

    # Phase 5 — Encrypt + Stellar anchor
    vault_fields = dict(report_id=None, encrypted_at=None,
                        doctor_stellar_pubkey=None, patient_stellar_pubkey=None,
                        stellar_tx_hash=None, stellar_explorer=None)

    patient_key = patient_stellar_pubkey.strip()
    if patient_key:
        try:
            vault_info = await encrypt_and_store(response_dict, patient_key)
            vault_fields.update(vault_info)
            print(f"[main] Encrypted for patient, report: {vault_info['report_id']}")
        except Exception as e:
            print(f"[main] Encryption failed (non-fatal): {e}")
    else:
        print("[main] No patient key — analysis returned without encryption.")

    response_dict.update(vault_fields)
    return AnalysisResponse(**response_dict)


# ===========================================================================
# PATIENT REPORTS — patient provides their keypair, gets all their reports
# ===========================================================================

@app.post("/api/v1/patient/reports", response_model=PatientReportsResponse)
async def patient_reports(
    patient_stellar_secret: str = Form(...),
):
    """
    Return every report encrypted for this patient.

    The patient provides ONLY their Stellar secret key (S...).
    The backend derives the matching public key from it, scans vault entries,
    and decrypts each using NaCl Box(doctor_pub, patient_priv).

    The secret key is used only in memory — never stored.
    """
    if not patient_stellar_secret.strip():
        raise HTTPException(400, "patient_stellar_secret is required.")

    try:
        reports, derived_pubkey = await get_patient_reports(
            patient_stellar_secret.strip(),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Could not load reports: {e}")

    return PatientReportsResponse(
        patient_stellar_pubkey=derived_pubkey,
        total=len(reports),
        reports=[AnalysisResponse(**r) for r in reports],
    )