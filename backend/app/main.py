"""
main.py
-------
MedLens FastAPI orchestrator.

The "air traffic controller" of the MedLens backend. Receives a chest X-ray
and an optional clinical note, then coordinates five specialist agents
through four pipeline phases:

    PHASE 1 (parallel) — Vision + Context
    PHASE 2 (serial)   — Retrieval
    PHASE 3 (serial)   — Report generation (Groq LLM)
    PHASE 4 (serial)   — Verification + Triage

Returns a single, fully-typed JSON response defined in schemas.AnalysisResponse.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env from the backend directory (one level up from app/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

from app.schemas import (
    AnalysisResponse,
    Citation,
    VerificationItem,
    LatencyBreakdown,
    StructuredReport,
)
from app.agents.vision_agent       import run_vision_and_gradcam
from app.agents.context_agent      import run_biomedbert
from app.agents.retrieval_agent    import query_chromadb
from app.agents.report_agent       import generate_llm_report
from app.agents.verification_agent import verify_and_triage


# ===========================================================================
# APP SETUP
# ===========================================================================

app = FastAPI(
    title="MedLens API Orchestrator",
    description="Multimodal · Multi-Agent · RAG-grounded radiology co-pilot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Streamlit + Next.js both connect freely
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# HEALTH ENDPOINT — used by the Streamlit "is backend alive?" probe
# ===========================================================================

@app.get("/")
async def root():
    return {
        "service": "MedLens API",
        "status":  "online",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ===========================================================================
# MAIN ANALYZE ENDPOINT
# ===========================================================================

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_chest_xray(
    image:         UploadFile = File(...),
    clinical_note: str        = Form(""),
    session_id:    str        = Form("default-session"),
):
    """
    Run the full MedLens pipeline on one chest X-ray.

    Form fields:
        image         : the X-ray file (PNG/JPG) — required
        clinical_note : free-text clinical note  — optional
        session_id    : opaque session identifier — optional (for future audit logs)

    Returns: AnalysisResponse — see schemas.py for the exact shape.
    """

    start_total = time.time()

    # -----------------------------------------------------------------
    # Read image bytes once, share across agents
    # -----------------------------------------------------------------
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    # -----------------------------------------------------------------
    # PHASE 1 — Vision + Context in parallel (asyncio.gather)
    # Both are CPU-bound model inferences running in their own thread pools,
    # so they truly run concurrently.
    # -----------------------------------------------------------------
    t0_parallel = time.time()
    try:
        (pathologies, heatmap_b64), (embeddings, entities) = await asyncio.gather(
            run_vision_and_gradcam(image_bytes),
            run_biomedbert(clinical_note),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Phase 1 (vision/context) failed: {e}",
        )
    t_parallel = time.time() - t0_parallel

    # -----------------------------------------------------------------
    # PHASE 2 — Retrieval (needs entities + pathologies from phase 1)
    # -----------------------------------------------------------------
    t0_ret = time.time()
    try:
        retrieval_query, retrieved_passages = await query_chromadb(
            entities, pathologies, top_k=5
        )
    except FileNotFoundError as e:
        # Surface the chroma-store-missing error with full guidance
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Phase 2 (retrieval) failed: {e}",
        )
    t_ret = time.time() - t0_ret

    # -----------------------------------------------------------------
    # PHASE 3 — Report generation (Groq LLM)
    # -----------------------------------------------------------------
    t0_rep = time.time()
    try:
        raw_report, short_citations = await generate_llm_report(
            pathologies, entities, retrieved_passages
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Phase 3 (report) failed: {e}",
        )
    t_rep = time.time() - t0_rep

    # -----------------------------------------------------------------
    # PHASE 4 — Verification + Triage
    # -----------------------------------------------------------------
    t0_ver = time.time()
    try:
        triage_level, triage_justification, verification = await verify_and_triage(
            raw_report, retrieved_passages, pathologies
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Phase 4 (verification/triage) failed: {e}",
        )
    t_ver = time.time() - t0_ver

    # -----------------------------------------------------------------
    # Assemble full citation objects (marker + passage_id + source + passage)
    # -----------------------------------------------------------------
    passage_map = {p["passage_id"]: p for p in retrieved_passages}
    compiled_citations = []
    for cit in short_citations:
        pas = passage_map.get(cit["passage_id"])
        if pas is None:
            continue
        compiled_citations.append(
            Citation(
                marker=cit["marker"],
                passage_id=pas["passage_id"],
                source=pas["source"],
                passage=pas["passage"],
            )
        )

    verification_items = [VerificationItem(**v) for v in verification]

    total_latency = time.time() - start_total

    return AnalysisResponse(
        pathologies=pathologies,
        heatmap_base64=heatmap_b64,
        structured_report=StructuredReport(**raw_report),
        citations=compiled_citations,
        verification=verification_items,
        triage_level=triage_level,
        triage_justification=triage_justification,
        clinical_entities=entities,
        retrieval_query=retrieval_query,
        latency_breakdown=LatencyBreakdown(
            vision=round(t_parallel, 2),
            context=round(t_parallel, 2),  # ran in parallel with vision
            retrieval=round(t_ret, 2),
            report=round(t_rep, 2),
            verification=round(t_ver, 2),
            total=round(total_latency, 2),
        ),
    )
