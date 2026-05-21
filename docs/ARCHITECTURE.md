# MedLens — Architecture Reference

This document explains how the pieces fit together, for your own reference
and for anyone reading the codebase for the first time.

## Request lifecycle

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT FRONTEND                               │
│  Upload X-ray  +  Clinical note  →  POST /api/v1/analyze                  │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                  FASTAPI ORCHESTRATOR  (app/main.py)                      │
│                                                                           │
│   PHASE 1 — asyncio.gather (parallel)                                     │
│   ┌──────────────────────┐       ┌──────────────────────┐                 │
│   │   Vision Agent       │       │   Context Agent      │                 │
│   │   TorchXRayVision    │       │   BiomedBERT         │                 │
│   │   + GradCAM          │       │   + entity regex     │                 │
│   └──────────┬───────────┘       └──────────┬───────────┘                 │
│              │ pathologies, heatmap         │ embeddings, entities        │
│              └──────────────┬───────────────┘                             │
│                             ▼                                             │
│   PHASE 2 — Retrieval Agent  (ChromaDB + all-MiniLM-L6-v2)                │
│              ▼ build query from entities + pathologies → top-K passages   │
│                                                                           │
│   PHASE 3 — Report Agent  (Groq Llama-3.3-70B)                            │
│              ▼ generate impression/findings/recs with [n] citations       │
│                                                                           │
│   PHASE 4 — Verification + Triage Agent  (cosine sim + rule lookup)       │
│              ▼ per-claim supported/unsupported + STAT/URGENT/ROUTINE      │
│                                                                           │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    AnalysisResponse JSON                                  │
│   pathologies · heatmap · structured_report · citations ·                 │
│   verification · triage_level · triage_justification · latencies          │
└───────────────────────────────────────────────────────────────────────────┘
```

## File responsibilities

| File | Job | Async entry point |
|------|-----|-------------------|
| `app/main.py` | Orchestrator — fans out to agents, assembles response | `analyze_chest_xray()` |
| `app/schemas.py` | Pydantic models for the API contract | — |
| `app/agents/vision_agent.py` | TorchXRayVision DenseNet + GradCAM | `run_vision_and_gradcam()` |
| `app/agents/context_agent.py` | BiomedBERT embedding + clinical entity regex | `run_biomedbert()` |
| `app/agents/retrieval_agent.py` | Build query, embed with MiniLM, query Chroma | `query_chromadb()` |
| `app/agents/report_agent.py` | Prompt Groq, parse JSON, extract `[n]` citations | `generate_llm_report()` |
| `app/agents/verification_agent.py` | Claim-level verification + rule-based triage | `verify_and_triage()` |
| `frontend/streamlit_app.py` | Three-column UI: image, report, metadata | — |
| `scripts/build_index.py` | Optional — rebuild ChromaDB from a text corpus | — |

## Why two embedding models?

- **BiomedBERT (768-d)** is used in the Context Agent to capture the
  *semantic meaning* of the clinical note. It is biomedical-domain-trained
  and outperforms general BERT on radiology text.

- **all-MiniLM-L6-v2 (384-d)** is used in both the Retrieval Agent and the
  Verification Agent because the existing `chroma_store` was built with it,
  and the entire verification pipeline must use the same embedding space as
  the index it queries.

The two are not interchangeable — the BiomedBERT embedding from the Context
Agent is not used for vector search; it's reserved for the entities-and-context
pathway and is available for future use (e.g. caching, semantic deduplication).

## Why a thread pool inside each agent?

PyTorch inference (vision, BERT, sentence-transformers) and the Groq SDK
(which is synchronous HTTP) all *block* — meaning if they ran directly inside
an `async def`, FastAPI's event loop would freeze and no other request could
be served until the slow agent returned.

Each agent therefore wraps its blocking work in `loop.run_in_executor(...)`
against a small `ThreadPoolExecutor`. This is the standard FastAPI pattern
for mixing CPU-heavy ML work with async I/O.

## Latency target

The blueprint targets **< 5 seconds end-to-end** on a CPU laptop. On a fresh
boot the first request takes longer (8–15 s) because three large models
need to be loaded into memory:

- TorchXRayVision DenseNet (~100 MB)
- BiomedBERT (~440 MB)
- all-MiniLM-L6-v2 (~90 MB)

Every subsequent request hits the warm caches and lands well under 5 s.
This is intentional — the `_get_model()` / `_get_collection()` patterns
inside each agent implement lazy loading on the first call.
