"""
retrieval_agent.py
------------------
Self-contained retrieval agent for MedLens backend.

Queries a persistent ChromaDB store of curated radiology passages
(Radiopaedia, RSNA, NIH Bookshelf, Wikipedia-medical) using the
all-MiniLM-L6-v2 sentence transformer.

Called by app/main.py as:
    passages = await query_chromadb(entities, pathologies, top_k=5)

Why all-MiniLM-L6-v2 and not BiomedBERT?
    The existing chroma_store (data/chroma_store) was built with
    all-MiniLM-L6-v2 embeddings. BiomedBERT (768-d, used for context
    semantics) is dimensionally and distributionally incompatible with
    that index. The BiomedBERT embedding from context_agent is therefore
    used downstream by the report agent for prompt context — not here.

Input:
    entities    : dict   — {age, sex, chief_complaint, comorbidities}
    pathologies : dict   — {pathology_name: probability, ...} sorted desc
    top_k       : int    — number of passages to return (default 5)

Output:
    List[dict] — each item has keys:
        passage_id : str   — stable id (e.g. "rad_102", or chroma's auto id)
        source     : str   — e.g. "Radiopaedia"
        passage    : str   — the actual passage text
        score      : float — similarity distance (lower = closer)
        topic      : str   — coarse topic tag, if available
"""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Thread pool — sentence transformer inference + chroma I/O are blocking.
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Path setup
# Resolves to:    <backend>/data/chroma_store
# Allows override via env var CHROMA_PATH if you want to put it elsewhere.
# ---------------------------------------------------------------------------
_DEFAULT_CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "chroma_store",
)
CHROMA_PATH      = os.environ.get("CHROMA_PATH", _DEFAULT_CHROMA_PATH)
COLLECTION_NAME  = os.environ.get("CHROMA_COLLECTION", "medical_knowledge")


# ---------------------------------------------------------------------------
# Lazy-loaded singletons — encoder ~90 MB, chroma client opens a SQLite file.
# ---------------------------------------------------------------------------
_encoder    = None
_collection = None
_chroma_client = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        print("[retrieval_agent] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[retrieval_agent] Encoder ready.")
    return _encoder


def _get_collection():
    global _collection, _chroma_client
    if _collection is None:
        if not os.path.isdir(CHROMA_PATH):
            raise FileNotFoundError(
                f"[retrieval_agent] ChromaDB store not found at: {CHROMA_PATH}\n"
                f"  Set CHROMA_PATH env var, or copy your existing store to that path.\n"
                f"  See README section 'ChromaDB Setup' for details."
            )
        print(f"[retrieval_agent] Opening ChromaDB at {CHROMA_PATH} ...")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection    = _chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"[retrieval_agent] Collection '{COLLECTION_NAME}' ready "
              f"({_collection.count()} passages).")
    return _collection


# ===========================================================================
# Query builder — fuses pathology findings + clinical entities into one string
# ===========================================================================

def _build_query(entities: Dict, pathologies: Dict, pathology_threshold: float = 0.30) -> str:
    """
    Build a natural-language query string from the multimodal context.

    Mirrors the query-builder pattern in integrated_pipeline.py but driven
    by the new async data structures.

    Example output:
        "Atelectasis Effusion shortness of breath COPD age 45 male"
    """
    parts: List[str] = []

    # Pathologies that crossed the confidence threshold
    for disease, score in pathologies.items():
        if score >= pathology_threshold:
            parts.append(disease)

    # Chief complaint
    cc = entities.get("chief_complaint")
    if cc:
        parts.append(cc)

    # Comorbidities
    for c in entities.get("comorbidities", []) or []:
        parts.append(c)

    # Age & sex
    age = entities.get("age")
    if age is not None:
        parts.append(f"age {age}")
    sex = entities.get("sex")
    if sex:
        parts.append(sex)

    # Fallback if nothing fired (e.g. all pathologies below threshold and no note)
    if not parts:
        # Use the single highest-confidence pathology even if below threshold
        if pathologies:
            parts.append(next(iter(pathologies)))
        else:
            parts.append("chest radiograph findings")

    return " ".join(parts)


# ===========================================================================
# Synchronous core (runs in thread pool)
# ===========================================================================

def _run_sync(entities: Dict, pathologies: Dict, top_k: int) -> tuple[str, List[Dict]]:
    encoder    = _get_encoder()
    collection = _get_collection()

    query = _build_query(entities, pathologies)
    query_embedding = encoder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    docs       = results.get("documents", [[]])[0]
    metadatas  = results.get("metadatas", [[]])[0] or [{}] * len(docs)
    distances  = results.get("distances", [[]])[0] or [0.0] * len(docs)
    ids        = results.get("ids",       [[]])[0] or [f"pid_{i}" for i in range(len(docs))]

    out: List[Dict] = []
    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        out.append({
            "passage_id": str(ids[i]) if i < len(ids) else f"pid_{i}",
            "passage":    doc,
            "source":     meta.get("source", "MedicalKnowledgeBase") if meta else "MedicalKnowledgeBase",
            "topic":      meta.get("topic", "N/A") if meta else "N/A",
            "score":      float(distances[i]) if i < len(distances) else 0.0,
        })

    return query, out


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

async def query_chromadb(entities: Dict, pathologies: Dict, top_k: int = 5) -> tuple[str, List[Dict]]:
    """
    Async entry point called by the FastAPI orchestrator.

    Returns:
        query     : str        — the query string actually sent to chroma
                                  (exposed so the frontend can show "why this evidence")
        passages  : List[dict] — top-K retrieved passages
    """
    loop = asyncio.get_event_loop()
    query, passages = await loop.run_in_executor(
        _executor, _run_sync, entities, pathologies, top_k
    )
    return query, passages
