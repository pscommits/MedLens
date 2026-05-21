"""
context_agent.py
----------------
Self-contained context agent for MedLens backend.
Integrates BiomedBERT embeddings + clinical entity extraction.

No external .py imports — all logic is inlined here.

Called by app/main.py as:
    embeddings, entities = await run_biomedbert(clinical_note)

Input:
    clinical_note : str  — free-text clinical note
                           e.g. "45M, SOB x2d, fever 38.5C, hx COPD"
                           Empty string is handled gracefully (zero vector returned).

Output:
    embeddings : list[float]  — 768 floats representing the semantic meaning of the note.
                                 Passed to retrieval_agent to query ChromaDB for
                                 similar medical literature passages.

    entities   : dict         — structured fields extracted from the note text:
                                 {
                                   "age":             int or None,
                                   "sex":             "male" / "female" / None,
                                   "chief_complaint": str or None,
                                   "comorbidities":   list[str]  e.g. ["COPD", "HTN"]
                                 }
"""

import re
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Thread pool — same reasoning as vision_agent: BERT inference is blocking.
# Running it directly in async def would freeze FastAPI's event loop.
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Model cache — BiomedBERT is ~440 MB and takes ~8-12 s to load.
# Loaded once on first request, reused for all subsequent requests.
# ---------------------------------------------------------------------------
_tokenizer  = None
_bert_model = None

_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


def _get_model():
    global _tokenizer, _bert_model
    if _tokenizer is None or _bert_model is None:
        print(f"[context_agent] Loading BiomedBERT ({_MODEL_NAME})...")
        _tokenizer  = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _bert_model = AutoModel.from_pretrained(_MODEL_NAME)
        _bert_model.eval()
        print("[context_agent] BiomedBERT ready.")
    return _tokenizer, _bert_model


# ===========================================================================
# SECTION 1 — Embedding generation  (inlined from cnote.py)
# ===========================================================================

def _generate_embedding(tokenizer, model, text: str) -> list[float]:
    """
    Run text through BiomedBERT and return the CLS token embedding.

    The CLS token is the first token of every BERT input. After training,
    it acts as a summary of the entire input sequence — perfect for
    semantic similarity searches in ChromaDB.

    Steps:
        1. Tokenize the text (truncate to 256 tokens max)
        2. Forward pass through BiomedBERT (no gradient needed)
        3. Extract the CLS token from last_hidden_state[:, 0, :]
        4. Return as a plain Python list[float] (768 values)

    Args:
        tokenizer : loaded BiomedBERT tokenizer
        model     : loaded BiomedBERT model in eval mode
        text      : clinical note string

    Returns:
        list[float] of length 768
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token = index 0 of the sequence dimension
    cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_vector.astype(np.float32).tolist()   # list[float], length 768


# ===========================================================================
# SECTION 2 — Entity extraction  (inlined from cnote.py)
# ===========================================================================

def _extract_entities(text: str) -> dict:
    """
    Extract structured clinical fields from free-text using regex rules.

    Handles common shorthand formats used in clinical notes:
        "45M"       → age=45, sex="male"
        "32 female" → age=32, sex="female"
        "SOB"       → chief_complaint="shortness of breath"
        "hx COPD"   → comorbidities=["COPD"]

    Args:
        text : raw clinical note string

    Returns:
        dict with keys: age, sex, chief_complaint, comorbidities
    """
    entities = {
        "age":             None,
        "sex":             None,
        "chief_complaint": None,
        "comorbidities":   [],
    }

    lower = text.lower()

    # --- Age + sex ---
    # Matches: "45M", "45 M", "45 male", "32F", "32 female"
    match = re.search(r"(\d{1,3})\s?(m|f|male|female)", lower)
    if match:
        entities["age"] = int(match.group(1))
        entities["sex"] = "male" if match.group(2).startswith("m") else "female"

    # --- Chief complaint — first keyword match wins ---
    complaint_map = {
        "sob":        "shortness of breath",
        "dyspnea":    "shortness of breath",
        "chest pain": "chest pain",
        "fever":      "fever",
        "cough":      "cough",
        "hemoptysis": "hemoptysis",
        "wheezing":   "wheezing",
    }
    for keyword, label in complaint_map.items():
        if keyword in lower:
            entities["chief_complaint"] = label
            break

    # --- Comorbidities — all matches collected ---
    disease_keywords = [
        "copd", "hypertension", "diabetes", "asthma",
        "cad", "ckd", "hf", "af", "htn",
    ]
    entities["comorbidities"] = [d.upper() for d in disease_keywords if d in lower]

    return entities


# ===========================================================================
# SECTION 3 — Synchronous pipeline (runs in thread pool)
# ===========================================================================

def _run_sync(clinical_note: str) -> tuple[list[float], dict]:
    """
    Full synchronous pipeline: load model → embed → extract entities.

    Handles the empty-note edge case: returns a 768-dimensional zero
    vector and an empty entities dict so the rest of the pipeline
    doesn't break when no clinical note is provided.

    This function is blocking (BERT inference) so it must NOT be called
    directly in an async context. Dispatched via run_in_executor below.
    """
    # Empty note guard — return zeros + empty entities
    if not clinical_note.strip():
        print("[context_agent] Empty clinical note received. Returning zero embedding.")
        return [0.0] * 768, {}

    tokenizer, bert_model = _get_model()

    embeddings = _generate_embedding(tokenizer, bert_model, clinical_note)
    entities   = _extract_entities(clinical_note)

    return embeddings, entities


# ===========================================================================
# PUBLIC ENTRY POINT  —  called by app/main.py
# ===========================================================================

async def run_biomedbert(clinical_note: str) -> tuple[list[float], dict]:
    """
    Async entry point called by the FastAPI orchestrator in app/main.py.

    Dispatches the blocking BERT work to a thread pool so FastAPI's
    event loop stays free to handle other requests concurrently.

    Args:
        clinical_note : str — the free-text clinical note from the form submission.
                              Pass "" if no note provided; handled gracefully.

    Returns:
        embeddings : list[float]   768 semantic float values for ChromaDB search
        entities   : dict          {age, sex, chief_complaint, comorbidities}
    """
    loop = asyncio.get_event_loop()
    embeddings, entities = await loop.run_in_executor(
        _executor, _run_sync, clinical_note
    )
    return embeddings, entities