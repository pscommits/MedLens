"""
context_agent.py
----------------
Self-contained context agent for MedLens backend.
Integrates BiomedBERT embeddings + clinical entity extraction.

Called by app/main.py as:
    embeddings, entities = await run_biomedbert(clinical_note)

Input:
    clinical_note : str  — free-text clinical note
                           e.g. "45M, SOB x2d, fever 38.5C, hx COPD"
                           Empty string is handled gracefully (zero vector returned).

Output:
    embeddings : list[float]  — 768 floats representing the semantic meaning of the note.
    entities   : dict         — {age, sex, chief_complaint, comorbidities}
"""

import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# Thread pool — BERT inference is blocking; keep FastAPI's event loop free.
_executor = ThreadPoolExecutor(max_workers=2)


# Model cache — BiomedBERT is ~440 MB and takes ~8-12 s to load on first request.
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
# Embedding generation
# ===========================================================================

def _generate_embedding(tokenizer, model, text: str) -> list[float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_vector.astype(np.float32).tolist()


# ===========================================================================
# Entity extraction
# ===========================================================================

def _extract_entities(text: str) -> dict:
    entities = {
        "age":             None,
        "sex":             None,
        "chief_complaint": None,
        "comorbidities":   [],
    }

    lower = text.lower()

    # Age + sex — "45M", "45 M", "45 male", "32F", "32 female"
    match = re.search(r"(\d{1,3})\s?(m|f|male|female)", lower)
    if match:
        entities["age"] = int(match.group(1))
        entities["sex"] = "male" if match.group(2).startswith("m") else "female"

    # Chief complaint — first keyword match wins
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

    # Comorbidities — all matches collected
    disease_keywords = [
        "copd", "hypertension", "diabetes", "asthma",
        "cad", "ckd", "hf", "af", "htn",
    ]
    entities["comorbidities"] = [d.upper() for d in disease_keywords if d in lower]

    return entities


# ===========================================================================
# Synchronous pipeline (runs in thread pool)
# ===========================================================================

def _run_sync(clinical_note: str) -> tuple[list[float], dict]:
    if not clinical_note.strip():
        print("[context_agent] Empty clinical note received. Returning zero embedding.")
        return [0.0] * 768, {}

    tokenizer, bert_model = _get_model()

    embeddings = _generate_embedding(tokenizer, bert_model, clinical_note)
    entities   = _extract_entities(clinical_note)

    return embeddings, entities


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

async def run_biomedbert(clinical_note: str) -> tuple[list[float], dict]:
    loop = asyncio.get_event_loop()
    embeddings, entities = await loop.run_in_executor(
        _executor, _run_sync, clinical_note
    )
    return embeddings, entities
