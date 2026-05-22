"""
verification_agent.py
---------------------
Self-contained verification + triage agent for MedLens backend.

Combines the two trust-gate steps from the blueprint into one async entry
point, matching the backend.pdf signature `verify_and_triage(...)`.

Step 1 — Verification:
    Splits the LLM-drafted report into sentences, re-embeds each one with
    all-MiniLM-L6-v2, computes cosine similarity vs the retrieved passages,
    and flags any claim below the threshold as unsupported.

Step 2 — Triage:
    Applies a deterministic rule-based lookup over pathology probabilities.
    The clinician sees STAT / URGENT / ROUTINE plus a one-line justification.

Called by app/main.py as:
    triage_level, triage_justification, verification = await verify_and_triage(
        report, passages, pathologies
    )
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# Thread pool — embedding work is CPU-blocking.
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Model cache — same encoder as retrieval_agent (same backbone for consistency)
# ---------------------------------------------------------------------------
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        print("[verification_agent] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[verification_agent] Encoder ready.")
    return _encoder


# ===========================================================================
# SECTION 1 — Sentence splitting (NLTK-free, no external download)
# ===========================================================================

def _split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter — avoids the NLTK punkt download from the
    original verifier_agent.py (which fails on locked-down servers).

    Splits on '.', '!', '?' followed by whitespace, while keeping common
    abbreviations like 'e.g.', 'i.e.', 'Dr.', 'No.' intact via a placeholder
    swap. Good enough for radiology report text.
    """
    if not text:
        return []

    placeholders = {
        "e.g.":  "<EG_DOT>",
        "i.e.":  "<IE_DOT>",
        "Dr.":   "<DR_DOT>",
        "Mr.":   "<MR_DOT>",
        "Mrs.":  "<MRS_DOT>",
        "Ms.":   "<MS_DOT>",
        "No.":   "<NO_DOT>",
        "vs.":   "<VS_DOT>",
        "St.":   "<ST_DOT>",
    }
    work = text
    for k, v in placeholders.items():
        work = work.replace(k, v)

    # Naive splitter — capture trailing punctuation, drop empty fragments
    import re
    raw = re.split(r"(?<=[.!?])\s+", work)
    out: List[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        for k, v in placeholders.items():
            s = s.replace(v, k)
        out.append(s)
    return out


# ===========================================================================
# SECTION 2 — Claim-level verification
# ===========================================================================

def _verify_claims(report_text: str, passages: List[Dict], threshold: float = 0.45) -> List[Dict]:
    """
    For each sentence in the report, find the best-matching passage by
    cosine similarity. Mark it 'supported' if score >= threshold.
    """
    sentences = _split_sentences(report_text)
    if not sentences:
        return []

    if not passages:
        return [
            {"sentence": s, "score": 0.0, "supported": False, "evidence": ""}
            for s in sentences
        ]

    encoder = _get_encoder()

    # Pre-encode all passages once (much faster than re-encoding per sentence)
    passage_texts      = [p["passage"] for p in passages]
    passage_embeddings = encoder.encode(passage_texts, convert_to_tensor=True)

    results: List[Dict] = []
    for sent in sentences:
        sent_embedding = encoder.encode(sent, convert_to_tensor=True)
        sims = util.cos_sim(sent_embedding, passage_embeddings)[0]
        best_idx   = int(sims.argmax())
        best_score = float(sims[best_idx])
        evidence   = passage_texts[best_idx][:300]

        results.append({
            "sentence":  sent,
            "score":     round(best_score, 3),
            "supported": best_score >= threshold,
            "evidence":  evidence,
        })

    return results


# ===========================================================================
# SECTION 3 — Rule-based triage (deterministic, auditable)
# ===========================================================================

def _triage(pathologies: Dict[str, float]) -> Tuple[str, str]:
    """
    Apply the urgency rules from the blueprint:

        STAT     — pneumothorax (>=0.30), pneumoperitoneum, large effusion, edema (>=0.50)
        URGENT   — pneumonia (>=0.35), pleural effusion (>=0.40), mass (>=0.50), consolidation
        ROUTINE  — everything else

    Returns (level, justification).
    """
    lower = {k.lower(): v for k, v in pathologies.items()}

    # ---- STAT rules ----
    if lower.get("pneumothorax", 0.0) >= 0.30:
        return ("STAT",
                f"Possible pneumothorax detected (p={lower['pneumothorax']:.2f}); "
                "may require immediate intervention.")

    if lower.get("edema", 0.0) >= 0.50:
        return ("STAT",
                f"Pulmonary edema strongly suggested (p={lower['edema']:.2f}); "
                "urgent clinical correlation advised.")

    # ---- URGENT rules ----
    if lower.get("pneumonia", 0.0) >= 0.35:
        return ("URGENT",
                f"Pneumonia probability is significant (p={lower['pneumonia']:.2f}); "
                "physician review recommended within the hour.")

    if lower.get("effusion", 0.0) >= 0.40 or lower.get("pleural effusion", 0.0) >= 0.40:
        score = max(lower.get('effusion', 0.0), lower.get('pleural effusion', 0.0))
        return ("URGENT",
                f"Pleural effusion suggested (p={score:.2f}); clinical review required.")

    if lower.get("mass", 0.0) >= 0.50 or lower.get("lung lesion", 0.0) >= 0.50:
        score = max(lower.get('mass', 0.0), lower.get('lung lesion', 0.0))
        return ("URGENT",
                f"Possible mass / lesion detected (p={score:.2f}); follow-up imaging required.")

    if lower.get("consolidation", 0.0) >= 0.45:
        return ("URGENT",
                f"Consolidation likely (p={lower['consolidation']:.2f}); clinical correlation recommended.")

    # ---- ROUTINE fallback ----
    return ("ROUTINE",
            "No high-risk pathology crossed the urgency threshold. "
            "Routine clinical correlation recommended.")


# ===========================================================================
# SECTION 4 — Synchronous core (runs in thread pool)
# ===========================================================================

def _run_sync(
    report: Dict,
    passages: List[Dict],
    pathologies: Dict,
) -> Tuple[str, str, List[Dict]]:
    # Concatenate all report sections for verification
    report_text = " ".join([
        report.get("impression", ""),
        report.get("findings", ""),
        report.get("recommendations", ""),
    ])

    verification = _verify_claims(report_text, passages)
    triage_level, triage_justification = _triage(pathologies)

    return triage_level, triage_justification, verification


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

async def verify_and_triage(
    report: Dict,
    passages: List[Dict],
    pathologies: Dict,
) -> Tuple[str, str, List[Dict]]:
    """
    Async entry point called by the FastAPI orchestrator.

    Returns:
        triage_level         : str  — "STAT" / "URGENT" / "ROUTINE"
        triage_justification : str  — one-sentence explanation
        verification         : list — per-sentence supported/unsupported flags
    """
    loop = asyncio.get_event_loop()
    triage_level, triage_justification, verification = await loop.run_in_executor(
        _executor, _run_sync, report, passages, pathologies
    )
    return triage_level, triage_justification, verification
