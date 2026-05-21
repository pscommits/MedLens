"""
report_agent.py
---------------
Self-contained report agent for MedLens backend.

Uses the Groq API to generate a structured radiology report grounded
in retrieved passages, with inline [n] citations that map 1-to-1 to the
passages.

Called by app/main.py as:
    report, citations = await generate_llm_report(pathologies, entities, passages)

Output shape (matches schemas.StructuredReport + List[Citation]):
    report = {
        "impression":       "...",
        "findings":         "...",
        "recommendations":  "...",
    }
    citations = [
        {"marker": "[1]", "passage_id": "rad_102"},
        {"marker": "[2]", "passage_id": "rad_217"},
        ...
    ]
"""

import os
import re
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

from groq import Groq


# ---------------------------------------------------------------------------
# Thread pool — Groq client is a sync HTTP client; keep FastAPI loop free.
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Config from .env
# ---------------------------------------------------------------------------
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.2"))


# ---------------------------------------------------------------------------
# Lazy-loaded client (so import doesn't fail if GROQ_API_KEY is unset)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "[report_agent] GROQ_API_KEY is not set. "
                "Add it to backend/.env or export it in the shell."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ===========================================================================
# Prompt builder
# ===========================================================================

def _build_prompt(pathologies: Dict, entities: Dict, passages: List[Dict]) -> str:
    """
    Build a structured prompt instructing the LLM to:
      - Only use information from the provided passages
      - Insert inline [n] citation markers tied to those passages
      - Return strictly-parseable JSON
    """
    # Pathologies (top 5 only, to keep prompt focused)
    top_paths = list(pathologies.items())[:5]
    paths_str = "\n".join(
        f"  - {name}: {score:.3f}" for name, score in top_paths
    )

    # Entities
    ent_str = (
        f"  Age: {entities.get('age', 'N/A')}\n"
        f"  Sex: {entities.get('sex', 'N/A')}\n"
        f"  Chief complaint: {entities.get('chief_complaint', 'N/A')}\n"
        f"  Comorbidities: {', '.join(entities.get('comorbidities', []) or []) or 'None'}"
    )

    # Numbered passages — these are the citation targets
    passages_str = "\n\n".join(
        f"[{i+1}] (id={p['passage_id']}, source={p['source']})\n{p['passage']}"
        for i, p in enumerate(passages)
    )

    prompt = f"""You are an expert radiology AI assistant. Generate a structured radiology report.

PATHOLOGY PREDICTIONS (from CNN classifier):
{paths_str}

CLINICAL CONTEXT:
{ent_str}

RETRIEVED MEDICAL EVIDENCE (you MUST cite these by marker):
{passages_str}

INSTRUCTIONS:
1. Write a concise, medically accurate report in three sections: impression, findings, recommendations.
2. Every clinical claim must be followed by an inline citation marker like [1], [2], etc.
   referring to the numbered passages above.
3. Do NOT invent information not supported by the passages or pathology predictions.
4. If pathologies are low-confidence (<0.30), describe them as "possible" or "cannot be excluded".
5. Output ONLY a JSON object with this exact shape — no markdown, no commentary:

{{
  "impression":       "<one-paragraph impression with [n] markers>",
  "findings":         "<detailed findings with [n] markers>",
  "recommendations":  "<follow-up suggestions with [n] markers>"
}}
"""
    return prompt


# ===========================================================================
# Citation extractor
# ===========================================================================

_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


def _extract_citations(report: Dict, passages: List[Dict]) -> List[Dict]:
    """
    Walk the impression/findings/recommendations text, pull every [n] marker,
    and map each one back to the corresponding passage_id.

    De-duplicates by marker so we don't return [1] three times.
    """
    full_text = " ".join([
        report.get("impression", ""),
        report.get("findings", ""),
        report.get("recommendations", ""),
    ])

    seen = set()
    out: List[Dict] = []
    for match in _MARKER_PATTERN.finditer(full_text):
        n = int(match.group(1))
        if n in seen:
            continue
        seen.add(n)
        if 1 <= n <= len(passages):
            out.append({
                "marker":     f"[{n}]",
                "passage_id": passages[n - 1]["passage_id"],
            })
    return out


# ===========================================================================
# Response parser — robust against minor LLM JSON quirks
# ===========================================================================

def _parse_response(raw: str) -> Dict:
    """
    Parse the LLM response into the {impression, findings, recommendations} dict.

    Handles three common failure modes:
      - LLM wraps JSON in ```json ... ``` code fences
      - LLM adds a preamble before the JSON
      - LLM uses smart quotes (rare with llama, but safe to handle)
    """
    text = raw.strip()

    # Strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find the first '{' and the last '}' — most robust extraction
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    # Normalize smart quotes if present
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Fallback — return the raw text as the impression so the pipeline doesn't die
        print(f"[report_agent] JSON parse failed: {e}. Returning raw text as impression.")
        return {
            "impression":       raw.strip()[:1500],
            "findings":         "Report generation produced non-JSON output; raw text shown in impression.",
            "recommendations":  "Manual review required.",
        }

    # Make sure all three keys exist
    return {
        "impression":      data.get("impression",      "No impression generated."),
        "findings":        data.get("findings",        "No findings generated."),
        "recommendations": data.get("recommendations", "No recommendations generated."),
    }


# ===========================================================================
# Synchronous core (runs in thread pool)
# ===========================================================================

def _run_sync(pathologies: Dict, entities: Dict, passages: List[Dict]) -> Tuple[Dict, List[Dict]]:
    client = _get_client()
    prompt = _build_prompt(pathologies, entities, passages)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=GROQ_TEMPERATURE,
    )

    raw = response.choices[0].message.content
    report = _parse_response(raw)
    citations = _extract_citations(report, passages)

    return report, citations


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

async def generate_llm_report(
    pathologies: Dict,
    entities: Dict,
    passages: List[Dict],
) -> Tuple[Dict, List[Dict]]:
    """
    Async entry point called by the FastAPI orchestrator.

    Returns:
        report    : dict — {impression, findings, recommendations}
        citations : list — [{marker, passage_id}, ...]
    """
    loop = asyncio.get_event_loop()
    report, citations = await loop.run_in_executor(
        _executor, _run_sync, pathologies, entities, passages
    )
    return report, citations
