"""
schemas.py
----------
Pydantic models that define the strict, production-grade shape of the
/api/v1/analyze response.

Mirrors the JSON blueprint from backend.pdf but adds a `verification`
list so the frontend can show per-claim "Supported / Unsupported" badges —
which is the single most demo-worthy feature called out in the MedLens
blueprint.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """One citation marker in the report, fully expanded with the actual passage."""
    marker: str = Field(..., description='Inline marker e.g. "[1]"')
    passage_id: str = Field(..., description="Stable id of the retrieved passage")
    source: str = Field(..., description="Human-readable source name e.g. 'Radiopaedia'")
    passage: str = Field(..., description="Full text of the retrieved passage")


class StructuredReport(BaseModel):
    """The three canonical sections of a radiology report."""
    impression: str
    findings: str
    recommendations: str


class VerificationItem(BaseModel):
    """Per-sentence verification result from the verifier agent."""
    sentence: str
    supported: bool
    score: float = Field(..., description="Cosine similarity vs best evidence passage")
    evidence: str = Field(..., description="Snippet of the best-matching passage")


class LatencyBreakdown(BaseModel):
    """Wall-clock timing for every pipeline phase, in seconds."""
    vision: float
    context: float
    retrieval: float
    report: float
    verification: float
    total: float


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """Final JSON returned by POST /api/v1/analyze."""
    pathologies: Dict[str, float]
    heatmap_base64: str
    structured_report: StructuredReport
    citations: List[Citation]
    verification: List[VerificationItem]
    triage_level: str = Field(..., description="STAT | URGENT | ROUTINE")
    triage_justification: str
    clinical_entities: Dict = Field(default_factory=dict)
    retrieval_query: Optional[str] = None
    latency_breakdown: LatencyBreakdown


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    phase: Optional[str] = None
