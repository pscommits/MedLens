"""schemas.py — MedLens API response models."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Citation(BaseModel):
    marker: str
    passage_id: str
    source: str
    passage: str


class StructuredReport(BaseModel):
    impression: str
    findings: str
    recommendations: str


class VerificationItem(BaseModel):
    sentence: str
    supported: bool
    score: float
    evidence: str


class LatencyBreakdown(BaseModel):
    vision: float
    context: float
    retrieval: float
    report: float
    verification: float
    total: float


class AnalysisResponse(BaseModel):
    """Single analysis result — returned by /analyze and inside /patient/reports."""

    # Core AI results
    pathologies:          Dict[str, float]
    heatmap_base64:       str
    structured_report:    StructuredReport
    citations:            List[Citation]
    verification:         List[VerificationItem]
    triage_level:         str
    triage_justification: str
    clinical_entities:    Dict = Field(default_factory=dict)
    retrieval_query:      Optional[str] = None
    latency_breakdown:    LatencyBreakdown

    # Vault + identity
    report_id:               Optional[str] = None
    encrypted_at:            Optional[float] = None   # unix timestamp
    doctor_stellar_pubkey:   Optional[str] = None
    patient_stellar_pubkey:  Optional[str] = None
    stellar_tx_hash:         Optional[str] = None
    stellar_explorer:        Optional[str] = None


class PatientReportsResponse(BaseModel):
    """Returned by POST /api/v1/patient/reports."""
    patient_stellar_pubkey: str
    total: int
    reports: List[AnalysisResponse]


class ErrorResponse(BaseModel):
    detail: str
    phase: Optional[str] = None
