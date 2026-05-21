from pydantic import BaseModel
from typing import List, Dict


class Citation(BaseModel):
    marker: str
    passage_id: str
    source: str
    passage: str


class StructuredReport(BaseModel):
    impression: str
    findings: str
    recommendations: str


class LatencyBreakdown(BaseModel):
    vision: float
    context: float
    retrieval: float
    report: float
    total: float


class AnalysisResponse(BaseModel):
    pathologies: Dict[str, float]
    heatmap_base64: str
    structured_report: StructuredReport
    citations: List[Citation]
    triage_level: str
    triage_justification: str
    latency_breakdown: LatencyBreakdown

    