# app/agents/verification_agent.py
import asyncio
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util

_executor = ThreadPoolExecutor(max_workers=2)
_model = None

def _get_model():
    global _model
    if _model is None:
        print("[verification_agent] Loading verification model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _run_sync(
    report: Dict,
    passages: List[Dict],
    threshold: float = 0.55
) -> Dict:
    """
    Verify each claim in the report against retrieved passages.
    """
    model = _get_model()
    
    verified_findings = []
    low_confidence_claims = []
    
    findings = report.get("findings", "")
    if not findings:
        return {
            "verified_report": report,
            "confidence_per_claim": {},
            "low_confidence_claims": []
        }
    
    # Split findings into sentences
    sentences = [s.strip() for s in findings.split(".") if s.strip()]
    
    confidence_per_claim = {}
    
    for sentence in sentences:
        if not sentence:
            continue
        
        # Embed the claim
        claim_embedding = model.encode(sentence, convert_to_tensor=True)
        
        best_score = 0.0
        best_passage = ""
        
        # Find best matching passage
        for passage in passages:
            passage_text = passage.get("passage", "")
            if not passage_text:
                continue
            
            passage_embedding = model.encode(passage_text, convert_to_tensor=True)
            similarity = util.cos_sim(claim_embedding, passage_embedding).item()
            
            if similarity > best_score:
                best_score = similarity
                best_passage = passage_text
        
        confidence_per_claim[sentence] = best_score
        
        if best_score < threshold:
            low_confidence_claims.append(sentence)
    
    return {
        "verified_report": report,
        "confidence_per_claim": confidence_per_claim,
        "low_confidence_claims": low_confidence_claims
    }

async def verify_and_triage(
    report: Dict,
    passages: List[Dict],
    pathologies: Dict[str, float]
) -> Dict:
    """
    Verify report + assign triage level.
    """
    loop = asyncio.get_event_loop()
    verification = await loop.run_in_executor(
        _executor, _run_sync, report, passages
    )
    
    # Assign triage based on pathologies
    triage_level = "ROUTINE"
    triage_reason = "No high-risk findings detected"
    
    for pathology, score in pathologies.items():
        if "Pneumothorax" in pathology and score > 0.4:
            triage_level = "STAT"
            triage_reason = "Pneumothorax detected - immediate action required"
            break
        elif ("Pneumonia" in pathology or "Consolidation" in pathology) and score > 0.35:
            if triage_level != "STAT":
                triage_level = "URGENT"
                triage_reason = f"{pathology} detected - urgent review needed"
        elif "Effusion" in pathology and score > 0.4:
            if triage_level != "STAT":
                triage_level = "URGENT"
                triage_reason = "Significant effusion detected"
    
    return {
        "verification": verification,
        "triage_level": triage_level,
        "triage_reason": triage_reason
    }