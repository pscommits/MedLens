# app/agents/triage_agent.py
import asyncio
from typing import Dict

async def get_triage_level(pathologies: Dict[str, float]) -> Dict:
    """
    Assign triage level based on pathology scores.
    """
    triage_level = "ROUTINE"
    justification = "No significant findings"
    
    for pathology, score in pathologies.items():
        # STAT criteria
        if "Pneumothorax" in pathology and score > 0.40:
            return {
                "level": "STAT",
                "justification": "Pneumothorax detected - may require immediate intervention"
            }
        
        if "Tension" in pathology and score > 0.30:
            return {
                "level": "STAT",
                "justification": "Tension pneumothorax suspected - emergency condition"
            }
        
        # URGENT criteria
        if ("Pneumonia" in pathology or "Consolidation" in pathology) and score > 0.35:
            triage_level = "URGENT"
            justification = f"{pathology} detected - urgent physician review recommended"
        
        if "Effusion" in pathology and score > 0.40:
            if triage_level != "URGENT":
                triage_level = "URGENT"
            justification = f"Significant {pathology} detected - requires clinical evaluation"
        
        if "Edema" in pathology and score > 0.40:
            if triage_level != "URGENT":
                triage_level = "URGENT"
            justification = "Pulmonary edema detected - urgent assessment needed"
    
    return {
        "level": triage_level,
        "justification": justification
    }
