"""
cnote.py
--------
BiomedBERT clinical note processor.

Mirrors the style of vision.py — plain functions, no classes, no schemas.
Returns simple Python dicts and strings.

Functions:
    load_model()          -> loads tokenizer + BiomedBERT model
    generate_embedding()  -> returns base64-encoded CLS token embedding
    extract_entities()    -> returns dict of age, sex, chief_complaint, comorbidities
    process_note()        -> runs both and returns a single combined dict
"""

import re
import base64

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


def load_model():
    """
    Load BiomedBERT tokenizer and model.

    Returns:
        (tokenizer, model) tuple — pass both to generate_embedding()
    """
    print("Loading BiomedBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print("BiomedBERT loaded.")
    return tokenizer, model


def generate_embedding(tokenizer, model, text):
    """
    Generate a 768-d CLS token embedding from clinical note text.

    Args:
        tokenizer: Loaded BiomedBERT tokenizer
        model:     Loaded BiomedBERT model
        text:      Clinical note string

    Returns:
        Base64-encoded string of the float32 embedding vector
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

    # CLS token is the first token of last_hidden_state
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    embedding_b64 = base64.b64encode(cls_embedding.astype(np.float32).tobytes()).decode("utf-8")
    return embedding_b64


def extract_entities(text):
    """
    Extract key clinical entities from a note using regex rules.

    Returns a dict with:
        age             : int or None
        sex             : "male" / "female" / None
        chief_complaint : str or None
        comorbidities   : list of str (uppercase disease codes)
    """
    entities = {
        "age": None,
        "sex": None,
        "chief_complaint": None,
        "comorbidities": [],
    }

    lower = text.lower()

    # Age + sex  (e.g. "45M", "32 female")
    match = re.search(r"(\d{1,3})\s?(m|f|male|female)", lower)
    if match:
        entities["age"] = int(match.group(1))
        entities["sex"] = "male" if match.group(2).startswith("m") else "female"

    # Chief complaint — first match wins
    complaint_map = {
        "sob":        "shortness of breath",
        "dyspnea":    "shortness of breath",
        "fever":      "fever",
        "cough":      "cough",
        "chest pain": "chest pain",
        "hemoptysis": "hemoptysis",
        "wheezing":   "wheezing",
    }
    for keyword, label in complaint_map.items():
        if keyword in lower:
            entities["chief_complaint"] = label
            break

    # Comorbidities
    disease_keywords = ["copd", "hypertension", "diabetes", "asthma", "cad", "ckd", "hf", "af", "htn"]
    entities["comorbidities"] = [d.upper() for d in disease_keywords if d in lower]

    return entities


def process_note(tokenizer, model, text):
    """
    Run both embedding + entity extraction on a clinical note.

    Args:
        tokenizer: Loaded BiomedBERT tokenizer
        model:     Loaded BiomedBERT model
        text:      Clinical note string

    Returns:
        Dict with keys: embedding_b64, entities
    """
    if not text.strip():
        zero = np.zeros(768, dtype=np.float32)
        return {
            "embedding_b64": base64.b64encode(zero.tobytes()).decode("utf-8"),
            "entities": {},
        }

    embedding_b64 = generate_embedding(tokenizer, model, text)
    entities = extract_entities(text)

    return {
        "embedding_b64": embedding_b64,
        "entities": entities,
    }