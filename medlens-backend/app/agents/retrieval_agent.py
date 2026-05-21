# app/agents/retrieval_agent.py
# PURPOSE: Query ChromaDB with BiomedBERT embeddings + pathology scores
# Returns: top 5 most relevant medical passages

import asyncio
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import chromadb
from sentence_transformers import SentenceTransformer

# Thread pool for blocking ChromaDB operations
_executor = ThreadPoolExecutor(max_workers=2)

# Get ChromaDB path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_store")

# Load embedding model once
_model = None

def _get_model():
    global _model
    if _model is None:
        print("[retrieval_agent] Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _run_sync(
    pathologies: Dict[str, float],
    entities: dict,
    top_k: int = 5
) -> List[Dict]:
    """
    Synchronous retrieval function (runs in thread pool).
    
    Args:
        pathologies: {"Pneumonia": 0.78, "Effusion": 0.42, ...}
        entities: {"age": 65, "sex": "male", "chief_complaint": "fever", ...}
        top_k: number of passages to return
    
    Returns:
        List of dicts: [{"passage_id": "p_0", "source": "Wikipedia", "passage": "...", "score": 0.92}, ...]
    """
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name="medical_knowledge")
    except Exception as e:
        print(f"[retrieval_agent] ChromaDB connection failed: {e}")
        return []
    
    # Build search query from pathologies + entities
    query_parts = []
    
    # Add top pathologies
    for pathology, score in sorted(pathologies.items(), key=lambda x: x[1], reverse=True)[:3]:
        if score > 0.25:  # Only include detected pathologies
            query_parts.append(pathology.lower())
    
    # Add clinical entities
    if entities.get("chief_complaint"):
        query_parts.append(entities["chief_complaint"])
    
    if entities.get("symptoms"):
        query_parts.extend(entities["symptoms"][:2])
    
    if entities.get("comorbidities"):
        query_parts.extend(entities["comorbidities"][:1])
    
    if not query_parts:
        print("[retrieval_agent] No query built — returning empty results")
        return []
    
    query = " ".join(query_parts)
    print(f"[retrieval_agent] Query: {query}")
    
    # Embed the query
    model = _get_model()
    query_embedding = model.encode(query).tolist()
    
    # Search ChromaDB
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count())
        )
    except Exception as e:
        print(f"[retrieval_agent] ChromaDB query failed: {e}")
        return []
    
    # Format results
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    passages = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        # Convert distance to similarity score (lower distance = higher similarity)
        score = 1.0 - dist  # Normalize to [0, 1]
        
        passages.append({
            "passage_id": f"p_{i}",
            "passage": doc,
            "source": meta.get("source", "Unknown"),
            "score": float(score)
        })
    
    return passages

async def query_chromadb(
    pathologies: Dict[str, float],
    entities: dict,
    top_k: int = 5
) -> List[Dict]:
    """
    Async entry point called by main.py orchestrator.
    
    Dispatches blocking ChromaDB work to thread pool.
    """
    loop = asyncio.get_event_loop()
    passages = await loop.run_in_executor(
        _executor, _run_sync, pathologies, entities, top_k
    )
    return passages
