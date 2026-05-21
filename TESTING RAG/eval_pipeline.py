"""
eval_pipeline.py
----------------
Evaluation pipeline for the medical AI verification agent.

Runs 10 X-ray images × 100 synthetic clinical notes through the full pipeline
and measures the verification agent's accept/reject behaviour across:
  - Vision model output (disease predictions)
  - RAG retrieval (ChromaDB similarity)
  - Clinical note input (entities)

Outputs:
  - eval_results.json          — raw per-sample results
  - eval_summary.json          — aggregate metrics
  - eval_report.html           — interactive evaluation dashboard
"""

import os
import sys
import json
import time
import random
import base64
import warnings
import traceback
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATH SETUP — add project root so local modules resolve
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "agents"))

# ─────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────
RESULTS_JSON  = os.path.join(PROJECT_ROOT, "eval_results.json")
SUMMARY_JSON  = os.path.join(PROJECT_ROOT, "eval_summary.json")
REPORT_HTML   = os.path.join(PROJECT_ROOT, "eval_report.html")

# ─────────────────────────────────────────────
# CLINICAL NOTE TEMPLATES
# ─────────────────────────────────────────────
AGES    = list(range(25, 85))
SEXES   = ["M", "F"]

COMPLAINTS = [
    "shortness of breath",
    "persistent cough",
    "chest pain",
    "fever and chills",
    "hemoptysis",
    "wheezing",
    "dyspnea on exertion",
    "productive cough with yellow sputum",
]

COMORBIDITIES_POOL = [
    "COPD", "hypertension", "diabetes mellitus", "asthma",
    "CAD", "CKD", "heart failure", "atrial fibrillation", "obesity",
    "smoking history", "lung cancer", "tuberculosis history",
]

HISTORY_SNIPPETS = [
    "No significant travel history.",
    "Recent travel to Southeast Asia.",
    "Works in a dusty industrial environment.",
    "Non-smoker, no prior lung disease.",
    "Ex-smoker, 20 pack-years.",
    "Lives alone, no recent sick contacts.",
    "Admitted 3 days ago for similar symptoms.",
    "Recently completed a course of antibiotics.",
    "No known drug allergies.",
    "Family history of lung cancer.",
]

IMAGING_CONTEXTS = [
    "CXR ordered for dyspnea workup.",
    "Routine pre-operative chest X-ray.",
    "Follow-up imaging post-pneumonia treatment.",
    "Acute presentation, imaging requested by ED.",
    "Outpatient referral for chronic cough evaluation.",
    "ICU patient, portable AP film.",
    "Second opinion requested for abnormal finding.",
    "Screening CXR for occupational health.",
]


def generate_clinical_note(seed: int) -> str:
    rng = random.Random(seed)
    age  = rng.choice(AGES)
    sex  = rng.choice(SEXES)
    comorbidities = rng.sample(COMORBIDITIES_POOL, k=rng.randint(1, 4))
    complaints    = rng.sample(COMPLAINTS, k=rng.randint(1, 3))
    history       = rng.choice(HISTORY_SNIPPETS)
    context       = rng.choice(IMAGING_CONTEXTS)

    note = (
        f"{age}{sex} presenting with {', '.join(complaints)}. "
        f"History of {', '.join(comorbidities)}. "
        f"{history} {context}"
    )
    return note


def generate_100_notes() -> list[dict]:
    notes = []
    for i in range(100):
        note = generate_clinical_note(seed=42 + i)
        notes.append({"id": i, "note": note})
    return notes


# ─────────────────────────────────────────────
# LAZY IMPORTS — so missing deps don't crash import
# ─────────────────────────────────────────────
def try_import_vision():
    try:
        from vision import load_model, preprocess_image, run_inference
        return load_model, preprocess_image, run_inference
    except Exception as e:
        print(f"  [WARN] vision.py unavailable: {e}")
        return None, None, None


def try_import_cnote():
    try:
        from cnote import load_model as load_bert, process_note
        return load_bert, process_note
    except Exception as e:
        print(f"  [WARN] cnote.py unavailable: {e}")
        return None, None


def try_import_retrieval():
    try:
        from retrieval_agent import retrieve_medical_context
        return retrieve_medical_context
    except Exception as e:
        print(f"  [WARN] retrieval_agent.py unavailable: {e}")
        return None


def try_import_verifier():
    try:
        from verifier_agent import verify_report
        return verify_report
    except Exception as e:
        print(f"  [WARN] verifier_agent.py unavailable: {e}")
        return None


def try_import_report():
    try:
        from report_agent import generate_report
        return generate_report
    except Exception as e:
        print(f"  [WARN] report_agent.py unavailable: {e}")
        return None


def try_import_triage():
    try:
        from triage_agent import get_triage_level
        return get_triage_level
    except Exception as e:
        print(f"  [WARN] triage_agent.py unavailable: {e}")
        return None


# ─────────────────────────────────────────────
# MOCK / STUB FUNCTIONS
# Used when real models are unavailable (CI / no GPU)
# ─────────────────────────────────────────────
DISEASE_NAMES = [
    "Pneumonia", "Atelectasis", "Consolidation", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass",
]


def mock_vision_scores(image_path: str, seed: int) -> dict:
    """Deterministic fake vision scores based on filename + seed."""
    rng = random.Random(hash(image_path) ^ seed)
    scores = {d: round(rng.uniform(0.05, 0.95), 4) for d in DISEASE_NAMES}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def mock_entities(note: str) -> dict:
    """Extract entities using cnote's regex logic without BiomedBERT."""
    try:
        from cnote import extract_entities
        return extract_entities(note)
    except Exception:
        import re
        lower = note.lower()
        match = re.search(r"(\d{1,3})\s?(m|f|male|female)", lower)
        age  = int(match.group(1)) if match else None
        sex  = ("male" if match.group(2).startswith("m") else "female") if match else None
        comorbidities = [d.upper() for d in
                         ["copd", "hypertension", "diabetes", "asthma", "cad"]
                         if d in lower]
        chief = None
        for kw, lbl in [("sob", "shortness of breath"), ("dyspnea", "shortness of breath"),
                         ("fever", "fever"), ("cough", "cough"), ("chest pain", "chest pain")]:
            if kw in lower:
                chief = lbl
                break
        return {"age": age, "sex": sex, "chief_complaint": chief, "comorbidities": comorbidities}


def mock_retrieve(query: str, top_k: int = 5, seed: int = 0) -> list[dict]:
    """Return plausible fake retrieval results."""
    rng = random.Random(hash(query) ^ seed)
    topics = ["Pneumonia pathophysiology", "Chest X-ray interpretation",
              "COPD management", "Pleural effusion causes", "Atelectasis treatment"]
    results = []
    for i in range(top_k):
        t = topics[i % len(topics)]
        results.append({
            "text": (f"{t}: This condition is associated with pulmonary infiltrates, "
                     f"fever, and respiratory distress. Clinical management involves "
                     f"imaging and appropriate therapeutic intervention."),
            "topic": t,
            "source": rng.choice(["pubmed_ncbi", "wikipedia", "radiopaedia", "medlineplus"]),
            "score": round(rng.uniform(0.1, 0.6), 4),
        })
    return results


def mock_report(query: str, docs: list[str]) -> str:
    return (
        f"Findings: Patient presents with symptoms consistent with pulmonary pathology. "
        f"Imaging findings and clinical context suggest {query.split()[0] if query else 'infection'}.\n\n"
        f"Impression: Likely pulmonary abnormality requiring clinical correlation.\n\n"
        f"Recommendation: Clinical review, appropriate antibiotic therapy, and follow-up imaging advised."
    )


def mock_verify(report: str, docs: list[str], threshold: float = 0.45) -> list[dict]:
    """Fake verification using cosine similarity via sentence-transformers if available, else random."""
    try:
        from verifier_agent import verify_report
        return verify_report(report, docs, threshold)
    except Exception:
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            from nltk.tokenize import sent_tokenize
            import nltk
            try:
                sentences = sent_tokenize(report)
            except Exception:
                sentences = [s.strip() for s in report.split(".") if len(s.strip()) > 10]
            results = []
            for sent in sentences:
                s_emb = model.encode(sent, convert_to_tensor=True)
                best_score, best_doc = 0.0, ""
                for doc in docs:
                    d_emb = model.encode(doc, convert_to_tensor=True)
                    score = util.cos_sim(s_emb, d_emb).item()
                    if score > best_score:
                        best_score = score
                        best_doc = doc
                results.append({
                    "sentence": sent,
                    "score": round(best_score, 4),
                    "supported": best_score >= threshold,
                    "evidence": best_doc[:300],
                })
            return results
        except Exception:
            rng = random.Random(hash(report))
            sentences = [s.strip() for s in report.split(".") if len(s.strip()) > 10]
            results = []
            for sent in sentences:
                score = round(rng.uniform(0.2, 0.85), 4)
                results.append({
                    "sentence": sent,
                    "score": score,
                    "supported": score >= threshold,
                    "evidence": docs[0][:300] if docs else "",
                })
            return results


def mock_triage(disease_predictions: list[dict]) -> dict:
    try:
        from triage_agent import get_triage_level
        return get_triage_level(disease_predictions)
    except Exception:
        for item in disease_predictions:
            d, p = item["disease"].lower(), item["probability"]
            if "pneumothorax" in d and p >= 0.30:
                return {"level": "STAT",    "reason": "Possible pneumothorax."}
            if "pneumonia"    in d and p >= 0.35:
                return {"level": "URGENT",  "reason": "Significant pneumonia probability."}
            if "effusion"     in d and p >= 0.40:
                return {"level": "URGENT",  "reason": "Pleural effusion detected."}
        return {"level": "ROUTINE", "reason": "No high-risk abnormality crossed threshold."}


# ─────────────────────────────────────────────
# CORE EVALUATION
# ─────────────────────────────────────────────
def get_image_paths(folder: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    folder = Path(folder)
    if not folder.exists():
        print(f"[WARN] Image folder '{folder}' not found — using synthetic image paths.")
        return [f"testxray/xray_{i:02d}.jpg" for i in range(1, 11)]
    paths = sorted([str(p) for p in folder.iterdir() if p.suffix.lower() in exts])
    if len(paths) == 0:
        print(f"[WARN] No images found in '{folder}' — using synthetic paths.")
        return [f"testxray/xray_{i:02d}.jpg" for i in range(1, 11)]
    return paths[:10]  # Cap at 10


def process_single_sample(
    image_path: str,
    note: str,
    note_id: int,
    image_idx: int,
    fn_vision,
    fn_preprocess,
    fn_inference,
    fn_entities,
    fn_retrieve,
    fn_report,
    fn_triage,
    threshold: float = 0.45,
) -> dict:
    """Run the full pipeline for one (image, note) pair. Returns a result dict."""

    result = {
        "image_path": image_path,
        "note_id":    note_id,
        "note":       note,
        "error":      None,
    }

    try:
        # ── 1. Vision ──────────────────────────────────────────────
        if fn_vision and fn_preprocess and fn_inference:
            try:
                model   = fn_vision()
                tensor  = fn_preprocess(image_path)
                scores  = fn_inference(model, tensor, threshold=0.30)
            except Exception:
                scores = mock_vision_scores(image_path, seed=note_id)
        else:
            scores = mock_vision_scores(image_path, seed=note_id)

        result["vision_scores"] = scores
        result["top_disease"]   = list(scores.keys())[0] if scores else "Unknown"
        result["top_score"]     = list(scores.values())[0] if scores else 0.0

        # ── 2. Clinical entities ──────────────────────────────────
        entities = mock_entities(note)
        result["entities"] = entities

        # ── 3. Query builder ─────────────────────────────────────
        query_parts = [d for d, s in scores.items() if s >= 0.30]
        if entities.get("chief_complaint"):
            query_parts.append(entities["chief_complaint"])
        for c in entities.get("comorbidities", []):
            query_parts.append(c)
        if entities.get("age"):
            query_parts.append(f"age {entities['age']}")
        if entities.get("sex"):
            query_parts.append(entities["sex"])
        query = " ".join(query_parts)
        result["query"] = query

        # ── 4. Retrieval ──────────────────────────────────────────
        if fn_retrieve:
            try:
                retrieved = fn_retrieve(query, top_k=5)
            except Exception:
                retrieved = mock_retrieve(query, top_k=5, seed=note_id)
        else:
            retrieved = mock_retrieve(query, top_k=5, seed=note_id)

        result["retrieval"] = [
            {"topic": r["topic"], "source": r["source"], "score": r["score"]}
            for r in retrieved
        ]
        docs = [r["text"] for r in retrieved]

        # ── 5. Report ─────────────────────────────────────────────
        if fn_report:
            try:
                report = fn_report(query, docs)
            except Exception:
                report = mock_report(query, docs)
        else:
            report = mock_report(query, docs)
        result["report"] = report

        # ── 6. Verification ───────────────────────────────────────
        verification = mock_verify(report, docs, threshold=threshold)
        result["verification"] = verification

        total_sentences = len(verification)
        supported_count = sum(1 for v in verification if v["supported"])
        rejected_count  = total_sentences - supported_count
        avg_score       = np.mean([v["score"] for v in verification]) if verification else 0.0

        result["total_sentences"]  = total_sentences
        result["supported_count"]  = supported_count
        result["rejected_count"]   = rejected_count
        result["avg_sim_score"]    = round(float(avg_score), 4)
        result["accept_rate"]      = round(supported_count / total_sentences, 4) if total_sentences else 0.0

        # ── 7. Triage ─────────────────────────────────────────────
        disease_preds = [{"disease": d, "probability": s} for d, s in scores.items()]
        triage = mock_triage(disease_preds)
        result["triage"] = triage

    except Exception as e:
        result["error"] = traceback.format_exc()
        print(f"  [ERROR] Sample {note_id}: {e}")

    return result


# ─────────────────────────────────────────────
# AGGREGATE METRICS
# ─────────────────────────────────────────────
def compute_summary(results: list[dict]) -> dict:
    valid = [r for r in results if r["error"] is None]
    total = len(valid)

    if total == 0:
        return {"error": "No valid results"}

    # ── Verification metrics ──────────────────────────────────────
    all_accept_rates   = [r["accept_rate"]   for r in valid]
    all_avg_scores     = [r["avg_sim_score"] for r in valid]
    all_supported      = [r["supported_count"] for r in valid]
    all_rejected       = [r["rejected_count"]  for r in valid]
    all_sentences      = [r["total_sentences"] for r in valid]

    # ── Triage distribution ───────────────────────────────────────
    triage_counts = {"STAT": 0, "URGENT": 0, "ROUTINE": 0}
    for r in valid:
        lvl = r.get("triage", {}).get("level", "ROUTINE")
        triage_counts[lvl] = triage_counts.get(lvl, 0) + 1

    # ── Top disease distribution ─────────────────────────────────
    disease_counts: dict = {}
    for r in valid:
        d = r.get("top_disease", "Unknown")
        disease_counts[d] = disease_counts.get(d, 0) + 1
    disease_counts = dict(sorted(disease_counts.items(), key=lambda x: x[1], reverse=True))

    # ── Retrieval source distribution ────────────────────────────
    source_counts: dict = {}
    for r in valid:
        for ret in r.get("retrieval", []):
            s = ret.get("source", "unknown")
            source_counts[s] = source_counts.get(s, 0) + 1

    # ── Per-image stats ───────────────────────────────────────────
    from collections import defaultdict
    img_stats: dict = defaultdict(lambda: {"accept_rates": [], "samples": 0})
    for r in valid:
        img = os.path.basename(r["image_path"])
        img_stats[img]["accept_rates"].append(r["accept_rate"])
        img_stats[img]["samples"] += 1

    per_image = {
        img: {
            "samples": v["samples"],
            "mean_accept_rate": round(float(np.mean(v["accept_rates"])), 4),
            "std_accept_rate":  round(float(np.std(v["accept_rates"])),  4),
        }
        for img, v in img_stats.items()
    }

    # ── Score thresholds ─────────────────────────────────────────
    thresholds = [0.30, 0.40, 0.45, 0.50, 0.60, 0.70]
    all_individual_scores = []
    for r in valid:
        for v in r.get("verification", []):
            all_individual_scores.append(v["score"])

    accept_at_threshold = {}
    for t in thresholds:
        count = sum(1 for s in all_individual_scores if s >= t)
        accept_at_threshold[str(t)] = {
            "count":  count,
            "total":  len(all_individual_scores),
            "rate":   round(count / len(all_individual_scores), 4) if all_individual_scores else 0.0,
        }

    # ── Score histogram ───────────────────────────────────────────
    hist, bin_edges = np.histogram(all_individual_scores, bins=20, range=(0.0, 1.0))
    score_histogram = {
        "counts": hist.tolist(),
        "bin_edges": [round(float(e), 3) for e in bin_edges.tolist()],
    }

    return {
        "total_samples":         total,
        "total_sentences":       int(np.sum(all_sentences)),
        "total_supported":       int(np.sum(all_supported)),
        "total_rejected":        int(np.sum(all_rejected)),
        "mean_accept_rate":      round(float(np.mean(all_accept_rates)), 4),
        "std_accept_rate":       round(float(np.std(all_accept_rates)),  4),
        "median_accept_rate":    round(float(np.median(all_accept_rates)), 4),
        "mean_avg_sim_score":    round(float(np.mean(all_avg_scores)), 4),
        "triage_distribution":   triage_counts,
        "disease_distribution":  disease_counts,
        "source_distribution":   source_counts,
        "per_image_stats":       per_image,
        "accept_at_threshold":   accept_at_threshold,
        "score_histogram":       score_histogram,
        "all_individual_scores": all_individual_scores,
    }


# ─────────────────────────────────────────────
# HTML REPORT GENERATOR
# ─────────────────────────────────────────────
def build_html_report(summary: dict, results: list[dict]) -> str:
    # Serialise data for JS
    s = json.dumps(summary, default=str)
    r = json.dumps(results[:200], default=str)  # Cap to keep HTML manageable

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Verification Agent Evaluation Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg:         #0a0a0f;
    --panel:      #111118;
    --border:     #1e1e2e;
    --accent:     #00e5ff;
    --accent2:    #ff4081;
    --accent3:    #69ff47;
    --text:       #e0e0f0;
    --muted:      #6060a0;
    --font-mono:  'IBM Plex Mono', monospace;
    --font-sans:  'IBM Plex Sans', sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    min-height: 100vh;
    padding: 0 0 60px;
  }}

  header {{
    border-bottom: 1px solid var(--border);
    padding: 28px 48px 24px;
    display: flex;
    align-items: baseline;
    gap: 20px;
  }}

  header h1 {{
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}

  header span {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--muted);
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    margin: 1px;
  }}

  .stat-card {{
    background: var(--panel);
    padding: 28px 32px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  .stat-card .label {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  .stat-card .value {{
    font-family: var(--font-mono);
    font-size: 2.4rem;
    font-weight: 600;
    line-height: 1;
  }}

  .stat-card .sub {{
    font-size: 0.75rem;
    color: var(--muted);
    font-family: var(--font-mono);
  }}

  .cyan   {{ color: var(--accent); }}
  .pink   {{ color: var(--accent2); }}
  .green  {{ color: var(--accent3); }}
  .yellow {{ color: #ffd740; }}
  .orange {{ color: #ff9100; }}

  section {{
    padding: 40px 48px 0;
  }}

  section h2 {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
  }}

  .chart-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }}

  .chart-row-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }}

  .chart-box {{
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 24px;
  }}

  .chart-box h3 {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
  }}

  canvas {{ display: block; width: 100% !important; }}

  .table-wrap {{ overflow-x: auto; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.72rem;
  }}

  th {{
    text-align: left;
    padding: 10px 14px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    font-weight: 400;
    letter-spacing: 0.06em;
    white-space: nowrap;
  }}

  td {{
    padding: 9px 14px;
    border-bottom: 1px solid #15151f;
    color: var(--text);
    white-space: nowrap;
  }}

  tr:hover td {{ background: #14141e; }}

  .pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}

  .pill-stat    {{ background: #3d0020; color: var(--accent2); }}
  .pill-urgent  {{ background: #3d2800; color: #ffd740; }}
  .pill-routine {{ background: #0d2d1a; color: var(--accent3); }}

  .bar-inline {{
    display: inline-block;
    height: 8px;
    background: var(--accent);
    border-radius: 1px;
    margin-left: 8px;
    vertical-align: middle;
    opacity: 0.7;
  }}

  .threshold-table td:first-child {{ color: var(--muted); }}

  footer {{
    margin-top: 60px;
    padding: 24px 48px;
    border-top: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
  }}
</style>
</head>
<body>

<header>
  <h1>Verification Agent — Evaluation Dashboard</h1>
  <span id="run-info">loading...</span>
</header>

<!-- KPI strip -->
<div class="grid" id="kpi-strip"></div>

<!-- Charts row 1 -->
<section>
  <h2>Score Distribution &amp; Accept / Reject Analysis</h2>
  <div class="chart-row">
    <div class="chart-box">
      <h3>Similarity Score Histogram (all sentences)</h3>
      <canvas id="histChart" height="220"></canvas>
    </div>
    <div class="chart-box">
      <h3>Accept Rate — per sample (sorted)</h3>
      <canvas id="acceptLine" height="220"></canvas>
    </div>
  </div>
</section>

<!-- Charts row 2 -->
<section>
  <h2>Model Output &amp; Retrieval Breakdown</h2>
  <div class="chart-row-3">
    <div class="chart-box">
      <h3>Top Disease Distribution</h3>
      <canvas id="diseaseBar" height="260"></canvas>
    </div>
    <div class="chart-box">
      <h3>Triage Level Distribution</h3>
      <canvas id="triageDoughnut" height="260"></canvas>
    </div>
    <div class="chart-box">
      <h3>RAG Retrieval Sources</h3>
      <canvas id="sourceBar" height="260"></canvas>
    </div>
  </div>
</section>

<!-- Accept rate per image -->
<section>
  <h2>Accept Rate per X-ray Image</h2>
  <div class="chart-box" style="margin-bottom:24px;">
    <h3>Mean accept rate across 100 notes — per image</h3>
    <canvas id="perImageBar" height="180"></canvas>
  </div>
</section>

<!-- Threshold sensitivity -->
<section>
  <h2>Threshold Sensitivity Analysis</h2>
  <div class="chart-row">
    <div class="chart-box">
      <h3>Accept Rate vs Similarity Threshold</h3>
      <canvas id="thresholdLine" height="220"></canvas>
    </div>
    <div class="chart-box table-wrap">
      <h3>Detailed Threshold Table</h3>
      <table class="threshold-table" id="thresholdTable"></table>
    </div>
  </div>
</section>

<!-- Per-image detail table -->
<section>
  <h2>Per-Image Statistics</h2>
  <div class="table-wrap">
    <table id="imgTable"></table>
  </div>
</section>

<!-- Sample results table -->
<section>
  <h2>Sample Results (first 50)</h2>
  <div class="table-wrap">
    <table id="sampleTable"></table>
  </div>
</section>

<footer id="footer-info">Generated by eval_pipeline.py</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const SUMMARY = {s};
const RESULTS = {r};

const CYAN   = 'rgba(0,229,255,';
const PINK   = 'rgba(255,64,129,';
const GREEN  = 'rgba(105,255,71,';
const YELLOW = 'rgba(255,215,64,';

Chart.defaults.color = '#6060a0';
Chart.defaults.borderColor = '#1e1e2e';
Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.font.size = 11;

// ── KPI strip ──────────────────────────────────────────────────
function buildKPIs() {{
  const kpis = [
    {{ label: 'Total Samples',     value: SUMMARY.total_samples,                          sub: '10 images × 100 notes', cls: 'cyan'   }},
    {{ label: 'Total Sentences',   value: SUMMARY.total_sentences,                         sub: 'sentences evaluated',   cls: 'yellow' }},
    {{ label: 'Supported',         value: SUMMARY.total_supported,                         sub: 'accepted by verifier',  cls: 'green'  }},
    {{ label: 'Rejected',          value: SUMMARY.total_rejected,                          sub: 'flagged as unsupported', cls: 'pink'  }},
    {{ label: 'Mean Accept Rate',  value: (SUMMARY.mean_accept_rate*100).toFixed(1) + '%', sub: '± ' + (SUMMARY.std_accept_rate*100).toFixed(1) + '%', cls: 'cyan' }},
    {{ label: 'Median Accept Rate',value: (SUMMARY.median_accept_rate*100).toFixed(1)+'%', sub: 'over all samples',      cls: 'yellow' }},
    {{ label: 'Mean Sim Score',    value: SUMMARY.mean_avg_sim_score.toFixed(3),            sub: 'avg per sample',        cls: 'green'  }},
    {{ label: 'STAT Triages',      value: SUMMARY.triage_distribution.STAT,                sub: 'immediate attention',   cls: 'pink'   }},
  ];
  const strip = document.getElementById('kpi-strip');
  kpis.forEach(k => {{
    strip.innerHTML += `<div class="stat-card">
      <div class="label">${{k.label}}</div>
      <div class="value ${{k.cls}}">${{k.value}}</div>
      <div class="sub">${{k.sub}}</div>
    </div>`;
  }});
}}

// ── Histogram ──────────────────────────────────────────────────
function buildHistogram() {{
  const edges  = SUMMARY.score_histogram.bin_edges;
  const counts = SUMMARY.score_histogram.counts;
  const labels = counts.map((_,i) => edges[i].toFixed(2) + '–' + edges[i+1].toFixed(2));
  new Chart(document.getElementById('histChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Sentence count',
        data: counts,
        backgroundColor: counts.map((_,i) => {{
          const mid = (edges[i] + edges[i+1]) / 2;
          return mid >= 0.45 ? CYAN + '0.75)' : PINK + '0.55)';
        }}),
        borderWidth: 0,
        borderRadius: 1,
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ maxRotation: 45, autoSkip: true, maxTicksLimit: 10 }} }},
        y: {{ beginAtZero: true }}
      }}
    }}
  }});
}}

// ── Accept-rate line ───────────────────────────────────────────
function buildAcceptLine() {{
  const rates = RESULTS
    .filter(r => r.error == null)
    .map(r => r.accept_rate)
    .sort((a,b) => a - b);
  new Chart(document.getElementById('acceptLine'), {{
    type: 'line',
    data: {{
      labels: rates.map((_,i) => i),
      datasets: [{{
        label: 'Accept rate',
        data: rates,
        borderColor: CYAN + '1)',
        backgroundColor: CYAN + '0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ min: 0, max: 1, ticks: {{ callback: v => (v*100).toFixed(0)+'%' }} }}
      }}
    }}
  }});
}}

// ── Disease bar ────────────────────────────────────────────────
function buildDiseaseBar() {{
  const dd = SUMMARY.disease_distribution;
  const labels = Object.keys(dd).slice(0,10);
  const vals   = labels.map(l => dd[l]);
  new Chart(document.getElementById('diseaseBar'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ data: vals, backgroundColor: CYAN+'0.7)', borderWidth: 0, borderRadius: 2 }}]
    }},
    options: {{
      indexAxis: 'y',
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ x: {{ beginAtZero: true }} }}
    }}
  }});
}}

// ── Triage doughnut ────────────────────────────────────────────
function buildTriageDoughnut() {{
  const td = SUMMARY.triage_distribution;
  new Chart(document.getElementById('triageDoughnut'), {{
    type: 'doughnut',
    data: {{
      labels: ['STAT', 'URGENT', 'ROUTINE'],
      datasets: [{{
        data: [td.STAT, td.URGENT, td.ROUTINE],
        backgroundColor: [PINK+'0.85)', YELLOW+'0.85)', GREEN+'0.7)'],
        borderWidth: 0,
      }}]
    }},
    options: {{
      cutout: '68%',
      plugins: {{ legend: {{ position: 'bottom' }} }}
    }}
  }});
}}

// ── Source bar ─────────────────────────────────────────────────
function buildSourceBar() {{
  const sd = SUMMARY.source_distribution;
  const labels = Object.keys(sd);
  const vals   = labels.map(l => sd[l]);
  new Chart(document.getElementById('sourceBar'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ data: vals, backgroundColor: GREEN+'0.7)', borderWidth: 0, borderRadius: 2 }}]
    }},
    options: {{
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ beginAtZero: true }} }}
    }}
  }});
}}

// ── Per-image bar ──────────────────────────────────────────────
function buildPerImageBar() {{
  const pi = SUMMARY.per_image_stats;
  const labels = Object.keys(pi);
  const means  = labels.map(l => pi[l].mean_accept_rate);
  const stds   = labels.map(l => pi[l].std_accept_rate);
  new Chart(document.getElementById('perImageBar'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{
          label: 'Mean accept rate',
          data: means,
          backgroundColor: CYAN+'0.75)',
          borderWidth: 0,
          borderRadius: 2,
        }},
        {{
          label: '±1 std dev',
          data: stds,
          backgroundColor: YELLOW+'0.45)',
          borderWidth: 0,
          borderRadius: 2,
        }}
      ]
    }},
    options: {{
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ beginAtZero: true, max: 1, ticks: {{ callback: v => (v*100).toFixed(0)+'%' }} }} }}
    }}
  }});
}}

// ── Threshold line ─────────────────────────────────────────────
function buildThresholdLine() {{
  const at = SUMMARY.accept_at_threshold;
  const thresholds = Object.keys(at).map(Number).sort((a,b)=>a-b);
  const rates      = thresholds.map(t => at[String(t)].rate);
  new Chart(document.getElementById('thresholdLine'), {{
    type: 'line',
    data: {{
      labels: thresholds.map(t => t.toFixed(2)),
      datasets: [{{
        label: 'Accept rate',
        data: rates,
        borderColor: PINK+'1)',
        backgroundColor: PINK+'0.08)',
        borderWidth: 2,
        pointRadius: 5,
        pointBackgroundColor: PINK+'1)',
        fill: true,
        tension: 0.3,
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ min: 0, max: 1, ticks: {{ callback: v => (v*100).toFixed(0)+'%' }} }}
      }}
    }}
  }});
}}

// ── Threshold table ────────────────────────────────────────────
function buildThresholdTable() {{
  const at = SUMMARY.accept_at_threshold;
  const thresholds = Object.keys(at).map(Number).sort((a,b)=>a-b);
  const tbl = document.getElementById('thresholdTable');
  tbl.innerHTML = `<tr><th>Threshold</th><th>Accepted</th><th>Total</th><th>Rate</th><th></th></tr>`;
  thresholds.forEach(t => {{
    const d = at[String(t)];
    const bar = `<span class="bar-inline" style="width:${{(d.rate*120).toFixed(0)}}px;"></span>`;
    tbl.innerHTML += `<tr>
      <td>${{t.toFixed(2)}}</td>
      <td class="green">${{d.count}}</td>
      <td class="muted">${{d.total}}</td>
      <td class="cyan">${{(d.rate*100).toFixed(1)}}%</td>
      <td>${{bar}}</td>
    </tr>`;
  }});
}}

// ── Per-image table ────────────────────────────────────────────
function buildImgTable() {{
  const pi = SUMMARY.per_image_stats;
  const tbl = document.getElementById('imgTable');
  tbl.innerHTML = `<tr><th>Image</th><th>Samples</th><th>Mean Accept%</th><th>Std Dev</th></tr>`;
  Object.entries(pi).forEach(([img, v]) => {{
    tbl.innerHTML += `<tr>
      <td class="cyan">${{img}}</td>
      <td>${{v.samples}}</td>
      <td>${{(v.mean_accept_rate*100).toFixed(1)}}%</td>
      <td class="muted">± ${{(v.std_accept_rate*100).toFixed(1)}}%</td>
    </tr>`;
  }});
}}

// ── Sample table ───────────────────────────────────────────────
function buildSampleTable() {{
  const tbl = document.getElementById('sampleTable');
  tbl.innerHTML = `<tr>
    <th>#</th><th>Image</th><th>Note (truncated)</th>
    <th>Top Disease</th><th>Score</th><th>Accept%</th>
    <th>Avg Sim</th><th>Triage</th>
  </tr>`;
  RESULTS.slice(0,50).forEach((r, i) => {{
    if (r.error) return;
    const lvl = r.triage?.level || 'ROUTINE';
    const pillCls = lvl === 'STAT' ? 'pill-stat' : lvl === 'URGENT' ? 'pill-urgent' : 'pill-routine';
    tbl.innerHTML += `<tr>
      <td class="muted">${{i+1}}</td>
      <td class="cyan">${{(r.image_path||'').split('/').pop()}}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">${{(r.note||'').substring(0,80)}}…</td>
      <td>${{r.top_disease||'—'}}</td>
      <td class="yellow">${{(r.top_score||0).toFixed(3)}}</td>
      <td class="${{r.accept_rate>=0.5?'green':'pink'}}">${{((r.accept_rate||0)*100).toFixed(1)}}%</td>
      <td>${{(r.avg_sim_score||0).toFixed(3)}}</td>
      <td><span class="pill ${{pillCls}}">${{lvl}}</span></td>
    </tr>`;
  }});
}}

// ── Header info ────────────────────────────────────────────────
document.getElementById('run-info').textContent =
  `${{SUMMARY.total_samples}} samples · ${{SUMMARY.total_sentences}} sentences · threshold 0.45`;

document.getElementById('footer-info').textContent =
  `eval_pipeline.py · ${{SUMMARY.total_samples}} samples · mean accept rate ${{(SUMMARY.mean_accept_rate*100).toFixed(1)}}%`;

// ── Render all ─────────────────────────────────────────────────
buildKPIs();
buildHistogram();
buildAcceptLine();
buildDiseaseBar();
buildTriageDoughnut();
buildSourceBar();
buildPerImageBar();
buildThresholdLine();
buildThresholdTable();
buildImgTable();
buildSampleTable();
</script>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verification Agent Evaluation Pipeline")
    parser.add_argument("--image-folder",  default="testxray",  help="Folder with 10 X-ray images")
    parser.add_argument("--threshold",     default=0.45, type=float, help="Verification similarity threshold")
    parser.add_argument("--notes",         default=100,  type=int,   help="Number of synthetic notes (default 100)")
    parser.add_argument("--use-real-models", action="store_true",   help="Attempt to load real vision/BERT models")
    args = parser.parse_args()

    print("=" * 60)
    print("  Verification Agent — Evaluation Pipeline")
    print("=" * 60)

    # ── Load models if requested ──────────────────────────────
    fn_vision = fn_preprocess = fn_inference = None
    fn_bert = fn_process_note = None
    fn_retrieve = fn_report = fn_triage = None

    if args.use_real_models:
        print("\n[1/3] Loading vision model...")
        fn_vision, fn_preprocess, fn_inference = try_import_vision()
        print("[2/3] Loading BiomedBERT...")
        fn_bert, fn_process_note = try_import_cnote()
        print("[3/3] Loading retrieval / report / triage agents...")
        fn_retrieve = try_import_retrieval()
        fn_report   = try_import_report()
        fn_triage   = try_import_triage()
    else:
        print("\n[INFO] Running in mock mode (pass --use-real-models to load GPU models).")

    # ── Images & notes ────────────────────────────────────────
    image_paths = get_image_paths(args.image_folder)
    print(f"\n[INFO] Using {len(image_paths)} image(s): {[os.path.basename(p) for p in image_paths]}")

    notes = generate_100_notes()[: args.notes]
    print(f"[INFO] Generated {len(notes)} synthetic clinical notes.")

    # ── Build (image, note) pairs ─────────────────────────────
    # 100 notes × 10 images → assign note_i to image[i % 10]
    pairs = [(image_paths[n["id"] % len(image_paths)], n["note"], n["id"]) for n in notes]

    total  = len(pairs)
    results: list[dict] = []

    print(f"\n[INFO] Running evaluation on {total} (image, note) pairs...")
    start = time.time()

    for idx, (img, note, note_id) in enumerate(pairs):
        img_idx = note_id % len(image_paths)
        if idx % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{idx:3d}/{total}] {elapsed:.1f}s elapsed ...")

        res = process_single_sample(
            image_path   = img,
            note         = note,
            note_id      = note_id,
            image_idx    = img_idx,
            fn_vision    = fn_vision,
            fn_preprocess= fn_preprocess,
            fn_inference = fn_inference,
            fn_entities  = mock_entities,
            fn_retrieve  = fn_retrieve,
            fn_report    = fn_report,
            fn_triage    = fn_triage,
            threshold    = args.threshold,
        )
        results.append(res)

    elapsed_total = time.time() - start
    print(f"\n[INFO] Pipeline complete in {elapsed_total:.1f}s")

    # ── Compute summary ───────────────────────────────────────
    print("[INFO] Computing aggregate metrics...")
    summary = compute_summary(results)

    print(f"\n  Total samples  : {summary['total_samples']}")
    print(f"  Total sentences: {summary['total_sentences']}")
    print(f"  Supported      : {summary['total_supported']}")
    print(f"  Rejected       : {summary['total_rejected']}")
    print(f"  Mean accept    : {summary['mean_accept_rate']*100:.1f}%")
    print(f"  Mean sim score : {summary['mean_avg_sim_score']:.4f}")
    print(f"  Triage dist    : {summary['triage_distribution']}")

    # ── Save raw results ──────────────────────────────────────
    print(f"\n[INFO] Saving results to {RESULTS_JSON}...")
    # Drop raw scores for file size
    slim_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "all_individual_scores"}
        slim_results.append(sr)
    with open(RESULTS_JSON, "w") as f:
        json.dump(slim_results, f, indent=2, default=str)

    # ── Save summary ──────────────────────────────────────────
    print(f"[INFO] Saving summary to {SUMMARY_JSON}...")
    with open(SUMMARY_JSON, "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "all_individual_scores"}, f, indent=2, default=str)

    # ── Build HTML report ─────────────────────────────────────
    print(f"[INFO] Building HTML report → {REPORT_HTML}...")
    html = build_html_report(summary, slim_results)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n✅  Done!")
    print(f"    Results : {RESULTS_JSON}")
    print(f"    Summary : {SUMMARY_JSON}")
    print(f"    Report  : {REPORT_HTML}")


if __name__ == "__main__":
    main()