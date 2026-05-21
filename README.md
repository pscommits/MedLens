# 🩺 MedLens — The Verifiable Radiology Co-Pilot

> *Every finding cited. Every region highlighted. Every decision auditable.*

A hackathon-ready, multi-agent, multimodal AI system that takes a chest X-ray
plus an optional clinical note and returns — in under five seconds — a
**triage banner**, a **structured radiology report with inline citations**,
and a **GradCAM heatmap** showing exactly what the model looked at.

Built around five specialist agents coordinated by a lightweight FastAPI
orchestrator, with a dedicated verification agent that cross-checks every
generated claim against retrieved medical literature.

---

## 🎯 What this repo contains

```
medlens/
├── backend/                    # FastAPI orchestrator + 5 specialist agents
│   ├── app/
│   │   ├── main.py             # Orchestrator (asyncio.gather fan-out)
│   │   ├── schemas.py          # Pydantic response models
│   │   └── agents/
│   │       ├── vision_agent.py         # TorchXRayVision + GradCAM
│   │       ├── context_agent.py        # BiomedBERT + entity extraction
│   │       ├── retrieval_agent.py      # ChromaDB + all-MiniLM-L6-v2
│   │       ├── report_agent.py         # Groq Llama-3.3-70B with [n] citations
│   │       └── verification_agent.py   # Claim verification + rule-based triage
│   ├── data/
│   │   └── chroma_store/       # ← your existing vector DB goes here
│   ├── .env                    # Groq API key + config (NOT committed)
│   ├── .env.example
│   ├── requirements.txt
│   ├── run.sh / run.bat
│
├── frontend/                   # Streamlit UI
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── run.sh / run.bat
│
├── scripts/
│   └── build_index.py          # Optional ChromaDB rebuilder
│
├── docs/
│   └── ARCHITECTURE.md         # Diagrams + design notes
│
├── .gitignore
└── README.md                   # ← you are here
```

---

## ⚡ Quick start (5 minutes, two terminals)

### Prerequisites
- **Python 3.10 or 3.11** (3.12 also works but some torch wheels are slower)
- ~5 GB free disk space (models + chroma_store)
- Your existing populated `chroma_store` folder from your earlier work
- A free [Groq API key](https://console.groq.com) — *the included `.env`
  already has your key from `report_agent.py` so you can skip this step
  unless you want to rotate it*

### Step 1 — Drop in your ChromaDB

Copy your existing populated `chroma_store` folder into the backend:

```bash
# Linux / macOS
cp -r /path/to/your/existing/chroma_store medlens/backend/data/chroma_store
```

```powershell
# Windows PowerShell
Copy-Item -Recurse C:\path\to\your\existing\chroma_store .\medlens\backend\data\chroma_store
```

The retrieval agent expects a collection named `medical_knowledge` (matches
your `retrieval_agent.py`). If yours is named differently, edit
`backend/.env` and set `CHROMA_COLLECTION=your_name`.

### Step 2 — Backend (Terminal 1)

```bash
cd medlens/backend

# Create and activate a virtual environment
python -m venv venv

# Activate it
# Linux/Mac:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies (takes 3–5 minutes the first time)
pip install -r requirements.txt

# Start the FastAPI server
./run.sh                # Linux / Mac
# or
run.bat                 # Windows
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Open <http://127.0.0.1:8000/docs> in a browser to see the Swagger UI.

### Step 3 — Frontend (Terminal 2)

In a **new** terminal:

```bash
cd medlens/frontend

# Create venv (separate from backend)
python -m venv venv

# Activate
source venv/bin/activate              # Linux / Mac
# or
.\venv\Scripts\Activate.ps1           # Windows

# Install (fast — Streamlit + requests)
pip install -r requirements.txt

# Launch
./run.sh                # Linux / Mac
# or
run.bat                 # Windows
```

Streamlit opens automatically at <http://localhost:8501>.

### Step 4 — Analyze your first X-ray

1. In the sidebar: upload any chest X-ray (PNG/JPG)
2. The clinical note is pre-filled with a sample — replace it or leave it
3. Click **Analyze X-ray**
4. Wait 8–15 seconds the *first* time (models loading), 3–6 seconds after

---

## 🧪 Testing the backend directly (without Streamlit)

You can also hit the API from the Swagger UI at <http://127.0.0.1:8000/docs>:

1. Expand `POST /api/v1/analyze`
2. Click **Try it out**
3. Upload an image, type a clinical note, set `session_id` to anything
4. Click **Execute**

Or with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -F "image=@/path/to/xray.jpg" \
  -F "clinical_note=45M with fever, cough, SOB. Hx COPD." \
  -F "session_id=test-1"
```

---

## 🎤 Demo script (90 seconds for judges)

Memorize this — it follows the storyboard from the project blueprint.

| Time | Action | What to say |
|------|--------|-------------|
| 0:00–0:15 | Show the title page + sidebar | *"Radiologists miss 4% of urgent CXR findings due to fatigue. Existing AI tools are black boxes nobody trusts. MedLens fixes that."* |
| 0:15–0:35 | Upload a dramatic case (pneumothorax or large effusion). Paste a real clinical note like *"55M, sudden dyspnea, decreased breath sounds R-sided, hx COPD"* | *"One X-ray, one note. Five AI agents are now running in parallel."* |
| 0:35–0:55 | Result appears. Point to the **triage banner**. | *"STAT — possible pneumothorax. Justification right there. No clicking, no reading. The ER nurse knows in one second."* |
| 0:55–1:15 | Point to the **GradCAM heatmap** + the **citations panel**. Open one citation. | *"Every red region drove a finding. Every finding cites a real medical passage you can read. This is what vanilla ChatGPT cannot do."* |
| 1:15–1:30 | Open the **verification panel**. Highlight a "Supported" claim and any "Unsupported" one. | *"And the verifier flags any sentence we couldn't ground in evidence. That's the trust gate that turns this from a demo into a defensible medical tool."* |

**Closing line:** *"Every claim cited. Every region highlighted. Every decision auditable."*

### Demo tips

- **Pick a dramatic image.** Pneumothorax, large pleural effusion, or
  obvious pneumonia from the NIH ChestX-ray14 set produce unmistakable
  heatmaps. Subtle findings make the GradCAM look like a blob — skip those.
- **Pre-warm the backend.** Run one analysis before the judges arrive. The
  *second* request is the fast one.
- **Pre-test your network.** If the venue WiFi blocks Groq, switch to a
  hotspot before the demo starts.
- **Keep a backup screenshot.** If anything fails live, you have a
  pre-rendered result to show.

---

## ⚙️ Configuration

### `backend/.env`

```bash
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile     # or llama-3.1-8b-instant for speed
GROQ_TEMPERATURE=0.2

# Optional overrides
CHROMA_PATH=/absolute/path/to/chroma_store
CHROMA_COLLECTION=medical_knowledge
```

### Frontend backend URL

By default Streamlit talks to `http://127.0.0.1:8000`. To point at a
remote backend (e.g. an ngrok tunnel for a phone demo):

```bash
export MEDLENS_BACKEND_URL=https://abc123.ngrok.io
streamlit run streamlit_app.py
```

Or just edit the text box in the Streamlit sidebar at runtime.

---

## 🏗️ Architecture at a glance

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full diagram and
design notes. Short version:

```
        Streamlit UI
             │
             ▼ POST /api/v1/analyze
   ┌─────────────────────────┐
   │  FastAPI Orchestrator   │
   │                         │
   │  Phase 1: Vision ║ Context   (parallel)
   │  Phase 2: Retrieval
   │  Phase 3: Report (Groq LLM)
   │  Phase 4: Verify + Triage
   └─────────────────────────┘
             │
             ▼ AnalysisResponse JSON
        Streamlit renders:
        triage · heatmap · cited report · verification · latencies
```

---

## 🔧 Troubleshooting

### `[retrieval_agent] ChromaDB store not found at: .../backend/data/chroma_store`

You haven't copied your existing `chroma_store` into `backend/data/`. Either:
- Copy it: `cp -r /path/to/chroma_store backend/data/chroma_store`
- Or set `CHROMA_PATH=/absolute/path` in `backend/.env`

### `Collection medical_knowledge does not exist`

Your collection has a different name. Set `CHROMA_COLLECTION=actual_name`
in `backend/.env`.

### Backend stuck on `[vision_agent] Loading TorchXRayVision DenseNet...`

This is **normal on the first request**. The model is being downloaded from
the TorchXRayVision CDN (~100 MB) and cached to `~/.cache/torchxrayvision`.
Subsequent runs use the cache and load in ~3 seconds.

Same applies to BiomedBERT (~440 MB on first run, cached to
`~/.cache/huggingface`).

### `[report_agent] GROQ_API_KEY is not set`

Make sure `backend/.env` exists and contains your key. The orchestrator
loads `.env` via `python-dotenv` on startup, so you must restart uvicorn
after editing it.

### `groq.RateLimitError` or `groq.APIError`

Groq's free tier has per-minute rate limits. Either:
- Wait 60 seconds and try again
- Switch to a smaller model: `GROQ_MODEL=llama-3.1-8b-instant`
- Upgrade your Groq plan

### Streamlit says "✗ Backend offline"

The frontend is checking `GET /health` on the backend URL. Confirm the
backend is actually running by visiting <http://127.0.0.1:8000/docs> directly.

### `ModuleNotFoundError: No module named 'app'`

You're running uvicorn from the wrong directory. `cd backend` first, then
`./run.sh`. The `app.main:app` module path is relative to the backend folder.

### Heatmap is a blob covering the whole lung

This usually means the X-ray was very low contrast or already heavily
windowed. Try a different image. The blueprint also recommends applying a
lung mask before GradCAM as a v2 improvement.

### "Address already in use" on port 8000

Another process is using port 8000. Either kill it, or change the port:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Then update the Streamlit sidebar URL to `http://127.0.0.1:8001`.

---

## 📊 What's in each agent

| Agent | Model | Size | First-load time | Per-request time |
|-------|-------|------|-----------------|------------------|
| Vision | TorchXRayVision DenseNet-121 | ~100 MB | ~3 s | ~600 ms |
| Context | BiomedBERT (PubMedBERT) | ~440 MB | ~8 s | ~200 ms |
| Retrieval | all-MiniLM-L6-v2 + ChromaDB | ~90 MB | ~2 s | ~150 ms |
| Report | Groq Llama-3.3-70B | API | 0 s | ~1.5 s |
| Verification | all-MiniLM-L6-v2 (shared) | shared | 0 s | ~300 ms |

Total cold-start: ~15 s. Total warm request: ~3.5 s.

---

## 🚀 Future extensions

The blueprint lists several v2 ideas. The easiest to add:

1. **Anatomical region labels** — intersect each heatmap with a lung-zone
   mask (RUL, RML, RLL, LUL, lingula, LLL) and add the label to citations.
2. **Multiple heatmaps** — currently we render only the top-1 pathology.
   Extend the vision agent to return one heatmap per above-threshold finding.
3. **Audit log** — write every `analyze` request to a SQLite log with
   request hash, latencies, and triage decision. Useful for the
   regulatory-pathway story.
4. **Offline mode** — swap the Groq agent for a local Ollama-served Phi-3.5
   when the demo machine has no internet. The blueprint already recommends
   this as a fallback.
5. **Next.js polish layer** — once the Streamlit demo is locked in, build a
   Next.js + Tailwind frontend that hits the same backend. The API contract
   is already locked by `schemas.py`.

---

## 🧠 Credit & references

This project synthesizes findings from five surveyed papers on multimodal
medical AI, XAI in radiology, and clinical agent systems — see the project
blueprint PDF for the full citation map and design rationale.

Built for hackathon delivery in 48 hours. Zero model training required.
Every component is free, open, and reproducible on a laptop.

---

## 📜 License

This repository is provided as-is for hackathon and educational use.

**⚠️ This is not a medical device.** It is a research / educational
prototype. It is not approved by the FDA, EMA, or any regulatory body.
A clinician must always be in the loop.
