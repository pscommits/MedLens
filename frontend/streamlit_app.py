"""
streamlit_app.py
----------------
MedLens frontend — single-page Streamlit UI.

Layout:
    Sidebar       : upload area, clinical note, "Analyze" button, backend health
    Top banner    : color-coded triage card (STAT / URGENT / ROUTINE)
    Left column   : original X-ray + GradCAM heatmap (side by side)
    Middle column : structured report with inline [n] citations,
                    verification badges per sentence, expandable citation modals
    Right column  : pathology scores, clinical entities, latency breakdown,
                    retrieved evidence

Run with:
    streamlit run streamlit_app.py
"""

import os
import io
import base64
import time
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEFAULT_BACKEND = os.environ.get("MEDLENS_BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="MedLens — Radiology Co-Pilot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Light styling — keeps Streamlit's defaults but tightens spacing and
# styles the triage banner. No external assets.
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .triage-banner {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .triage-STAT    { background: linear-gradient(135deg, #c0392b, #e74c3c); }
    .triage-URGENT  { background: linear-gradient(135deg, #d35400, #f39c12); }
    .triage-ROUTINE { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .triage-justification {
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.25rem;
    }
    .verification-supported  { color: #27ae60; font-weight: 600; }
    .verification-unsupported { color: #c0392b; font-weight: 600; }
    .citation-marker {
        background-color: #3498db;
        color: white;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.8em;
        font-weight: 600;
        cursor: help;
    }
    .latency-badge {
        background: #ecf0f1;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helpers
# =============================================================================

def check_backend(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_analyze(url: str, image_bytes: bytes, image_name: str, clinical_note: str):
    """POST to the orchestrator. Returns dict or raises with a clean message."""
    files = {"image": (image_name, image_bytes, "image/png")}
    data  = {"clinical_note": clinical_note, "session_id": "streamlit-demo"}
    r = requests.post(
        f"{url}/api/v1/analyze",
        files=files,
        data=data,
        timeout=180,
    )
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Backend returned {r.status_code}: {detail}")
    return r.json()


def b64_data_uri_to_bytes(data_uri: str) -> bytes:
    """Strip the 'data:image/png;base64,' prefix and decode."""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    return base64.b64decode(data_uri)


# =============================================================================
# Sidebar — controls
# =============================================================================

with st.sidebar:
    st.title("🩺 MedLens")
    st.caption("Multimodal radiology co-pilot · v1.0")

    backend_url = st.text_input(
        "Backend URL",
        value=DEFAULT_BACKEND,
        help="FastAPI orchestrator endpoint",
    )

    if check_backend(backend_url):
        st.success("✓ Backend online")
    else:
        st.error("✗ Backend offline")
        st.caption("Start it with `./run.sh` in the backend folder.")

    st.divider()

    st.subheader("Input")
    uploaded = st.file_uploader(
        "Chest X-ray (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    clinical_note = st.text_area(
        "Clinical note (optional)",
        value="45M with fever, cough, chest pain and shortness of breath. History of COPD.",
        height=120,
        help="Free-text; age/sex/comorbidities are parsed automatically.",
    )

    analyze_clicked = st.button("Analyze X-ray", type="primary", use_container_width=True)

    st.divider()
    with st.expander("ℹ️  About"):
        st.markdown("""
**MedLens** runs five specialist AI agents in parallel:

1. **Vision** — TorchXRayVision DenseNet detects 18 pathologies
2. **Context** — BiomedBERT extracts age, sex, symptoms
3. **Retrieval** — ChromaDB pulls grounded medical evidence
4. **Report** — Groq LLM drafts a cited radiology report
5. **Verification & Triage** — Every claim is checked, urgency is assigned

Built for hackathon judges who appreciate working software over slides.
        """)


# =============================================================================
# Main pane
# =============================================================================

st.title("MedLens — The Verifiable Radiology Co-Pilot")
st.caption("Every finding cited · Every region highlighted · Every decision auditable")

if not analyze_clicked or uploaded is None:
    st.info("👈 Upload a chest X-ray in the sidebar and click **Analyze X-ray** to begin.")

    with st.expander("📖  How to read the results", expanded=False):
        st.markdown("""
After you click Analyze you'll see:
- A **color-coded triage banner** at the top (STAT / URGENT / ROUTINE)
- The **original X-ray** alongside a **GradCAM heatmap** showing what the
  model looked at
- A **structured radiology report** with inline `[1]` `[2]` citation markers
  you can click to see the source passage
- A **verification panel** flagging any claim the model couldn't ground
  in retrieved evidence
- The **pathology probability table** and the **latency breakdown**
        """)
    st.stop()

# -------------------------------------------------------------------------
# Run analysis
# -------------------------------------------------------------------------

image_bytes = uploaded.read()

with st.spinner("Running 5-agent pipeline... (~3–8 s)"):
    t0 = time.time()
    try:
        result = call_analyze(backend_url, image_bytes, uploaded.name, clinical_note)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()
    wall = time.time() - t0

# -------------------------------------------------------------------------
# Triage banner
# -------------------------------------------------------------------------

triage_level = result.get("triage_level", "ROUTINE")
triage_just  = result.get("triage_justification", "")

emoji = {"STAT": "🚨", "URGENT": "⚠️", "ROUTINE": "✅"}.get(triage_level, "ℹ️")
st.markdown(
    f"""
    <div class="triage-banner triage-{triage_level}">
        {emoji} TRIAGE: {triage_level}
        <div class="triage-justification">{triage_just}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# Three-column main layout
# -------------------------------------------------------------------------

col_img, col_report, col_meta = st.columns([1.1, 1.6, 1.1], gap="medium")

# ----- LEFT COLUMN — images -----
with col_img:
    st.markdown('<div class="section-header">Image + GradCAM Heatmap</div>', unsafe_allow_html=True)

    st.image(image_bytes, caption="Original X-ray", use_container_width=True)

    heatmap_uri = result.get("heatmap_base64", "")
    if heatmap_uri:
        try:
            heatmap_bytes = b64_data_uri_to_bytes(heatmap_uri)
            st.image(
                heatmap_bytes,
                caption="GradCAM — top-1 pathology",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not decode heatmap: {e}")

# ----- MIDDLE COLUMN — structured report -----
with col_report:
    st.markdown('<div class="section-header">Structured Report</div>', unsafe_allow_html=True)

    report = result.get("structured_report", {})

    st.markdown("**Impression**")
    st.write(report.get("impression", ""))

    st.markdown("**Findings**")
    st.write(report.get("findings", ""))

    st.markdown("**Recommendations**")
    st.write(report.get("recommendations", ""))

    # Verification panel
    st.markdown('<div class="section-header">Claim Verification</div>', unsafe_allow_html=True)
    verification = result.get("verification", [])
    if not verification:
        st.caption("No verifiable claims found.")
    else:
        n_supported    = sum(1 for v in verification if v.get("supported"))
        n_total        = len(verification)
        pct = 100 * n_supported // max(n_total, 1)
        st.markdown(
            f"**{n_supported}/{n_total}** claims supported by retrieved evidence "
            f"({pct}%)"
        )

        for i, v in enumerate(verification, 1):
            badge_class = "verification-supported" if v.get("supported") else "verification-unsupported"
            badge_text  = "✓ Supported" if v.get("supported") else "✗ Unsupported"
            score = v.get("score", 0.0)
            with st.expander(
                f"Claim {i}  ·  Score {score:.2f}  ·  "
                f"{'✓ Supported' if v.get('supported') else '✗ Unsupported'}",
                expanded=False,
            ):
                st.markdown(f"**Sentence:** {v.get('sentence', '')}")
                st.markdown(
                    f"**Verdict:** "
                    f"<span class='{badge_class}'>{badge_text}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Best-matching evidence:**")
                st.caption(v.get("evidence", ""))

    # Citations
    st.markdown('<div class="section-header">Citations</div>', unsafe_allow_html=True)
    citations = result.get("citations", [])
    if not citations:
        st.caption("No inline citations were extracted from the report.")
    else:
        for cit in citations:
            with st.expander(f"{cit['marker']}  ·  {cit['source']}  ·  {cit['passage_id']}"):
                st.write(cit["passage"])

# ----- RIGHT COLUMN — metadata -----
with col_meta:
    st.markdown('<div class="section-header">Pathology Probabilities</div>', unsafe_allow_html=True)
    paths = result.get("pathologies", {})
    if paths:
        top10 = list(paths.items())[:10]
        for name, score in top10:
            st.progress(min(max(score, 0.0), 1.0), text=f"{name}: {score:.3f}")
    else:
        st.caption("No pathologies returned.")

    st.markdown('<div class="section-header">Clinical Entities</div>', unsafe_allow_html=True)
    entities = result.get("clinical_entities", {})
    if entities:
        st.write({
            "Age":             entities.get("age"),
            "Sex":             entities.get("sex"),
            "Chief complaint": entities.get("chief_complaint"),
            "Comorbidities":   entities.get("comorbidities", []),
        })
    else:
        st.caption("No clinical note parsed.")

    st.markdown('<div class="section-header">Retrieved Evidence</div>', unsafe_allow_html=True)
    st.caption(f"Query: `{result.get('retrieval_query', '')}`")
    # The full passages are inside citations; show a compact list
    if citations:
        for cit in citations:
            with st.expander(f"{cit['source']} · {cit['passage_id']}"):
                st.write(cit["passage"])

    st.markdown('<div class="section-header">Latency Breakdown</div>', unsafe_allow_html=True)
    lat = result.get("latency_breakdown", {})
    st.markdown(
        f"<span class='latency-badge'>Vision/Context (parallel): {lat.get('vision', 0):.2f}s</span> "
        f"<span class='latency-badge'>Retrieval: {lat.get('retrieval', 0):.2f}s</span> "
        f"<span class='latency-badge'>Report: {lat.get('report', 0):.2f}s</span> "
        f"<span class='latency-badge'>Verification: {lat.get('verification', 0):.2f}s</span> "
        f"<span class='latency-badge'><b>Total: {lat.get('total', 0):.2f}s</b></span>",
        unsafe_allow_html=True,
    )
    st.caption(f"Wall-clock incl. network: {wall:.2f}s")
