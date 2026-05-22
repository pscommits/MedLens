"""
streamlit_app.py — MedLens frontend v3.0

Sidebar has two sections:
  1. New Analysis (Doctor) — upload X-ray, enter patient's public key → encrypted
  2. My Reports (Patient)  — enter own public + secret key → see ALL reports
"""

import os
import base64
import time
import datetime
import requests
import streamlit as st

DEFAULT_BACKEND = os.environ.get("MEDLENS_BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="MedLens — Radiology Co-Pilot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .triage-banner {
        padding: 1rem 1.5rem; border-radius: 10px; font-size: 1.3rem;
        font-weight: 700; color: white; text-align: center; margin-bottom: 1rem;
    }
    .triage-STAT    { background: linear-gradient(135deg,#c0392b,#e74c3c); }
    .triage-URGENT  { background: linear-gradient(135deg,#d35400,#f39c12); }
    .triage-ROUTINE { background: linear-gradient(135deg,#27ae60,#2ecc71); }
    .triage-justification { font-size:0.9rem; font-weight:400; margin-top:0.4rem; }
    .section-header {
        font-size:1.05rem; font-weight:700; color:#2c3e50;
        border-bottom:2px solid #3498db; padding-bottom:0.2rem;
        margin-top:0.6rem; margin-bottom:0.4rem;
    }
    .report-card {
        border:1px solid #e2e8f0; border-radius:10px;
        padding:0.8rem 1rem; margin-bottom:0.6rem;
        background:#fafafa;
    }
    .report-date { font-size:0.8rem; color:#64748b; }
    .stat-badge  { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .urgent-badge{ background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .routine-badge{background:#dcfce7; color:#166534; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .confirm-card {
        background:#f0fdf4; border:1px solid #86efac; border-radius:10px;
        padding:0.9rem 1.1rem; margin:0.5rem 0 1rem 0; color:#15803d;
    }
    .latency-badge {
        background:#2c3e50; color:#ecf0f1; padding:3px 9px; border-radius:12px;
        font-size:0.82rem; display:inline-block; margin-right:5px; margin-bottom:3px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helpers
# =============================================================================

def check_backend(url):
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def call_analyze(url, image_bytes, name, note, patient_pub):
    r = requests.post(
        f"{url}/api/v1/analyze",
        files={"image": (name, image_bytes, "image/png")},
        data={"clinical_note": note, "session_id": "demo",
              "patient_stellar_pubkey": patient_pub},
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


def call_patient_reports(url, patient_secret):
    r = requests.post(
        f"{url}/api/v1/patient/reports",
        data={"patient_stellar_secret": patient_secret},
        timeout=60,
    )
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"({r.status_code}) {detail}")
    return r.json()


def b64_to_bytes(uri):
    if "," in uri:
        uri = uri.split(",", 1)[1]
    return base64.b64decode(uri)


def fmt_ts(ts):
    if not ts:
        return "Unknown date"
    return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M")


def triage_badge(level):
    cls = {"STAT": "stat-badge", "URGENT": "urgent-badge"}.get(level, "routine-badge")
    return f'<span class="{cls}">{level}</span>'


# =============================================================================
# Session state
# =============================================================================
DEFAULTS = {
    "mode":             None,    # "analyze" | "patient_reports"
    "analyze_result":   None,    # single AnalysisResponse dict
    "analyze_image":    None,    # bytes
    "patient_reports":  [],      # list of AnalysisResponse dicts
    "patient_pub":      "",      # derived from patient secret key by backend
    "selected_report":  0,       # index into patient_reports list
    "wall_time":        0.0,
    "error_msg":        None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("🩺 MedLens")
    st.caption("Multimodal radiology co-pilot · v3.0")

    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND)
    if check_backend(backend_url):
        st.success("✓ Backend online")
    else:
        st.error("✗ Backend offline")

    st.divider()

    # ── DOCTOR: new analysis ──────────────────────────────────────
    st.subheader("🔬 New Analysis")
    st.caption("Doctor uploads X-ray and encrypts for patient.")

    uploaded = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png","jpg","jpeg"])

    clinical_note = st.text_area(
        "Clinical note (optional)",
        value="45M with fever, cough, shortness of breath. Hx COPD.",
        height=85,
    )

    patient_pub_input = st.text_input(
        "Patient's Public Key (G...)",
        placeholder="GABC...XYZ",
        help="Patient's Stellar public key — report is encrypted so only they can read it.",
    )

    analyze_btn = st.button("Analyze & Encrypt for Patient",
                            type="primary", use_container_width=True)

    st.divider()

    # ── PATIENT: load all reports ─────────────────────────────────
    st.subheader("📂 My Reports")
    st.caption("Enter your secret key — your public key is derived automatically.")

    pt_sec_in = st.text_input(
        "Your Secret Key (S...)",
        placeholder="SABC...XYZ",
        type="password",
        key="pt_sec",
        help="Used only in memory to decrypt — never stored. Your public key is derived from this.",
    )

    load_btn = st.button("Load My Reports", use_container_width=True)

    st.divider()

    if st.button("🗑️ Clear", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    with st.expander("ℹ️  How it works"):
        st.markdown("""
**Doctor → Patient encryption:**
- Doctor enters patient's Stellar public key (G...)
- Report is encrypted: `Box(patient_pub, doctor_priv)`
- Only the patient can decrypt their own reports

**Patient access (secret key only):**
- Patient enters their Stellar secret key (S...)
- Public key is automatically derived from the secret key
- System finds all vault entries matching that public key
- Decrypts each: `Box(doctor_pub, patient_priv)`
- Lists all your reports, newest first

No need to enter your public key separately — your secret key is your complete identity.
        """)


# =============================================================================
# Button handlers
# =============================================================================

if analyze_btn:
    if uploaded is None:
        st.session_state.error_msg = "Please upload an X-ray."
    elif not patient_pub_input.strip():
        st.session_state.error_msg = "Enter the patient's Stellar Public Key (G...)."
    elif not patient_pub_input.strip().startswith("G"):
        st.session_state.error_msg = "Patient public key must start with 'G'."
    elif len(patient_pub_input.strip()) != 56:
        st.session_state.error_msg = "Stellar public keys are exactly 56 characters."
    else:
        img = uploaded.read()
        with st.spinner("Running 5-agent pipeline + encrypting..."):
            t0 = time.time()
            try:
                result = call_analyze(backend_url, img, uploaded.name,
                                      clinical_note, patient_pub_input.strip())
                st.session_state.analyze_result = result
                st.session_state.analyze_image  = img
                st.session_state.mode           = "analyze"
                st.session_state.wall_time      = time.time() - t0
                st.session_state.error_msg      = None
            except Exception as e:
                st.session_state.error_msg = f"Analysis failed: {e}"
        st.rerun()

if load_btn:
    sec = pt_sec_in.strip()
    if not sec:
        st.session_state.error_msg = "Enter your Stellar Secret Key (S...)."
    elif not sec.startswith("S"):
        st.session_state.error_msg = "Secret key must start with 'S'."
    elif len(sec) != 56:
        st.session_state.error_msg = "Stellar secret keys are exactly 56 characters."
    else:
        with st.spinner("Deriving your identity and decrypting your reports..."):
            t0 = time.time()
            try:
                data = call_patient_reports(backend_url, sec)
                st.session_state.patient_reports = data["reports"]
                st.session_state.patient_pub     = data["patient_stellar_pubkey"]
                st.session_state.mode            = "patient_reports"
                st.session_state.selected_report = 0
                st.session_state.wall_time       = time.time() - t0
                st.session_state.error_msg       = None
            except Exception as e:
                st.session_state.error_msg = f"Could not load reports: {e}"
        st.rerun()


# =============================================================================
# Main pane
# =============================================================================

st.title("MedLens — The Verifiable Radiology Co-Pilot")
st.caption("Every finding cited · Every region highlighted · Every report end-to-end encrypted")

if st.session_state.error_msg:
    st.error(st.session_state.error_msg)

mode = st.session_state.mode

if mode is None:
    st.info("👈 Use the sidebar to run a new analysis or load patient reports.")
    st.stop()


# =============================================================================
# MODE A: Fresh analysis result (doctor's view)
# =============================================================================

def render_report(result, image_bytes=None, compact=False):
    """Render a single AnalysisResponse dict. Used for both modes."""

    # Triage banner
    triage_level = result.get("triage_level", "ROUTINE")
    triage_just  = result.get("triage_justification", "")
    emoji = {"STAT": "🚨", "URGENT": "⚠️", "ROUTINE": "✅"}.get(triage_level, "ℹ️")
    st.markdown(
        f'<div class="triage-banner triage-{triage_level}">'
        f'{emoji} TRIAGE: {triage_level}'
        f'<div class="triage-justification">{triage_just}</div>'
        f'</div>', unsafe_allow_html=True,
    )

    col_img, col_report, col_meta = st.columns([1.1, 1.6, 1.1], gap="medium")

    with col_img:
        st.markdown('<div class="section-header">Image + GradCAM</div>',
                    unsafe_allow_html=True)
        if image_bytes:
            st.image(image_bytes, caption="Original X-ray", use_container_width=True)
        else:
            st.caption("Original image not available.")
        heatmap = result.get("heatmap_base64", "")
        if heatmap:
            try:
                st.image(b64_to_bytes(heatmap), caption="GradCAM", use_container_width=True)
            except Exception:
                st.caption("Heatmap unavailable.")

    with col_report:
        st.markdown('<div class="section-header">Structured Report</div>',
                    unsafe_allow_html=True)
        rep = result.get("structured_report", {})
        st.markdown("**Impression**");      st.write(rep.get("impression",""))
        st.markdown("**Findings**");        st.write(rep.get("findings",""))
        st.markdown("**Recommendations**"); st.write(rep.get("recommendations",""))

        st.markdown('<div class="section-header">Claim Verification</div>',
                    unsafe_allow_html=True)
        veri = result.get("verification", [])
        if veri:
            n_sup = sum(1 for v in veri if v.get("supported"))
            st.markdown(f"**{n_sup}/{len(veri)}** claims grounded in evidence")
            for i, v in enumerate(veri, 1):
                label = "✓ Supported" if v.get("supported") else "✗ Unsupported"
                with st.expander(f"Claim {i} · {v.get('score',0):.2f} · {label}"):
                    st.write(v.get("sentence",""))
                    st.caption(v.get("evidence",""))
        else:
            st.caption("No verification data.")

        st.markdown('<div class="section-header">Citations</div>', unsafe_allow_html=True)
        for cit in result.get("citations", []):
            with st.expander(f"{cit['marker']} · {cit['source']}"):
                st.write(cit["passage"])

    with col_meta:
        st.markdown('<div class="section-header">Pathologies</div>', unsafe_allow_html=True)
        for name, score in list(result.get("pathologies",{}).items())[:10]:
            st.progress(min(max(score,0.0),1.0), text=f"{name}: {score:.3f}")

        st.markdown('<div class="section-header">Clinical Entities</div>',
                    unsafe_allow_html=True)
        ent = result.get("clinical_entities", {})
        if ent:
            st.write({"Age": ent.get("age"), "Sex": ent.get("sex"),
                      "Chief complaint": ent.get("chief_complaint"),
                      "Comorbidities": ent.get("comorbidities",[])})

        st.markdown('<div class="section-header">Latency</div>', unsafe_allow_html=True)
        lat = result.get("latency_breakdown", {})
        st.markdown(
            f"<span class='latency-badge'>Vision: {lat.get('vision',0):.2f}s</span>"
            f"<span class='latency-badge'>Report: {lat.get('report',0):.2f}s</span>"
            f"<span class='latency-badge'><b>Total: {lat.get('total',0):.2f}s</b></span>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-header">🔐 Vault & Stellar</div>',
                    unsafe_allow_html=True)
        if result.get("report_id"):
            st.markdown(f"**Report ID:** `{result['report_id'][:18]}...`")
        if result.get("encrypted_at"):
            st.caption(f"Encrypted: {fmt_ts(result['encrypted_at'])}")
        doc_pub = result.get("doctor_stellar_pubkey","")
        if doc_pub:
            st.markdown(f"**Doctor key:** `{doc_pub[:14]}...`")
        tx  = result.get("stellar_tx_hash")
        url = result.get("stellar_explorer")
        if tx and url:
            st.markdown(f"**Stellar:** [`{tx[:18]}...`]({url})")
            st.caption("Anchored on blockchain.")


# ── DOCTOR MODE ──────────────────────────────────────────────────────────────
if mode == "analyze":
    result = st.session_state.analyze_result
    img    = st.session_state.analyze_image

    # Confirmation card
    if result.get("report_id"):
        pat = result.get("patient_stellar_pubkey","")
        st.markdown(
            f'<div class="confirm-card">'
            f'✅ <b>Report encrypted and stored in vault</b><br>'
            f'Patient key: <code>{pat[:20]}...{pat[-6:]}</code>&nbsp;&nbsp;'
            f'Report ID: <code>{result["report_id"][:18]}...</code><br>'
            f'<small>The patient can access this report by entering their Stellar <b>secret key</b> '
            f'in <b>My Reports</b> in the sidebar.</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Report was NOT encrypted (no patient key was provided).")

    render_report(result, img)


# ── PATIENT MODE ─────────────────────────────────────────────────────────────
elif mode == "patient_reports":
    reports = st.session_state.patient_reports
    pub     = st.session_state.patient_pub

    st.markdown(
        f"### 📂 Your Reports &nbsp;&nbsp;"
        f'<span style="font-size:0.9rem;color:#64748b;">'
        f'<code>{pub[:16]}...{pub[-6:]}</code></span>',
        unsafe_allow_html=True,
    )

    if not reports:
        st.info("No reports found for this patient key. "
                "Ask your doctor to run an analysis with your public key.")
        st.stop()

    st.caption(f"{len(reports)} report(s) found · newest first · loaded in {st.session_state.wall_time:.1f}s")
    st.divider()

    # Report selector — summary cards + full detail on selection
    report_labels = []
    for i, r in enumerate(reports):
        top_path = next(iter(r.get("pathologies", {})), "Unknown")
        score    = r.get("pathologies", {}).get(top_path, 0.0)
        level    = r.get("triage_level", "ROUTINE")
        date_str = fmt_ts(r.get("encrypted_at"))
        report_labels.append(f"#{i+1}  {date_str}  ·  {level}  ·  {top_path} ({score:.2f})")

    selected = st.selectbox(
        "Select a report to view:",
        options=range(len(reports)),
        format_func=lambda i: report_labels[i],
        index=st.session_state.selected_report,
    )
    st.session_state.selected_report = selected

    # Summary row of all reports
    st.markdown("#### All reports at a glance")
    cols = st.columns(min(len(reports), 4))
    for i, r in enumerate(reports[:4]):
        level = r.get("triage_level","ROUTINE")
        date_str = fmt_ts(r.get("encrypted_at"))
        top_path = next(iter(r.get("pathologies",{})),"—")
        badge_html = triage_badge(level)
        with cols[i]:
            st.markdown(
                f'<div class="report-card">'
                f'<div class="report-date">{date_str}</div>'
                f'{badge_html}<br>'
                f'<small>{top_path}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
    if len(reports) > 4:
        st.caption(f"... and {len(reports)-4} more. Use the selector above.")

    st.divider()
    st.markdown(f"#### Report {selected+1} — {report_labels[selected]}")

    render_report(reports[selected])