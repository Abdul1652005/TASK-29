import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_URL = os.getenv("API_URL", "https://task-29-production.up.railway.app")
GITHUB_URL = "https://github.com/Abdul1652005"
AUTHOR = "Abdul Rehman"

st.set_page_config(
    page_title="Ticket Classifier — PKCERT",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 60%, #0e1520 100%); }
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03);
    border-right: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(10px);
}
.hero-header {
    background: linear-gradient(135deg, rgba(52,211,153,0.12) 0%, rgba(59,130,246,0.08) 100%);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "";
    position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(52,211,153,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.1rem; font-weight: 800;
    background: linear-gradient(90deg, #34d399, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: #9ca3af; font-size: 0.95rem; margin-top: 0.3rem; }
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(6px);
}
.result-card {
    background: linear-gradient(135deg, rgba(52,211,153,0.1) 0%, rgba(16,185,129,0.06) 100%);
    border: 1px solid rgba(52,211,153,0.3);
    border-left: 4px solid #34d399;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin: 1rem 0;
    animation: slideIn 0.4s ease-out;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(-10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label { color: #9ca3af; font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.3rem; }
.result-value { color: #34d399; font-size: 1.9rem; font-weight: 700; }
.result-confidence { color: #d1d5db; font-size: 0.85rem; margin-top: 0.5rem; }
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-number { font-size: 1.8rem; font-weight: 700; color: #34d399; }
.metric-label  { font-size: 0.78rem; color: #9ca3af; letter-spacing: 1px; text-transform: uppercase; }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0 1rem; }
.chip {
    background: rgba(31,41,55,0.8);
    border: 1px solid rgba(255,255,255,0.1);
    color: #d1d5db;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s;
}
.chip:hover { background: rgba(52,211,153,0.2); border-color: #34d399; color: #34d399; }
.cat-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
}
.section-header {
    color: #9ca3af; font-size: 0.72rem;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 0.8rem; font-weight: 600;
}
.sb-badge {
    display: inline-block;
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.3);
    color: #34d399;
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    margin: 0.15rem 0.1rem;
}
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.82rem;
    margin-top: 2rem;
    padding: 1rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.footer a { color: #34d399; text-decoration: none; font-weight: 600; }
.footer a:hover { text-decoration: underline; }
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
}
.empty-icon  { font-size: 3rem; }
.empty-label { color: #9ca3af; margin-top: 0.8rem; font-size: 0.9rem; }
table.info-table { width: 100%; border-collapse: collapse; color: #d1d5db; font-size: 0.88rem; }
table.info-table td { padding: 0.4rem 0; }
table.info-table td:first-child { color: #9ca3af; width: 40%; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


CATEGORY_COLORS = [
    ("#34d399", "#132a23"), ("#60a5fa", "#0f1f35"), ("#f472b6", "#2a0f1e"),
    ("#a78bfa", "#1a0f35"), ("#fb923c", "#2a1505"), ("#facc15", "#2a2005"),
    ("#38bdf8", "#0a1f2a"), ("#f87171", "#2a0f0f"), ("#4ade80", "#0f2a12"),
    ("#e879f9", "#250f2a"),
]

EXAMPLES = [
    "My laptop screen is black and won't turn on",
    "I was charged twice for my subscription",
    "I can't reset my password",
    "The app crashes every time I open it",
    "My order hasn't arrived yet",
    "I need to cancel my account",
]


@st.cache_resource
def load_model():
    m_path = os.path.join(BASE_DIR, "ticket_model.joblib")
    if not os.path.exists(m_path):
        m_path = os.path.join(BASE_DIR, "ticket-classifier", "frontend", "ticket_model.joblib")
    v_path = os.path.join(BASE_DIR, "ticket_vectorizer.joblib")
    if not os.path.exists(v_path):
        v_path = os.path.join(BASE_DIR, "ticket-classifier", "frontend", "ticket_vectorizer.joblib")
    m = joblib.load(m_path)
    v = joblib.load(v_path)
    return m, v


model, vectorizer = load_model()
categories = list(model.classes_)
cat_color_map = {cat: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, cat in enumerate(categories)}


def predict_local(text_input: str):
    vec = vectorizer.transform([text_input.strip().lower()])
    prediction = model.predict(vec)[0]
    scores = model.decision_function(vec)[0]
    exp_s = np.exp(scores - np.max(scores))
    probs = exp_s / exp_s.sum()
    confidence_pct = float(probs.max()) * 100
    distribution = {cat: round(float(p) * 100, 2) for cat, p in zip(categories, probs)}
    return prediction, confidence_pct, distribution


if "history" not in st.session_state:
    st.session_state.history = []
if "total_classified" not in st.session_state:
    st.session_state.total_classified = 0
if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""
if "fill_example" not in st.session_state:
    st.session_state.fill_example = None

if st.session_state.fill_example:
    st.session_state.ticket_text = st.session_state.fill_example
    st.session_state.fill_example = None


with st.sidebar:
    st.markdown("### 🤖 Model Info")
    st.markdown("""
    <div class="glass-card">
        <div class="section-header">Architecture</div>
        <p style="color:#d1d5db;font-size:0.9rem;margin:0">TF-IDF Vectorizer<br>+ Linear SVM (LinearSVC)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-card">
        <div class="section-header">Categories ({len(categories)})</div>
        {''.join(f'<span class="sb-badge">{c}</span>' for c in categories)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-header">Pipeline</div>
        <p style="color:#9ca3af;font-size:0.82rem;line-height:1.8;margin:0">
            📓 Jupyter Notebook<br>
            ↓ Data cleaning + TF-IDF<br>
            ↓ LinearSVC training<br>
            ↓ <span style="color:#60a5fa">FastAPI backend</span><br>
            ↓ <span style="color:#34d399">Streamlit frontend</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{st.session_state.total_classified}</div>
            <div class="metric-label">Classified</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(categories)}</div>
            <div class="metric-label">Categories</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        df_export = pd.DataFrame(st.session_state.history)
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export History (CSV)",
            data=csv,
            file_name="ticket_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_classified = 0
            st.rerun()

    st.markdown("---")
    st.markdown(
        f'<p style="color:#6b7280;font-size:0.75rem;text-align:center">'
        f'Task 29 · PKCERT Internship &nbsp;·&nbsp; '
        f'<a href="{GITHUB_URL}" target="_blank" style="color:#34d399;text-decoration:none">github.com/Abdul1652005</a>'
        f'</p>',
        unsafe_allow_html=True,
    )


st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎫 Support Ticket Classifier</div>
    <div class="hero-sub">AI-powered ticket triage &nbsp;·&nbsp; TF-IDF + LinearSVC &nbsp;·&nbsp; Task 29 Capstone</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✨ Classify", "📋 History", "ℹ️ About"])


with tab1:
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<div class="section-header">Ticket Description</div>', unsafe_allow_html=True)

        text = st.text_area(
            label="ticket_input",
            label_visibility="collapsed",
            height=150,
            placeholder="Describe the support issue in detail…",
            key="ticket_text",
        )

        char_count = len(text)
        st.caption(f"{char_count} characters")

        st.markdown('<div class="section-header" style="margin-top:0.8rem">Quick Examples</div>', unsafe_allow_html=True)
        ex_cols = st.columns(3)
        for i, ex in enumerate(EXAMPLES):
            short = ex[:28] + "…" if len(ex) > 28 else ex
            if ex_cols[i % 3].button(short, key=f"ex_{i}", use_container_width=True):
                st.session_state.fill_example = ex
                st.rerun()

        classify_clicked = st.button("✨ Classify Ticket", type="primary", use_container_width=True)

    with right_col:
        if classify_clicked:
            if not text.strip():
                st.warning("⚠️ Please enter a ticket description.")
            else:
                with st.spinner("Analysing…"):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"text": text.strip()},
                            timeout=4,
                        )
                        response.raise_for_status()
                        result = response.json()
                        prediction = result["category"]
                        confidence_pct = result["confidence"]
                        distribution = result["distribution"]
                    except Exception:
                        prediction, confidence_pct, distribution = predict_local(text)

                st.session_state.total_classified += 1
                st.session_state.history.insert(0, {
                    "Ticket": text.strip()[:60] + ("…" if len(text.strip()) > 60 else ""),
                    "Category": prediction,
                    "Confidence": f"{confidence_pct:.1f}%",
                    "Time": datetime.now().strftime("%H:%M:%S"),
                })

                pred_color, pred_bg = cat_color_map.get(prediction, ("#34d399", "#132a23"))

                st.markdown(f"""
                <div class="result-card" style="
                    background: linear-gradient(135deg, {pred_bg}CC 0%, rgba(13, 17, 23, 0.9) 100%);
                    border: 1px solid {pred_color}55;
                    border-left: 4px solid {pred_color};">
                    <div class="result-label">Predicted Category</div>
                    <div class="result-value" style="color:{pred_color}">{prediction}</div>
                    <div class="result-confidence">Relative confidence: <b>{confidence_pct:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(confidence_pct / 100)

                st.markdown('<div class="section-header" style="margin-top:1.2rem">Score Distribution</div>', unsafe_allow_html=True)
                sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]
                chart_df = pd.DataFrame(sorted_items, columns=["Category", "Confidence (%)"])
                st.bar_chart(chart_df.set_index("Category"), color="#34d399", use_container_width=True)

        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🎫</div>
                <div class="empty-label">
                    Enter a ticket and click <b style="color:#34d399">Classify Ticket</b><br>to see predictions here.
                </div>
            </div>
            """, unsafe_allow_html=True)


with tab2:
    if not st.session_state.history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📋</div>
            <div class="empty-label">No classifications yet this session.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="section-header">{len(st.session_state.history)} classification(s) this session</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(
            df_hist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticket":     st.column_config.TextColumn("Ticket Snippet", width="large"),
                "Category":   st.column_config.TextColumn("Category",       width="medium"),
                "Confidence": st.column_config.TextColumn("Confidence",     width="small"),
                "Time":       st.column_config.TextColumn("Time",           width="small"),
            },
        )

        st.markdown('<div class="section-header" style="margin-top:1.5rem">Category Distribution This Session</div>', unsafe_allow_html=True)
        cat_counts = df_hist["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        st.bar_chart(cat_counts.set_index("Category"), color="#60a5fa", use_container_width=True)


with tab3:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">Project</div>
            <p style="color:#d1d5db;line-height:1.8;margin:0">
                Customer Support Ticket Classifier built as the <b style="color:#34d399">Task 29 Final Capstone</b>
                of the <b>PKCERT AI & Software Development Internship</b>.<br><br>
                Automatically categorizes free-text support tickets using a trained machine learning model,
                reducing manual triage time and improving response routing.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="section-header">Tech Stack</div>
            <table class="info-table">
                <tr><td>Model</td><td style="color:#d1d5db">TF-IDF + LinearSVC</td></tr>
                <tr><td>Backend</td><td style="color:#d1d5db">FastAPI + Uvicorn</td></tr>
                <tr><td>Frontend</td><td style="color:#d1d5db">Streamlit</td></tr>
                <tr><td>Training</td><td style="color:#d1d5db">Jupyter Notebook</td></tr>
                <tr><td>Serialization</td><td style="color:#d1d5db">joblib</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="section-header">Supported Categories</div>
            {''.join(
                f'<span class="cat-badge" style="background:{cat_color_map[c][1]};color:{cat_color_map[c][0]};border:1px solid {cat_color_map[c][0]}50">{c}</span>'
                for c in categories
            )}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="section-header">How It Works</div>
            <p style="color:#9ca3af;font-size:0.85rem;line-height:2;margin:0">
                1️⃣ &nbsp;Text is lowercased & cleaned<br>
                2️⃣ &nbsp;TF-IDF converts text → feature vector<br>
                3️⃣ &nbsp;LinearSVC produces decision scores<br>
                4️⃣ &nbsp;Softmax applied for relative confidence<br>
                5️⃣ &nbsp;Top category returned as prediction
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card">
            <div class="section-header">Author</div>
            <p style="color:#d1d5db;font-size:0.9rem;margin:0">
                <b>{AUTHOR}</b><br>
                <a href="{GITHUB_URL}" target="_blank"
                   style="color:#34d399;text-decoration:none;font-size:0.85rem">
                    🐙 github.com/Abdul1652005
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)


st.markdown(f"""
<div class="footer">
    Built by <b style="color:#34d399">{AUTHOR}</b> &nbsp;·&nbsp;
    PKCERT AI & Software Development Internship &nbsp;·&nbsp; Task 29 &nbsp;·&nbsp;
    <a href="{GITHUB_URL}" target="_blank">github.com/Abdul1652005</a>
</div>
""", unsafe_allow_html=True)
