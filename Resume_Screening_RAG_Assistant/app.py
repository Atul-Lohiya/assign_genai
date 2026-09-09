"""
app.py
------
Streamlit UI for the AI Resume Screening Assistant.

Run with:
    streamlit run app.py

Features:
    - Upload one or more resume PDFs
    - Enter / paste a Job Description
    - Evaluate every resume against the JD via a per-resume RAG pipeline
    - View Match Score, Matching/Missing Skills, Summary, Strengths,
      Weaknesses, and Hiring Recommendation for each candidate
    - Compare any two evaluated candidates side by side
    - Get an auto-ranked "best candidate" recommendation
    - Ask free-form questions about a specific resume (RAG Q&A)
"""

import os
import html as html_lib
import base64
from pathlib import Path

import streamlit as st
import pandas as pd

from rag_engine import build_engine, rank_candidates, PROVIDERS

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# # --------------------------------------------------------------------------- #
# # Background image (gradient artwork) — swap assets/background.png for any
# # other image to change the look; falls back to a CSS-only gradient if the
# # file isn't found so the app never breaks.
# # --------------------------------------------------------------------------- #
# APP_DIR = Path(__file__).parent
# BG_IMAGE_PATH = APP_DIR / "assets" / "background.png"


@st.cache_data
def _load_base64_image(path_str: str, mtime: float) -> str | None:
    """`mtime` is included purely so the cache key changes whenever the file
    on disk is replaced (even if the filename stays the same) — otherwise
    Streamlit would keep serving stale, previously-cached image bytes.
    (Must NOT be prefixed with an underscore: Streamlit's cache_data excludes
    underscore-prefixed params from the cache key, which would defeat the
    whole point of passing it in.)"""
    p = Path(path_str)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()

# --------------------------------------------------------------------------- #
# Design system: colors, icons, and small HTML component builders
# --------------------------------------------------------------------------- #
REC_STYLE = {
    "Strong Hire": {"color": "#16A34A", "bg": "#DCFCE7", "icon": "🌟"},
    "Hire":        {"color": "#2563EB", "bg": "#DBEAFE", "icon": "👍"},
    "Maybe":       {"color": "#D97706", "bg": "#FEF3C7", "icon": "🤔"},
    "No Hire":     {"color": "#DC2626", "bg": "#FEE2E2", "icon": "🚫"},
}
MEDALS = ["🥇", "🥈", "🥉"]


def score_color(score: int) -> str:
    if score >= 80:
        return "#16A34A"   # green
    if score >= 65:
        return "#2563EB"   # blue
    if score >= 45:
        return "#D97706"   # amber
    return "#DC2626"       # red


def esc(text: str) -> str:
    return html_lib.escape(str(text))


def score_ring_html(score: int, size: int = 108) -> str:
    color = score_color(score)
    inner = int(size * 0.78)
    return f"""
    <div style="width:{size}px;height:{size}px;border-radius:50%;
                background:conic-gradient({color} {score * 3.6}deg, #EEF0F6 0deg);
                display:flex;align-items:center;justify-content:center;
                box-shadow:0 2px 10px rgba(30,20,60,0.08);">
      <div style="width:{inner}px;height:{inner}px;border-radius:50%;background:#FFFFFF;
                  display:flex;flex-direction:column;align-items:center;justify-content:center;">
        <span style="font-size:26px;font-weight:800;color:{color};line-height:1;">{score}</span>
        <span style="font-size:10px;font-weight:600;color:#9AA1B4;letter-spacing:0.05em;">/ 100</span>
      </div>
    </div>
    """


def rec_badge_html(recommendation: str) -> str:
    s = REC_STYLE.get(recommendation, {"color": "#6B7280", "bg": "#F3F4F6", "icon": "•"})
    return f"""
    <span style="display:inline-block;padding:6px 14px;border-radius:999px;
                 background:{s['bg']};color:{s['color']};font-weight:700;
                 font-size:13px;letter-spacing:0.01em;">
      {s['icon']}&nbsp; {esc(recommendation)}
    </span>
    """


def chip_row_html(items, kind: str) -> str:
    """kind: 'match' (green) or 'miss' (red/amber)"""
    if not items:
        return "<span style='color:#9AA1B4;font-size:13px;'>None identified</span>"
    palette = (
        {"bg": "#E9FBF0", "fg": "#158A45", "border": "#BCEFD1"}
        if kind == "match"
        else {"bg": "#FDF1F1", "fg": "#C0392B", "border": "#F6D3D0"}
    )
    chips = "".join(
        f"""<span style="display:inline-block;margin:3px 6px 3px 0;padding:5px 12px;
                    border-radius:999px;background:{palette['bg']};color:{palette['fg']};
                    border:1px solid {palette['border']};font-size:12.5px;font-weight:600;">
              {esc(item)}
            </span>"""
        for item in items
    )
    return f"<div>{chips}</div>"


def list_html(items, icon: str) -> str:
    if not items:
        return "<span style='color:#9AA1B4;font-size:13px;'>—</span>"
    rows = "".join(
        f"""<div style="margin:4px 0;font-size:14px;line-height:1.45;">
              <span style="margin-right:6px;">{icon}</span>{esc(item)}
            </div>"""
        for item in items
    )
    return rows


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {{ font-family: 'Inter', sans-serif; }}

    /* Full-app gradient background */
    
    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Hero banner — frosted glass over the gradient */
    .hero {{
        background: rgba(255,255,255,0.62);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.55);
        padding: 34px 38px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(108,92,231,0.15);
    }}
    .hero h1 {{
        color: #4B3AA4;
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 6px 0;
    }}
    .hero p {{
        color: #5C5470;
        font-size: 15px;
        margin: 0;
    }}

    /* Candidate card — frosted glass */
    .candidate-card {{
        border: 1px solid rgba(255,255,255,0.6);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
        background: rgba(255,255,255,0.72);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 2px 14px rgba(30,20,60,0.08);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }}
    .candidate-card:hover {{
        box-shadow: 0 8px 26px rgba(30,20,60,0.14);
        transform: translateY(-1px);
    }}
    .candidate-name {{
        font-size: 19px;
        font-weight: 800;
        color: #1F2430;
        margin-bottom: 2px;
    }}
    .section-label {{
        font-size: 12.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #7A7F94;
        margin: 14px 0 6px 0;
    }}
    .divider-soft {{
        border: none;
        border-top: 1px solid rgba(139,143,163,0.25);
        margin: 14px 0;
    }}
    .winner-banner {{
        background: rgba(255,247,230,0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #FBE5B8;
        border-radius: 14px;
        padding: 16px 20px;
        font-size: 15px;
    }}

    /* Sidebar — frosted glass to match */
    section[data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.72);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(139,143,163,0.25);
    }}

    /* Buttons */
    div.stButton > button, div.stButton > button:focus {{
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6em 1.2em;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        font-weight: 600;
        font-size: 15px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "engines" not in st.session_state:
    st.session_state.engines = {}  # name -> ResumeRAGEngine
if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}  # name -> CandidateEvaluation
if "jd_used" not in st.session_state:
    st.session_state.jd_used = ""

# --------------------------------------------------------------------------- #
# Sidebar: configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("###Configuration")

    provider = st.selectbox(
        "LLM Provider",
        list(PROVIDERS.keys()),
        index=0,  # DeepSeek first (cheap/free-tier friendly)
        help="Pick which chat model powers the evaluation. Embeddings always "
        "run locally for free, regardless of this choice.",
    )
    provider_cfg = PROVIDERS[provider]

    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        value=os.environ.get(f"{provider.upper()}_API_KEY", ""),
        help="Your key is used only for this session and is never stored.",
    )
    llm_model = st.selectbox("LLM model", provider_cfg["models"], index=0)

    st.markdown(
        """
        <div style="font-size:12.5px;color:#8B8FA3;line-height:1.6;margin-top:6px;">
        - <b>Embeddings:</b> <code>BAAI/bge-small-en-v1.5</code> via FastEmbed
        - runs locally &amp; free, no API key needed.<br>
        - <b>Vector store:</b> FAISS (in-memory, per resume)
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='divider-soft'>", unsafe_allow_html=True)
    if st.button("Clear all data / start over", use_container_width=True):
        st.session_state.engines = {}
        st.session_state.evaluations = {}
        st.session_state.jd_used = ""
        st.rerun()

# --------------------------------------------------------------------------- #
# Hero header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h1>AI Resume Screening Assistant</h1>
        <p>LangChain + RAG (FAISS) resume screener — upload resumes, paste a JD,
        get grounded, structured candidate evaluations in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Inputs: JD + resumes
# --------------------------------------------------------------------------- #
col_jd, col_files = st.columns([1.2, 1])

with col_jd:
    st.markdown("#### 📋 Job Description")
    jd_text = st.text_area(
        "Job Description",
        height=240,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

with col_files:
    st.markdown("#### 📎 Resumes")
    uploaded_files = st.file_uploader(
        "Upload resume PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.markdown(
            f"<div style='font-size:13.5px;color:#4B5066;margin-top:4px;'>"
            f"📄 <b>{len(uploaded_files)}</b> resume(s) selected</div>",
            unsafe_allow_html=True,
        )
        for f in uploaded_files:
            st.markdown(
                f"<div style='font-size:13px;color:#8B8FA3;'>• {esc(f.name)}</div>",
                unsafe_allow_html=True,
            )

st.write("")
evaluate_clicked = st.button(
    "Evaluate resumes against JD", type="primary", use_container_width=True
)

# --------------------------------------------------------------------------- #
# Evaluation run
# --------------------------------------------------------------------------- #
if evaluate_clicked:
    if not api_key:
        st.error(f"Please enter your {provider} API key in the sidebar.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume PDF.")
    else:
        st.session_state.jd_used = jd_text
        progress = st.progress(0.0, text="Starting evaluation...")
        n = len(uploaded_files)
        for i, uf in enumerate(uploaded_files):
            name = os.path.splitext(uf.name)[0]
            progress.progress(
                i / n, text=f"🔎 Indexing & evaluating {uf.name} ({i+1}/{n})..."
            )
            try:
                engine = build_engine(
                    uf, api_key, llm_model, base_url=provider_cfg["base_url"]
                )
                evaluation = engine.evaluate(jd_text)
                st.session_state.engines[name] = engine
                st.session_state.evaluations[name] = evaluation
            except Exception as e:
                st.error(f"Failed to process {uf.name}: {e}")
        progress.progress(1.0, text="Done.")
        st.success(f"✅ Evaluated {len(st.session_state.evaluations)} resume(s).")

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
evals = st.session_state.evaluations

if evals:
    tab_individual, tab_compare, tab_rank = st.tabs(
        ["📄 Individual Evaluations", "⚖️ Compare Two", "🏆 Rank & Recommend"]
    )

    # ----------------------------- Individual ----------------------------- #
    with tab_individual:
        for name, ev in evals.items():
            with st.container():
                st.markdown('<div class="candidate-card">', unsafe_allow_html=True)

                head_l, head_r = st.columns([3, 1])
                with head_l:
                    st.markdown(
                        f'<div class="candidate-name">👤 {esc(ev.candidate_name)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(rec_badge_html(ev.recommendation), unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='margin-top:10px;font-size:14px;color:#4B5066;"
                        f"line-height:1.5;'>📝 {esc(ev.summary)}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='margin-top:10px;font-size:13px;color:#8B8FA3;"
                        f"font-style:italic;'>💡 {esc(ev.justification)}</div>",
                        unsafe_allow_html=True,
                    )
                with head_r:
                    st.markdown(
                        f"<div style='display:flex;justify-content:center;'>"
                        f"{score_ring_html(ev.match_score)}</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<hr class='divider-soft'>", unsafe_allow_html=True)

                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown('<div class="section-label">✅ Matching Skills</div>', unsafe_allow_html=True)
                    st.markdown(chip_row_html(ev.matching_skills, "match"), unsafe_allow_html=True)
                    st.markdown('<div class="section-label">💪 Strengths</div>', unsafe_allow_html=True)
                    st.markdown(list_html(ev.strengths, "✔️"), unsafe_allow_html=True)
                with sc2:
                    st.markdown('<div class="section-label">🚫 Missing Skills</div>', unsafe_allow_html=True)
                    st.markdown(chip_row_html(ev.missing_skills, "miss"), unsafe_allow_html=True)
                    st.markdown('<div class="section-label">⚠️ Weaknesses</div>', unsafe_allow_html=True)
                    st.markdown(list_html(ev.weaknesses, "•"), unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------- Compare -------------------------------- #
    with tab_compare:
        names = list(evals.keys())
        if len(names) < 2:
            st.info("Upload and evaluate at least two resumes to compare.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                name_a = st.selectbox("👤 Candidate A", names, index=0, key="cmp_a")
            with c2:
                name_b = st.selectbox(
                    "👤 Candidate B", names, index=min(1, len(names) - 1), key="cmp_b"
                )

            if name_a and name_b:
                ea, eb = evals[name_a], evals[name_b]
                winner = ea if ea.match_score >= eb.match_score else eb

                cc1, cc2 = st.columns(2)
                for col, ev in ((cc1, ea), (cc2, eb)):
                    crown = "👑 " if ev.candidate_name == winner.candidate_name else ""
                    with col:
                        st.markdown('<div class="candidate-card">', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="candidate-name">{crown}{esc(ev.candidate_name)}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='display:flex;justify-content:center;margin:10px 0;'>"
                            f"{score_ring_html(ev.match_score, size=92)}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='display:flex;justify-content:center;'>"
                            f"{rec_badge_html(ev.recommendation)}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("<hr class='divider-soft'>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">✅ Matching Skills</div>', unsafe_allow_html=True)
                        st.markdown(chip_row_html(ev.matching_skills, "match"), unsafe_allow_html=True)
                        st.markdown('<div class="section-label">🚫 Missing Skills</div>', unsafe_allow_html=True)
                        st.markdown(chip_row_html(ev.missing_skills, "miss"), unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(
                    f"""<div class="winner-banner">
                        🏆 For this JD, <b>{esc(winner.candidate_name)}</b> has the
                        stronger match (<b>{winner.match_score}/100</b>, {esc(winner.recommendation)}).
                        </div>""",
                    unsafe_allow_html=True,
                )

    # ------------------------------- Rank ---------------------------------- #
    with tab_rank:
        ranked = rank_candidates(list(evals.values()))

        if ranked:
            best = ranked[0]
            st.markdown(
                f"""<div class="winner-banner" style="margin-bottom:18px;">
                    🏆 <b>Best candidate: {esc(best.candidate_name)}</b>
                    &nbsp;·&nbsp; Score {best.match_score}/100 &nbsp;·&nbsp; {esc(best.recommendation)}
                    <div style="margin-top:6px;font-size:13.5px;color:#4B5066;">{esc(best.justification)}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        for i, ev in enumerate(ranked):
            medal = MEDALS[i] if i < 3 else f"#{i+1}"
            bar_color = score_color(ev.match_score)
            with st.container():
                st.markdown('<div class="candidate-card">', unsafe_allow_html=True)
                r1, r2, r3 = st.columns([0.4, 2.6, 1])
                with r1:
                    st.markdown(
                        f"<div style='font-size:26px;text-align:center;'>{medal}</div>",
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        f"<div class='candidate-name' style='margin-bottom:6px;'>{esc(ev.candidate_name)}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""<div style="background:#F0EEFA;border-radius:8px;height:10px;width:100%;">
                            <div style="background:{bar_color};width:{ev.match_score}%;height:10px;
                                        border-radius:8px;"></div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='margin-top:8px;'>{rec_badge_html(ev.recommendation)}</div>",
                        unsafe_allow_html=True,
                    )
                with r3:
                    st.markdown(
                        f"<div style='text-align:center;font-size:24px;font-weight:800;color:{bar_color};'>"
                        f"{ev.match_score}<span style='font-size:13px;color:#9AA1B4;'>/100</span></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="candidate-card" style="text-align:center;padding:40px;">
            <div style="font-size:40px;">🚀</div>
            <div style="font-size:16px;font-weight:700;margin-top:8px;color:#1F2430;">
                Ready when you are
            </div>
            <div style="font-size:14px;color:#8B8FA3;margin-top:4px;">
                Paste a Job Description, upload one or more resume PDFs, then click
                <b>Evaluate resumes against JD</b> to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
