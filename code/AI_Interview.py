import os
import time
import streamlit as st
from google import genai
from PyPDF2 import PdfReader
from docx import Document
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
#  PROJECT B2 – AI Interview Assistant & Candidate Evaluation System
#  Students  : Abdullah Bintaleb | 4410891  &  Ziyad Chihani | 4312645
#  Supervisor : Dr. Syed Bukhari
#  Tech Stack: Streamlit · Google Gemini · spaCy · Sentence-BERT
# ============================================================

# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Interview Assistant & Candidate Evaluation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS Styling ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1a1f3a, #1e1b4b, #0f172a) !important;
        background-size: 400% 400% !important;
        animation: gradientBG 12s ease infinite !important;
    }
    @keyframes gradientBG {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass card */
    .glass {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
    }

    /* Gradient title */
    .main-title {
        font-size: 2.6em;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.15em;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1em !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.4) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(168,85,247,0.5) !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #34d399 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
    }

    /* Text area */
    .stTextArea textarea {
        background: rgba(0,0,0,0.25) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
    }

    /* Info / Success / Warning / Error boxes text */
    [data-testid="stNotification"] p { color: #0f172a !important; }

    /* Hide Streamlit menu & footer */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Gemini API Client ----------
client = genai.Client(api_key="Your api key")

# ---------- Load NLP Models (cached – loads only once) ----------
@st.cache_resource(show_spinner="⏳ Loading AI models for the first time...")
def load_models():
    """
    Load two NLP models:
    1. spaCy  – for Named Entity Recognition (skill extraction)
    2. Sentence-BERT – for semantic similarity scoring
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None   # spaCy model not downloaded yet
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, sbert

nlp_model, sbert_model = load_models()

# ==============================================================
# CORE PROJECT FUNCTIONS
# ==============================================================

def read_cv(file_path: str) -> str:
    """
    Step 1 – CV Parsing
    Extracts raw text from an uploaded PDF or DOCX file.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def extract_skills(text: str) -> set:
    """
    Step 2 – Skill Extraction (NER with spaCy)
    Matches text against a predefined list of technical skills.
    Returns a set of matched skill names.
    """
    SKILL_LIST = [
        "python", "java", "c++", "machine learning", "deep learning", "nlp",
        "data science", "fastapi", "streamlit", "sql", "react", "aws", "docker",
        "spacy", "scikit-learn", "numpy", "pandas", "pytorch", "tensorflow",
        "javascript", "html", "css", "git", "linux", "agile", "scrum",
        "data analysis", "power bi", "tableau", "excel", "r", "mongodb", "flask"
    ]
    text_lower = text.lower()
    return {skill.title() for skill in SKILL_LIST if skill in text_lower}


def generate_questions(cv_text: str, job_desc: str, niche: str) -> list:
    """
    Step 3 – Interview Question Generation (using Google Gemini LLM)
    Generates 3 questions with increasing difficulty levels:
      Q1 – Very Easy   (fundamental concept)
      Q2 – Tricky      (requires careful thinking)
      Q3 – Hard        (tests deep expertise)
    """
    prompt = f"""You are a technical recruiter specializing in {niche}.
Based on the candidate's CV and Job Description, generate exactly 3 interview questions:

- Question 1 (Very Easy): A simple, beginner-level question about a fundamental concept.
- Question 2 (Easy but Tricky): Looks easy but requires careful thinking or exposes common mistakes.
- Question 3 (Hard): Tests deep knowledge, real experience, and problem-solving ability.

Job Description:
{job_desc}

Candidate CV:
{cv_text}

Return ONLY the 3 questions, one per line. No labels, no numbering."""
    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return [q.strip() for q in response.text.split("\n") if q.strip()][:3]


def get_ideal_answer(question: str) -> str:
    """
    Step 4a – Ideal Answer Generation (Google Gemini)
    Generates a model answer to use as the comparison baseline.
    """
    prompt = f"Give a concise, expert-level answer to this interview question:\n{question}"
    return client.models.generate_content(model="gemini-flash-latest", contents=prompt).text


def semantic_similarity(answer1: str, answer2: str) -> float:
    """
    Step 4b – Semantic Similarity (Sentence-BERT + Cosine Similarity)
    Encodes both answers into vectors and computes cosine similarity.
    Returns a float between 0 and 1 (1 = identical meaning).
    """
    vectors = sbert_model.encode([answer1, answer2])
    score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return float(score)


def get_feedback(question: str, user_ans: str, ideal_ans: str, sim_score: float) -> str:
    """
    Step 4c – LLM Feedback (Google Gemini)
    Combines the similarity score with qualitative feedback and a grade (A–F).
    """
    prompt = f"""You are an interview evaluator.

Question      : {question}
Candidate Said: {user_ans}
Ideal Answer  : {ideal_ans}
Semantic Match: {sim_score * 100:.1f}%

Provide:
1. Short qualitative feedback (strengths and what was missing).
2. A final Grade: A, B, C, D, or F."""
    return client.models.generate_content(model="gemini-flash-latest", contents=prompt).text


# ==============================================================
# UI – HEADER
# ==============================================================
st.markdown('<p class="main-title">🎓 AI Interview Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="main-title" style="font-size:1.6em;">& Candidate Evaluation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Project B2 &nbsp;|&nbsp; Abdullah Bintaleb | 4410891 &nbsp;&amp;&nbsp; Ziyad Chihani | 4312645 &nbsp;|&nbsp; Supervised by Dr. Syed Bukhari</p>', unsafe_allow_html=True)
st.markdown("---")

# ==============================================================
# UI – SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    niche = st.selectbox(
        "🏷️ Select Industry Niche",
        ["Data Science & AI", "Software Engineering", "Cybersecurity", "Cloud Computing", "General IT"]
    )
    job_desc = st.text_area(
        "📋 Paste Job Description",
        height=220,
        placeholder="e.g. We are looking for a Python Data Scientist with experience in NLP, pandas, and machine learning..."
    )
    cv_file = st.file_uploader("📄 Upload Candidate CV", type=["pdf", "docx"])
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a CV & paste a job description")
    st.markdown("2. The system extracts & compares skills")
    st.markdown("3. Gemini generates 3 interview questions")
    st.markdown("4. You answer each question")
    st.markdown("5. Sentence-BERT scores your answer semantically")
    st.markdown("6. Gemini gives you final feedback & grade")

# ==============================================================
# UI – MAIN CONTENT
# ==============================================================

if cv_file and job_desc:

    # Save uploaded file temporarily
    temp_path = "temp_cv_" + cv_file.name
    with open(temp_path, "wb") as f:
        f.write(cv_file.getbuffer())
    cv_text = read_cv(temp_path)

    # ----------------------------------------------------------
    # SECTION 1 – Skill Gap Analysis
    # ----------------------------------------------------------
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("## 📊 Skill Gap Analysis")
    st.caption("We extract skills from your CV and compare them with the job requirements.")

    cv_skills  = extract_skills(cv_text)
    jd_skills  = extract_skills(job_desc)
    matched    = cv_skills & jd_skills
    missing    = jd_skills - cv_skills

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Required Skills",  len(jd_skills))
    with col2:
        st.metric("✅ Skills Matched", len(matched))
    with col3:
        pct = int(len(matched) / len(jd_skills) * 100) if jd_skills else 0
        st.metric("🎯 Match Score", f"{pct}%")

    if jd_skills:
        st.progress(pct)

    c1, c2 = st.columns(2)
    with c1:
        st.success("**✅ Matched Skills**\n\n" + (", ".join(sorted(matched)) if matched else "None detected"))
    with c2:
        st.error("**⚠️ Skill Gaps**\n\n" + (", ".join(sorted(missing)) if missing else "No gaps – perfect match!"))
    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------
    # SECTION 2 – Interview
    # ----------------------------------------------------------
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("## 🤖 AI-Powered Interview")
    st.caption("3 questions generated by Gemini based on your CV and the job description.")

    if st.button("🚀 Generate Interview Questions"):
        with st.spinner("Gemini is analyzing the CV and generating questions..."):
            time.sleep(0.5)
            st.session_state.questions    = generate_questions(cv_text, job_desc, niche)
            st.session_state.q_index      = 0
            st.session_state.evaluations  = []

    DIFFICULTY = [
        ("🟢", "Level 1 — Very Easy"),
        ("🟡", "Level 2 — Easy but Tricky"),
        ("🔴", "Level 3 — Hard"),
    ]

    if "questions" in st.session_state and st.session_state.q_index < len(st.session_state.questions):
        idx = st.session_state.q_index
        q   = st.session_state.questions[idx]
        icon, label = DIFFICULTY[idx]

        st.markdown(f"### {icon} Question {idx + 1} of 3 &nbsp; — &nbsp; `{label}`")
        st.info(f"**{q}**")
        user_ans = st.text_area("✍️ Your Answer:", height=160, placeholder="Type your answer here...")

        if st.button("✅ Submit & Evaluate"):
            if user_ans.strip():
                with st.spinner("Running Sentence-BERT semantic similarity analysis..."):
                    ideal    = get_ideal_answer(q)
                    sim      = semantic_similarity(user_ans, ideal)
                    feedback = get_feedback(q, user_ans, ideal, sim)
                    st.session_state.evaluations.append({
                        "question":    q,
                        "user_answer": user_ans,
                        "ideal":       ideal,
                        "similarity":  sim,
                        "feedback":    feedback,
                        "difficulty":  label,
                    })
                    st.session_state.q_index += 1
                    st.rerun()
            else:
                st.warning("Please type your answer before submitting.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------
    # SECTION 3 – Final Report
    # ----------------------------------------------------------
    if "questions" in st.session_state and st.session_state.q_index >= len(st.session_state.questions):
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("## 🏆 Final Candidate Evaluation Report")
        st.caption("Scores are computed using Sentence-BERT cosine similarity + Gemini qualitative feedback.")

        for i, ev in enumerate(st.session_state.evaluations):
            icon, _ = DIFFICULTY[i]
            with st.expander(f"{icon} Q{i+1} — {ev['difficulty']}", expanded=True):
                st.markdown(f"**Question:** {ev['question']}")
                st.markdown(f"**Your Answer:** {ev['user_answer']}")
                st.markdown("---")
                col_score, col_fb = st.columns([1, 3])
                with col_score:
                    st.metric("Semantic Match", f"{ev['similarity']*100:.1f}%")
                with col_fb:
                    st.markdown("**📝 AI Feedback & Grade:**")
                    st.markdown(ev["feedback"])

        if st.button("🔄 Start a New Interview"):
            for key in ["questions", "q_index", "evaluations"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================
# UI – WELCOME SCREEN (shown before any input)
# ==============================================================
else:
    st.markdown("""
    <div class="glass" style="text-align:center; padding: 60px 30px;">
        <p class="main-title" style="font-size:2em;">🎓 AI Interview Assistant</p>
        <p class="main-title" style="font-size:1.3em;">&amp; Candidate Evaluation System</p>
        <p style="color:#94a3b8; margin-top:8px;">Project B2 &nbsp;|&nbsp; Abdullah Bintaleb | 4410891 &nbsp;&amp;&nbsp; Ziyad Chihani | 4312645 &nbsp;|&nbsp; Supervised by Dr. Syed Bukhari</p>
        <hr style="border-color:rgba(255,255,255,0.1); margin: 24px 0;">
        <p style="color:#94a3b8; font-size:1.05em;">
            Upload a <strong style="color:#e2e8f0;">CV</strong> and paste a <strong style="color:#e2e8f0;">Job Description</strong> in the sidebar to begin.
        </p>
        <br>
        <table style="margin:auto; text-align:left; color:#cbd5e1; border-spacing: 10px 8px;">
            <tr><td>📄</td><td><strong>CV Parsing</strong></td><td>Extracts text from PDF or DOCX</td></tr>
            <tr><td>🔍</td><td><strong>Skill Extraction</strong></td><td>Uses spaCy NER to find technical skills</td></tr>
            <tr><td>📊</td><td><strong>Gap Analysis</strong></td><td>Compares CV skills vs. job requirements</td></tr>
            <tr><td>🤖</td><td><strong>Question Generation</strong></td><td>Google Gemini creates 3 difficulty-leveled questions</td></tr>
            <tr><td>🧠</td><td><strong>Semantic Scoring</strong></td><td>Sentence-BERT measures answer similarity</td></tr>
            <tr><td>📝</td><td><strong>Feedback &amp; Grade</strong></td><td>Gemini gives qualitative feedback and A–F grade</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
