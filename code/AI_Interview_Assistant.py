import os
import streamlit as st
from google import genai
from PyPDF2 import PdfReader
from docx import Document
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---------- Configuration & Setup ----------
st.set_page_config(page_title="AI Interview Assistant (B2)", page_icon="🎓", layout="wide")

# Title and Student Info (Student Project Requirement)
st.title("🎓 AI Interview Assistant & Candidate Evaluation System (Project B2)")
st.markdown("**Prepared by:** Abdullah Bintaleb (4410891) & Ziyad Chihani (4312645)")
st.markdown("**Supervised by:** Dr. Syed Bukhari")
st.markdown("---")

# Initialize Gemini Client
# Make sure to set this API key securely in a real environment
client = genai.Client(api_key="Your API here ")

# Load ML Models (Cached to prevent reloading on every interaction)
@st.cache_resource
def load_models():
    # Load SpaCy for Skill Extraction (NER)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        nlp = None
        
    # Load Sentence-BERT for Semantic Similarity
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, sbert_model

nlp, sbert_model = load_models()

# ---------- Helper Functions ----------

def read_cv(file_path):
    """Extract text from PDF or DOCX files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    elif ext == ".docx":
        doc = Document(file_path)
        text = ""
        for p in doc.paragraphs:
            text += p.text + "\n"
        return text
    else:
        return "Unsupported file type"

def extract_skills(text, nlp_model):
    """
    Simple Skill Extraction using SpaCy. 
    For a student project, we simulate this by looking for specific Nouns/Proper Nouns 
    that might be technical terms, or matching against a known list of skills.
    To keep it simple, we will do a basic keyword match from a predefined list of tech skills.
    """
    # A simple knowledge base of common tech skills
    TECH_SKILLS = [
        "python", "java", "c++", "machine learning", "deep learning", "nlp", 
        "data science", "fastapi", "streamlit", "sql", "react", "aws", "docker",
        "spacy", "scikit-learn", "numpy", "pandas", "pytorch", "tensorflow",
        "javascript", "html", "css", "git", "linux", "agile", "scrum"
    ]
    
    extracted = set()
    if nlp_model is None:
        return list(extracted)
        
    doc = nlp_model(text.lower())
    # Extract noun chunks and tokens to match against our list
    for token in doc:
        if token.text in TECH_SKILLS:
            extracted.add(token.text.title())
            
    # Also check multi-word skills
    text_lower = text.lower()
    for skill in TECH_SKILLS:
        if skill in text_lower:
            extracted.add(skill.title())
            
    return list(extracted)

def generate_questions(cv_text, job_desc, niche):
    """Generate interview questions using Google Gemini."""
    prompt = f"""
    You are an expert technical recruiter specializing in {niche}.
    Based on the candidate's CV and the Job Description, generate 3 highly relevant interview questions.
    Focus on checking the candidate's actual experience and any potential skill gaps.
    
    Job Description:
    {job_desc}
    
    CV Content:
    {cv_text}
    
    Return ONLY the questions, each on a new line.
    """
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    # Filter out empty lines and return as list
    return [q.strip() for q in response.text.split("\n") if q.strip()]

def get_ideal_answer(question):
    """Generate an ideal answer for a question to use as a baseline for similarity."""
    prompt = f"Provide a concise, highly accurate, and professional answer to this interview question: {question}"
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text

def evaluate_answer_semantic(user_answer, ideal_answer, sbert_model):
    """Use Sentence-BERT to compute semantic similarity between user answer and ideal answer."""
    # Encode sentences into vectors
    embeddings = sbert_model.encode([user_answer, ideal_answer])
    # Compute Cosine Similarity between the two vectors
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

def get_llm_feedback(question, user_answer, ideal_answer, similarity_score):
    """Use LLM to provide qualitative feedback and a final grade."""
    prompt = f"""
    You are an interview evaluator.
    Question: {question}
    Candidate's Answer: {user_answer}
    Ideal Answer: {ideal_answer}
    Semantic Similarity Score: {similarity_score * 100:.1f}%
    
    Based on the answer and the similarity score, provide:
    1. A short qualitative feedback (what was good, what was missing).
    2. A final Grade (A, B, C, D, or F).
    """
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text

# ---------- Streamlit UI Workflow ----------

# Sidebar for inputs
with st.sidebar:
    st.header("📝 1. Input Information")
    niche_focus = st.selectbox("Select Niche Focus", ["Software Engineering", "Data Science & AI", "Cybersecurity", "General IT"])
    job_description = st.text_area("Paste Job Description Here")
    uploaded_file = st.file_uploader("Upload Candidate CV (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file and job_description:
    # Save temp file to read
    temp_path = "temp_" + uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    cv_text = read_cv(temp_path)
    
    # --- PHASE 1: Skill Gap & Visualization ---
    st.header("📊 2. Skill Gap Insights & Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Candidate Skills")
        cv_skills = extract_skills(cv_text, nlp)
        st.write(", ".join(cv_skills) if cv_skills else "No skills identified.")
        
    with col2:
        st.subheader("Required Skills (Job Desc)")
        jd_skills = extract_skills(job_description, nlp)
        st.write(", ".join(jd_skills) if jd_skills else "No skills identified.")
        
    # Calculate Gaps
    cv_skills_set = set(cv_skills)
    jd_skills_set = set(jd_skills)
    
    matched_skills = jd_skills_set.intersection(cv_skills_set)
    missing_skills = jd_skills_set.difference(cv_skills_set)
    
    # Visualization: Simple Bar Chart
    st.markdown("### Skill Match Analysis")
    if jd_skills_set:
        match_percentage = len(matched_skills) / len(jd_skills_set) * 100
        st.progress(int(match_percentage), text=f"Match Score: {int(match_percentage)}%")
        
        # Simple Dataframe for the chart
        chart_data = pd.DataFrame({
            "Category": ["Matched Skills", "Missing Skills"],
            "Count": [len(matched_skills), len(missing_skills)]
        }).set_index("Category")
        st.bar_chart(chart_data)
        
        if missing_skills:
            st.warning(f"**Missing Skills Detected:** {', '.join(missing_skills)}")
        else:
            st.success("Candidate matches all extracted required skills!")
    else:
        st.info("Could not extract technical skills from the Job Description to create a chart.")

    st.markdown("---")
    
    # --- PHASE 2: Interview Generation ---
    st.header("🤖 3. AI Interview Assistant")
    
    if st.button("Generate Interview Questions"):
        with st.spinner("Generating niche-specific questions..."):
            st.session_state.questions = generate_questions(cv_text, job_description, niche_focus)
            st.session_state.q_index = 0
            st.session_state.evaluations = []
            
    # --- PHASE 3: Interview & Evaluation ---
    if "questions" in st.session_state and st.session_state.q_index < len(st.session_state.questions):
        current_q = st.session_state.questions[st.session_state.q_index]
        st.subheader(f"Question {st.session_state.q_index + 1} of {len(st.session_state.questions)}")
        st.info(f"**{current_q}**")
        
        user_answer = st.text_area("Candidate Answer:", height=150)
        
        if st.button("Submit Answer"):
            if user_answer.strip() == "":
                st.error("Please provide an answer.")
            else:
                with st.spinner("Evaluating answer using Semantic Similarity & LLM..."):
                    ideal_ans = get_ideal_answer(current_q)
                    sim_score = evaluate_answer_semantic(user_answer, ideal_ans, sbert_model)
                    feedback = get_llm_feedback(current_q, user_answer, ideal_ans, sim_score)
                    
                    st.session_state.evaluations.append({
                        "question": current_q,
                        "user_answer": user_answer,
                        "ideal_answer": ideal_ans,
                        "similarity": sim_score,
                        "feedback": feedback
                    })
                    st.session_state.q_index += 1
                    st.rerun()

    # --- PHASE 4: Final Report ---
    if "questions" in st.session_state and st.session_state.q_index >= len(st.session_state.questions):
        st.header("📋 4. Final Evaluation Report")
        st.success("Interview Completed!")
        
        for i, eval_data in enumerate(st.session_state.evaluations):
            with st.expander(f"Q{i+1}: {eval_data['question']}"):
                st.write(f"**Candidate Answer:** {eval_data['user_answer']}")
                st.write(f"**Ideal Answer (Baseline):** {eval_data['ideal_answer']}")
                
                # Display metric visually
                score_pct = eval_data['similarity'] * 100
                st.metric("Semantic Similarity (Sentence-BERT)", f"{score_pct:.1f}%")
                
                st.write("**LLM Feedback & Grade:**")
                st.write(eval_data['feedback'])
        
        if st.button("Start New Interview"):
            del st.session_state.questions
            del st.session_state.q_index
            del st.session_state.evaluations
            st.rerun()
else:
    st.info("👈 Please upload a CV and paste a Job Description in the sidebar to begin.")
