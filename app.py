import streamlit as st
import spacy
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

# -------------------------------------------------
# SAFE MODEL LOADING (IMPORTANT FOR STREAMLIT CLOUD)
# -------------------------------------------------

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

nlp = load_spacy()
model = load_model()

# -------------------------------------------------
# SKILL DATABASE
# -------------------------------------------------

skills_db = {
    "python": ["python"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "computer vision": ["computer vision", "cv"],
    "artificial intelligence": ["ai", "artificial intelligence"],
    "nlp": ["nlp", "natural language processing"],
    "sql": ["sql"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch"],
    "opencv": ["opencv"],
    "data analysis": ["data analysis", "analytics"]
}

# -------------------------------------------------
# TEXT PREPROCESSING
# -------------------------------------------------

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and not t.is_punct])

# -------------------------------------------------
# SKILL EXTRACTION
# -------------------------------------------------

def extract_skills(text):
    text = text.lower()
    found = set()

    for skill, aliases in skills_db.items():
        for a in aliases:
            if re.search(r"\b" + re.escape(a) + r"\b", text):
                found.add(skill)
                break

    return list(found)

# -------------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# -------------------------------------------------
# CORE ANALYSIS
# -------------------------------------------------

def analyze_resume(resume_text, job_text):

    resume_clean = preprocess(resume_text)
    job_clean = preprocess(job_text)

    emb_resume = model.encode([resume_clean])
    emb_job = model.encode([job_clean])

    semantic_score = cosine_similarity(emb_resume, emb_job)[0][0]

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched = list(set(resume_skills).intersection(set(job_skills)))

    skill_score = len(matched) / len(job_skills) if job_skills else 0

    final_score = (0.7 * semantic_score) + (0.3 * skill_score)

    return final_score, resume_skills, job_skills, matched

# -------------------------------------------------
# UI
# -------------------------------------------------

st.title("AI Resume Matcher")
st.write("Upload multiple resumes and compare them with a job description.")

job_text = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Rank Candidates"):

    if not job_text.strip():
        st.warning("Please enter job description")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one resume")
        st.stop()

    results = []

    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)

        score, res_skills, job_skills, matched = analyze_resume(
            resume_text,
            job_text
        )

        results.append({
            "name": file.name,
            "score": round(score * 100, 2),
            "skills": res_skills,
            "matched": matched
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    st.subheader("Ranking Results")

    for i, r in enumerate(results, 1):
        st.markdown(f"### Rank {i}")
        st.write("**Name:**", r["name"])
        st.write("**Score:**", r["score"], "%")
        st.write("**Skills:**", r["skills"])
        st.write("**Matched Skills:**", r["matched"])
        st.divider()