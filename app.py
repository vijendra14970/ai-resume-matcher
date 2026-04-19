import streamlit as st
import spacy
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

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
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# -------------------------------------------------
# SKILL EXTRACTION
# -------------------------------------------------

def extract_skills(text):
    text = text.lower()
    found_skills = []

    for main_skill, aliases in skills_db.items():
        for term in aliases:
            if re.search(r"\b" + re.escape(term) + r"\b", text):
                found_skills.append(main_skill)
                break

    return list(set(found_skills))

# -------------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text

# -------------------------------------------------
# CORE SCORING FUNCTION
# -------------------------------------------------

def analyze_resume(resume_text, job_text):

    clean_resume = preprocess(resume_text)
    clean_job = preprocess(job_text)

    emb1 = model.encode([clean_resume])
    emb2 = model.encode([clean_job])

    semantic_score = cosine_similarity(emb1, emb2)[0][0]

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched = [s for s in job_skills if s in resume_skills]

    skill_score = len(matched) / len(job_skills) if job_skills else 0

    final_score = (0.7 * semantic_score) + (0.3 * skill_score)

    return final_score, resume_skills, job_skills, matched

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

st.title("AI Resume Matcher - Ranking System")

st.write("Upload multiple resumes and compare them with a job description.")

job_text = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Rank Candidates"):

    if uploaded_files and job_text.strip() != "":

        results = []

        # PROCESS EACH RESUME
        for file in uploaded_files:

            resume_text = extract_text_from_pdf(file)

            score, res_skills, job_skills, matched = analyze_resume(
                resume_text,
                job_text
            )

            results.append({
                "Candidate": file.name,
                "Score": round(score * 100, 2),
                "Skills": res_skills,
                "Matched Skills": matched
            })

        # SORT BY SCORE
        results = sorted(results, key=lambda x: x["Score"], reverse=True)

        # DISPLAY RANKING
        st.subheader("Candidate Ranking")

        for i, r in enumerate(results):
            st.write(f"Rank {i+1}")
            st.write("Name:", r["Candidate"])
            st.write("Score:", r["Score"], "%")
            st.write("Skills:", r["Skills"])
            st.write("Matched Skills:", r["Matched Skills"])
            st.write("---")

    else:
        st.warning("Please upload resumes and paste job description.")
