import streamlit as st
import spacy
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Load NLP Models
# -------------------------------------------------

# spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Sentence-BERT model for semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# Skill Dictionary with Aliases
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
# Text Preprocessing Function
# -------------------------------------------------

def preprocess(text):
    """
    Clean text using spaCy:
    - lowercase
    - remove stopwords
    - lemmatization
    """
    doc = nlp(text.lower())
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(clean_tokens)

# -------------------------------------------------
# Skill Extraction Function
# -------------------------------------------------

def extract_skills(text):
    """
    Detect skills using aliases with exact word matching.
    """
    text = text.lower()
    found_skills = []

    for main_skill, aliases in skills_db.items():
        for term in aliases:
            pattern = r"\b" + re.escape(term) + r"\b"

            if re.search(pattern, text):
                found_skills.append(main_skill)
                break

    return list(set(found_skills))

# -------------------------------------------------
# PDF Text Extraction
# -------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    """
    Read uploaded PDF and extract text page by page.
    """
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    return text

# -------------------------------------------------
# Main Analysis Function
# -------------------------------------------------

def analyze_resume(resume, job):

    # Step 1: Preprocess text
    clean_resume = preprocess(resume)
    clean_job = preprocess(job)

    # Step 2: Convert text to embeddings
    emb1 = model.encode([clean_resume])
    emb2 = model.encode([clean_job])

    # Step 3: Semantic similarity
    semantic_score = cosine_similarity(emb1, emb2)[0][0]

    # Step 4: Skill extraction
    resume_skills = extract_skills(resume)
    job_skills = extract_skills(job)

    # Step 5: Missing skills
    missing_skills = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    # Step 6: Matching skills
    matching_skills = [
        skill for skill in job_skills
        if skill in resume_skills
    ]

    # Step 7: Skill score
    if len(job_skills) > 0:
        skill_score = len(matching_skills) / len(job_skills)
    else:
        skill_score = 0

    # Step 8: Final weighted score
    final_score = (0.7 * semantic_score) + (0.3 * skill_score)

    return final_score, resume_skills, job_skills, missing_skills

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.title("AI Resume Matcher")

st.write("Upload your resume PDF and compare it with a job description.")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_text = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if uploaded_file is not None and job_text.strip() != "":

        # Extract resume text
        resume_text = extract_text_from_pdf(uploaded_file)

        # Analyze
        result = analyze_resume(resume_text, job_text)

        # Output
        st.subheader("Results")
        st.write("Final Match Score:", round(result[0] * 100, 2), "%")
        st.write("Resume Skills:", result[1])
        st.write("Job Skills:", result[2])
        st.write("Missing Skills:", result[3])

    else:
        st.warning("Please upload a resume and paste a job description.")