# AI Resume Matcher

An AI-powered resume screening system that compares a candidate's resume with a job description using Natural Language Processing, Sentence-BERT semantic similarity, and skill matching.

## Live demo
Try the deployed app here: https://smart-resume-screening.streamlit.app

## Project Overview

Recruiters often receive many resumes and manually compare them with job requirements.  
This project automates the first stage of screening by estimating how well a resume matches a job role.

The system analyzes:

- Semantic similarity between resume and job description
- Skill overlap
- Missing required skills

## Features

- Upload Resume in PDF format
- Paste Job Description
- Extract text from PDF automatically
- NLP-based text preprocessing
- Sentence-BERT semantic matching
- Skill extraction with aliases
- Missing skill detection
- Final weighted match score
- Clean Streamlit web interface

## Tech Stack

- Python
- Streamlit
- spaCy
- Sentence-Transformers
- Scikit-learn
- pdfplumber
- Regex

## How It Works

### Step 1: Resume Extraction
Reads uploaded PDF resume and extracts text.

### Step 2: Text Preprocessing
Uses spaCy for:
- Lowercasing
- Stopword removal
- Lemmatization

### Step 3: Semantic Matching
Sentence-BERT converts resume and job description into embeddings.

Cosine similarity compares meaning.

### Step 4: Skill Matching
Checks required skills present in resume.

### Step 5: Final Score

70% Semantic Similarity + 30% Skill Match


