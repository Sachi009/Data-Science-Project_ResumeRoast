"""
Resume Match Score App
----------------------
Simple Streamlit app to compare a resume and a job description and
produce:
- Overall match score (0–100)
- Text similarity score
- Keyword coverage score
- Found and missing keywords

Run with:
    streamlit run resume_match_app.py
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import re
from typing import List, Dict

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Collapse multiple spaces/newlines
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# 3. CORE MATCHING LOGIC
# ============================================================

def get_text_similarity(resume: str, job: str) -> float:
    docs = [resume, job]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(sim)


def extract_keywords(job_text: str, top_n: int = 25) -> List[str]:
    if not job_text:
        return []

    sentences = re.split(r"[.!?\n]", job_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        sentences = [job_text]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)

    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())

    top_idx = scores.argsort()[::-1][:top_n]
    keywords = terms[top_idx]

    keywords = [k for k in keywords if len(k) > 2 and not k.isdigit()]
    return keywords


def keyword_coverage(resume_text: str, keywords: List[str]) -> Dict:
    resume = resume_text.lower()
    found = []
    missing = []

    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, resume):
            found.append(kw)
        else:
            missing.append(kw)

    coverage = (len(found) / len(keywords)) if keywords else 0.0
    return {
        "coverage": coverage,
        "found_keywords": found,
        "missing_keywords": missing,
    }


def compute_match(resume_raw: str, job_raw: str) -> Dict:
    resume = clean_text(resume_raw)
    job = clean_text(job_raw)

    similarity = get_text_similarity(resume, job)
    keywords = extract_keywords(job, top_n=25)
    kw_stats = keyword_coverage(resume, keywords)

    final_score = 0.6 * similarity + 0.4 * kw_stats["coverage"]

    return {
        "similarity": similarity,
        "keyword_coverage": kw_stats["coverage"],
        "final_score": round(final_score * 100, 1),
        "found_keywords": kw_stats["found_keywords"],
        "missing_keywords": kw_stats["missing_keywords"],
    }


# ============================================================
# 4. STREAMLIT UI
# ============================================================

def build_ui():
    st.set_page_config(page_title="Resume Match Score", page_icon="📄", layout="wide")

    st.title("📄 Resume vs Job Description Match Analyzer")

    st.markdown("""
        This app compares your **resume** with a **job description** and gives you:
        - A **match score (0–100)**
        - A **text similarity score**
        - A **keyword coverage score**
        - Lists of **found** and **missing** keywords
    """)

    col1, col2 = st.columns(2)

    with col1:
        resume_text = st.text_area("✍️ Paste Your Resume", height=300)

    with col2:
        job_text = st.text_area("📌 Paste the Job Description", height=300)

    if st.button("🔍 Analyze Match"):
        if not resume_text.strip() or not job_text.strip():
            st.warning("Please paste both resume and job description.")
            return

        results = compute_match(resume_text, job_text)

        st.subheader("✅ Overall Match Score")
        st.metric("Match Score", f"{results['final_score']} / 100")

        col3, col4 = st.columns(2)
        with col3:
            st.write(f"**Text Similarity:** {results['similarity']*100:.1f}")
            st.write(f"**Keyword Coverage:** {results['keyword_coverage']*100:.1f}")

        with col4:
            score = results["final_score"]
            if score >= 80:
                st.write("Excellent match!")
            elif score >= 60:
                st.write("Good match, consider adding a few missing keywords.")
            elif score >= 40:
                st.write("Moderate match, resume needs improvements.")
            else:
                st.write("Low match, consider rewriting resume for this job.")

        st.subheader("Keywords Found")
        st.write(", ".join(results["found_keywords"]) or "None found")

        st.subheader("Missing Keywords")
        st.write(", ".join(results["missing_keywords"]) or "None missing")


# ============================================================
# 5. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_ui()
