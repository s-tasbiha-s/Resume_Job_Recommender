import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download NLTK stopwords (downloads only on the first run)
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# -------------------------------------------------------------
# 1. TEXT PREPROCESSING & PDF EXTRACTION
# -------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalizes text by converting to lowercase and removing special characters and stopwords."""
    if not text or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts raw text from an uploaded PDF resume."""
    pdf_reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + " "
    return extracted_text

def extract_skills_set(skill_string: str) -> set:
    """Parses semicolon/comma-separated skills into a normalized set."""
    if pd.isna(skill_string):
        return set()
    skills = [s.strip().lower() for s in re.split(r'[,;|]', skill_string) if s.strip()]
    return set(skills)

# -------------------------------------------------------------
# 2. CASE-BASED RS & CONTENT-BASED MATCHING ENGINE
# -------------------------------------------------------------
def run_recommender(resume_raw_text: str, jobs_df: pd.DataFrame, selected_title_filter: str, top_n: int = 5):
    """Combines TF-IDF Content-Based Filtering with Case-Based Reasoning constraints."""
    cleaned_resume = clean_text(resume_raw_text)
    resume_lower = resume_raw_text.lower()
    
    # Case-Based RS Rule: Filter dataset based on user-specified job role constraint
    filtered_df = jobs_df.copy()
    if selected_title_filter != "All":
        filtered_df = filtered_df[filtered_df['job_title'].str.contains(selected_title_filter, case=False, na=False)]
    
    if filtered_df.empty:
        return pd.DataFrame()

    # Preprocess job descriptions
    filtered_df['cleaned_desc'] = filtered_df['job_description'].apply(clean_text)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    all_corpus = [cleaned_resume] + filtered_df['cleaned_desc'].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_corpus)

    # Cosine Similarity Calculation
    resume_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(resume_vector, job_vectors).flatten()

    filtered_df['match_score'] = np.round(similarities * 100, 2)
    
    # Skill-Gap Analysis
    missing_skills_list = []
    matching_skills_list = []
    
    for _, row in filtered_df.iterrows():
        required_skills = extract_skills_set(str(row['required_skills']))
        matched = set()
        missing = set()
        
        for skill in required_skills:
            # Whole-word regex match to prevent partial overlaps (e.g., 'java' vs 'javascript')
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, resume_lower):
                matched.add(skill)
            else:
                missing.add(skill)
        
        matching_skills_list.append(", ".join([s.title() for s in sorted(matched)]) if matched else "None Detected")
        missing_skills_list.append(", ".join([s.title() for s in sorted(missing)]) if missing else "None (Fully Matched)")

    filtered_df['matched_skills'] = matching_skills_list
    filtered_df['missing_skills'] = missing_skills_list

    # Rank Top-N Jobs by Match Score
    results_df = filtered_df.sort_values(by='match_score', ascending=False).head(top_n)
    return results_df

# -------------------------------------------------------------
# 3. STREAMLIT WEB INTERFACE (GUI)
# -------------------------------------------------------------
def main():
    st.set_page_config(page_title="Resume-to-Job Matcher & Skill Gap Recommender", layout="wide")
    st.title("📄 Resume-to-Job Matching & Skill Gap Recommender")
    st.markdown("An explainable recommendation system combining **Content-Based Filtering (TF-IDF & Cosine Similarity)** with **Case-Based Reasoning**.")

    # Load Dataset
    data_path = os.path.join("data", "jobs.csv")
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at `{data_path}`. Please run `python prepare_data.py` first to generate the dataset.")
        return
    
    jobs_df = pd.read_csv(data_path)

    # Sidebar: Case-Based Preferences
    st.sidebar.header("🎯 Case-Based Preferences")
    unique_titles = ["All"] + sorted(jobs_df['job_title'].dropna().unique().tolist())
    selected_title = st.sidebar.selectbox("Select Target Job Role (Constraint):", unique_titles)
    top_n = st.sidebar.slider("Number of Top Recommendations:", 1, 10, 5)

    # Resume PDF Uploader
    uploaded_file = st.file_uploader("Upload Your Resume (PDF Format)", type=["pdf"])

    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        
        with st.expander("Preview Extracted Resume Text"):
            st.write(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))

        if st.button("Generate Job & Skill-Gap Recommendations", type="primary"):
            with st.spinner("Analyzing resume and computing similarity scores..."):
                results = run_recommender(resume_text, jobs_df, selected_title, top_n)
                
                if results.empty:
                    st.warning("No matching jobs found for the selected role constraint.")
                else:
                    st.success(f"Top {len(results)} Relevant Job Recommendations Found:")
                    
                    for _, row in results.iterrows():
                        st.markdown(f"### 💼 {row['job_title']} — Match Score: `{row['match_score']}%`")
                        st.write(f"**Job Description:** {row['job_description']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"✅ **Matched Skills:** {row['matched_skills']}")
                        with col2:
                            st.warning(f"⚠️ **Missing Skills (Skill Gap):** {row['missing_skills']}")
                        
                        st.divider()

if __name__ == "__main__":
    main()

    # streamlit run app.py