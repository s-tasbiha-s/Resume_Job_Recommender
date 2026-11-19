# 📄 Resume-to-Job Matching & Skill-Gap Recommender

An explainable recommendation system that analyzes a candidate's PDF resume, compares it against job descriptions, recommends the most relevant roles, and identifies missing skills to help candidates improve their job readiness.

---

## 🚀 Key Features
* **Automated PDF Resume Parsing:** Extracts and normalizes text directly from uploaded PDF resumes.
* **Hybrid Recommendation Engine:** Combines **Content-Based Filtering** (TF-IDF + Cosine Similarity) with **Case-Based Reasoning** (role/category constraint filtering).
* **Match Scoring:** Calculates a percentage similarity score for each recommended job.
* **Personalized Skill-Gap Analysis:** Compares job requirements against the candidate's resume to highlight both **Matched Skills** and **Missing Skills**.
* **Interactive Web Interface:** Built with Streamlit for seamless testing and demonstration.

---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Frontend / GUI:** Streamlit
* **Machine Learning & NLP:** Scikit-Learn (TF-IDF, Cosine Similarity), NLTK
* **Data Processing:** Pandas, NumPy
* **PDF Extraction:** pypdf

---

## 📂 Project Structure
```text
Resume_job_recommender/
│
├── data/
│   └── jobs.csv            # Cleaned job descriptions and required skills dataset
├── app.py                  # Main Streamlit web application & recommendation engine
├── prepare_data.py         # Data preprocessing script for raw Kaggle datasets
├── requirements.txt        # Required Python dependencies
└── README.md               # Project documentation
