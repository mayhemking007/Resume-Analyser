# Resume Analyser

**Short:** Web app (Flask) to upload **5+ resumes** and a **job description** — returns the **top 3 resumes** ranked by similarity (TF-IDF + cosine similarity) with similarity scores.

---

## Features
- Upload multiple resumes (PDF/DOCX/TXT) and a job description via a simple Flask UI.
- Preprocesses text (basic cleaning + optional lemmatization).
- Uses `TfidfVectorizer` and cosine similarity to score each resume vs the JD.
- Returns top-3 resumes with similarity scores (0–100%).
- Lightweight, easily deployable (Docker / Heroku / Render).

---

## Tech stack
- Python, Flask (UI + backend)
- scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
  - PyPDF2 / python-docx for extracting text from files

---

## Quickstart (local)

1. Clone repo:
```
git clone https://github.com/mayhemking007/Resume-Analyser.git
cd Resume-Analyser
```
2. Create virtual env & install:
```
python -m venv .venv
.venv\Scripts\activate on Windows
pip install -r requirements.txt
```
3. Run:
```
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
# or
python app.py
```

## Core algorithm (concept + minimal code)
- Combine resume texts and job description to build a TF-IDF vocabulary (recommended).
- Vectorize, then compute cosine similarity between the JD vector and each resume vector.
- Rank resumes descending and return top 3.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# resume_texts: list[str]  (length >= 5)
# job_desc: str
all_docs = resume_texts + [job_desc]

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.85)
tfidf_matrix = vectorizer.fit_transform(all_docs)

resume_matrix = tfidf_matrix[:-1]   # first N rows
jd_vector = tfidf_matrix[-1]        # last row

sims = cosine_similarity(jd_vector, resume_matrix)[0]  # shape (N,)
# get top 3
top_idx = sims.argsort()[::-1][:3]
top_results = [{"filename": filenames[i], "score": float(sims[i]*100)} for i in top_idx]
```

## Limitations & improvements
- TF-IDF is fast and explainable but misses semantic matches. Consider sentence-transformers (SBERT) for embeddings + cosine for better semantic matches.

- Improve resume parsing (section-aware: skills, experience, education) — weight skill matches higher.

- Normalize synonyms (e.g., “JavaScript” vs “JS”).

- Add unit tests for parser & ranking and set up CI.