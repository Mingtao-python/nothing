from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "dataset.txt")

docs = open(DATA_PATH, encoding="utf-8").read().splitlines()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

def tfidf_search(query, top_k=5):
    q = vectorizer.transform([query])
    scores = (X * q.T).toarray().flatten()
    idx = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], float(scores[i])) for i in idx]
