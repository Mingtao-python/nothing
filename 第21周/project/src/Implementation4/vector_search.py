import numpy as np
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "dataset.txt")
docs = open(DATA_PATH, encoding="utf-8").read().splitlines()
doc_vecs = model.encode(docs)

def vector_search(query, k=5):
    qv = model.encode([query])[0]
    scores = [float(np.dot(qv, dv) / (np.linalg.norm(qv) * np.linalg.norm(dv))) for dv in doc_vecs]
    idx = np.argsort(scores)[::-1][:k]
    return [(docs[i], scores[i]) for i in idx]
