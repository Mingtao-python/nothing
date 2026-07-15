import numpy as np
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "dataset.txt")
texts = open(DATA_PATH, encoding="utf-8").read().splitlines()
emb = model.encode(texts)

def embedding_similarity_matrix():
    matrix = []
    for i in range(len(texts)):
        row = []
        for j in range(len(texts)):
            sim = float(np.dot(emb[i], emb[j]) / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j])))
            row.append(sim)
        matrix.append(row)
    return matrix
