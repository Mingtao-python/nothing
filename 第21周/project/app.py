from flask import Flask, render_template, request
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# ====== 通用配置 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.txt")
docs = open(DATA_PATH, encoding="utf-8").read().splitlines()

# ====== Embedding 模型 ======
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = model.encode(docs)

def cosine_vec(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ====== Implementation 1：Similarity Engine ======
def cosine_similarity(text1, text2):
    v1 = model.encode([text1])[0]
    v2 = model.encode([text2])[0]
    return cosine_vec(v1, v2)

# ====== Implementation 2：TF-IDF Retrieval ======
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

def tfidf_search(query, top_k=5):
    q = vectorizer.transform([query])
    scores = (X * q.T).toarray().flatten()
    idx = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], float(scores[i])) for i in idx]

# ====== Implementation 3：Embedding Similarity Matrix ======
def embedding_similarity_matrix():
    emb = doc_vecs
    matrix = []
    for i in range(len(docs)):
        row = []
        for j in range(len(docs)):
            sim = cosine_vec(emb[i], emb[j])
            row.append(sim)
        matrix.append(row)
    return matrix

# ====== Implementation 4：Vector Search ======
def vector_search(query, k=5):
    qv = model.encode([query])[0]
    scores = [cosine_vec(qv, dv) for dv in doc_vecs]
    idx = np.argsort(scores)[::-1][:k]
    return [(docs[i], scores[i]) for i in idx]

# ====== Implementation 5：Embedding Search ======
def embedding_search(query, k=5):
    qv = model.encode([query])[0]
    scores = [cosine_vec(qv, dv) for dv in doc_vecs]
    idx = np.argsort(scores)[::-1][:k]
    return [(docs[i], scores[i]) for i in idx]

# ====== Implementation 6：Gradient Descent Simulation ======
def run_gradient_demo():
    X = np.linspace(0, 10, 50)
    Y = 2 * X + 1
    w = 0
    b = 0
    lr = 0.01
    losses = []
    for epoch in range(200):
        pred = w * X + b
        loss = np.mean((pred - Y) ** 2)
        losses.append(float(loss))
        dw = np.mean(2 * (pred - Y) * X)
        db = np.mean(2 * (pred - Y))
        w -= lr * dw
        b -= lr * db
    return losses

# ====== Implementation 7：Embedding Failure Analysis ======
def find_failure_cases(query):
    return [
        "Ambiguous word: bank (river vs finance)",
        "Out-of-domain query",
        "Very short or noisy text",
        "Rare words not in training data",
        "Query with multiple mixed intents"
    ]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        q = request.form["query"]

        cosine_result = cosine_similarity(q, "example text")
        tfidf_results = tfidf_search(q)
        embedding_matrix = embedding_similarity_matrix()
        vector_results = vector_search(q)
        embedding_results = embedding_search(q)
        loss_curve = run_gradient_demo()
        failure_cases = find_failure_cases(q)

        return render_template(
            "index.html",
            query=q,
            cosine_result=cosine_result,
            tfidf_results=tfidf_results,
            embedding_matrix=embedding_matrix,
            vector_results=vector_results,
            embedding_results=embedding_results,
            loss_curve=loss_curve,
            failure_cases=failure_cases,
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
