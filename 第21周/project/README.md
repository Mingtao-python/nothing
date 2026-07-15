# 📌 Project Overview
This project implements a complete AI Semantic Search Assistant capable of understanding natural‑language queries and retrieving the most relevant results using modern embedding‑based semantic search techniques.

The system demonstrates:

How text is converted into dense vectors

How similarity is computed using Cosine Similarity and Euclidean Distance

How TF‑IDF differs from semantic search

How a Prompt Filter improves input safety

How a simple Web UI can expose AI search functionality

This project is part of Week 21 – Embedding & Neural Network Fundamentals.

## ✨ Features
Embedding‑based Semantic Search (Sentence‑Transformers)

TF‑IDF Retrieval Engine for comparison

Cosine Similarity (custom implementation)

Euclidean Distance (custom implementation)

Top‑5 Ranking with similarity scores

Prompt Filter (length check, special‑character detection, basic malicious prompt detection)

Web Interface built with Flask

50+ Text Dataset for testing

20+ Query Test Cases

### 📁 Project Structure
Week21_Engineering/
│
├── Implementation1/Similarity_Engine/
│   └── similarity.py
│
├── Implementation2/TFIDF_Retrieval/
│   └── tfidf_search.py
│
├── Implementation3/Embedding_Similarity/
│   └── embedding_similarity.py
│
├── Implementation4/Vector_Search/
│   └── vector_search.py
│
├── Implementation5/Embedding_Search/
│   ├── embedding_search.py
│   ├── dataset.txt
│   └── app.py
│
├── Implementation6/Gradient_Descent/
│   └── gradient_demo.py
│
├── Implementation7/Embedding_Failure_Analysis/
│   └── failure_analysis.py
│
└── README.md
## ⚙️ Installation
1. Clone the repository

git clone https://github.com/mingtao-python/Week21.git
cd Week21_Engineering
2. Install dependencies

```
pip install -r requirements.txt
```

### Typical dependencies:

flask

scikit-learn

numpy

sentence-transformers

🚀 Usage
Run the Web UI
```
python app.py
```
Then open:
```
http://127.0.0.1:5000
```
Run TF‑IDF Search
```
python tfidf_search.py
```
Run Embedding Similarity
```
python embedding_similarity.py
```
Run Gradient Descent Simulation
```
python gradient_demo.py
```
## 🔍 How It Works
1. Prompt Filter
Rejects overly long inputs

Detects suspicious characters

Prevents basic prompt injection attempts

2. Embedding Generation
Converts text into dense vectors

Captures semantic meaning

3. Similarity Calculation
Cosine Similarity → measures direction

Euclidean Distance → measures magnitude

4. Ranking
Sorts results by similarity score

Returns Top‑5 most relevant texts

5. Web UI
Simple interface for entering queries

Displays ranked results and scores

### 📊 Example Output
Query: "machine learning"

Top‑5 Results:
1. deep learning uses neural networks — 0.67
2. transformers changed NLP — 0.43
3. vector search finds similar documents — 0.39
etc
## 📘 Model Analysis Summary
Embedding > Keyword Search  
Embeddings capture meaning, not just exact words.

Cosine Similarity > Euclidean Distance  
Text meaning depends on direction, not vector length.

## 🔐 Security Summary
Prompt Injection Entry Point: User input

Embedding Does NOT execute prompts

RAG can still be poisoned if malicious text enters the vector database

Prompt Filter reduces risk

Remaining risks: semantic bypass, indirect injection

## 🧠 Future Improvements!!!
Add FAISS / Milvus vector database

Add Reranker (e.g., cross‑encoder)

Add RAG pipeline

Add multi‑algorithm comparison dashboard

Add user analytics & logging

Add GPU acceleration

## 📄 License
There are no License

## 🙌 Acknowledgements
Sentence‑Transformers

Scikit‑Learn

Flask

Week 21 Course Materials