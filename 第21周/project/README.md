# AI Semantic Search Assistant

This project implements a complete **AI Semantic Search Assistant** capable of understanding natural-language queries and retrieving the most relevant results using modern embedding-based semantic search techniques.

The system demonstrates:

- How text is converted into **dense vectors**
- How similarity is computed using **Cosine Similarity** and **Euclidean Distance**
- How **TF-IDF** differs from semantic search
- How a **Prompt Filter** improves input safety
- How a simple **Web UI** can expose AI search functionality

This project is part of **Week 21 – Embedding & Neural Network Fundamentals**.

---

## ✨ Features

- Embedding-based Semantic Search (Sentence-Transformers)
- TF-IDF Retrieval Engine for comparison
- Cosine Similarity (custom implementation)
- Euclidean Distance (via vector operations)
- Top-5 Ranking with similarity scores
- Simple Web Interface built with Flask
- 50+ text dataset for testing
- 20+ query test cases

---

## 📁 Project Structure

```text
project/
│  app.py
│  dataset.txt
│  prompt_filter.py
│  README.md
│
├─src
│  ├─Implementation1
│  │      similarity.py
│  ├─Implementation2
│  │      tfidf_search.py
│  ├─Implementation3
│  │      embedding_similarity.py
│  ├─Implementation4
│  │      vector_search.py
│  ├─Implementation5
│  │      embedding_search.py
│  ├─Implementation6
│  │      gradient_descent_demo.py
│  └─Implementation7
│         embedding_failure_analysis.py
│
└─templates
       index.html
```
## ⚙️ Installation
```
git clone https://github.com/mingtao-python/Week21.git
cd Week21/project
pip install -r requirements.txt
```
### Typical dependencies:
- flask  
- scikit-learn  
- numpy  
- sentence-transformers  

## 🚀 Usage
Run the Web UI:

```
python app.py
```
Then open:

```link
http://127.0.0.1:5000
```
## 🔍 How It Works
### Implementation 1 – Similarity Engine  
Computes cosine similarity between two texts using embeddings.

### Implementation 2 – TF-IDF Retrieval  
Builds a TF-IDF index over dataset.txt and returns Top-5 results.

### Implementation 3 – Embedding Similarity Matrix  
Computes a full similarity matrix over all documents.

### Implementation 4 – Vector Search  
Performs Top-5 semantic search using dense vectors.

### Implementation 5 – Embedding Search  
Core semantic search engine used in the product.

### Implementation 6 – Gradient Descent Simulation  
Simulates training a simple linear model and records the loss curve.

### Implementation 7 – Embedding Failure Analysis  
Lists typical failure cases of embedding-based systems.

#### All modules are integrated into app.py and exposed via the Web UI.

## 📘 Model Analysis Summary
### Embedding > Keyword Search  
Embeddings capture semantic meaning instead of exact word matches.

### Cosine Similarity > Euclidean Distance  
For text, direction of vectors matters more than magnitude.

### Failure Cases  
Ambiguous words, out-of-domain queries, short/noisy text, rare words.

## 🔐 Security Summary
Prompt Injection entry point: user input

Embeddings do not execute prompts, but RAG pipelines can still be poisoned.

A prompt filter and input validation can reduce risk, but semantic bypass remains possible.

## 🧠 Future Improvements
- Integrate FAISS / Milvus as a vector database
- Add reranking with cross-encoders
- Add full RAG pipeline
- Add analytics and logging
- Add GPU acceleration for large-scale search

## 📄 License
No explicit license is provided for this project.

🙌 Acknowledgements
- Sentence-Transformers
- Scikit-Learn
- Flask
- Week 21 Course Materials