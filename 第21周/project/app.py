from flask import Flask, render_template, request
from src.Implementation1.similarity import cosine_similarity
from src.Implementation2.tfidf_search import search
from src.Implementation3.embedding_similarity import cosine
from src.Implementation4.vector_search import top_k
from src.Implementation5.embedding_search import search2

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        q = request.form["query"]
        cosine_result = cosine_similarity(q, "example text")
        tfidf_results = search(q)
        embedding_matrix = cosine()
        vector_results = top_k(q)
        embedding_results = search2(q)
        return render_template(
            "index.html",
            cosine_result=cosine_result,
            tfidf_results=tfidf_results,
            embedding_matrix=embedding_matrix,
            vector_results=vector_results,
            embedding_results=embedding_results
        )

    return render_template("index.html")