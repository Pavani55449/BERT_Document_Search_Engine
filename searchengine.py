from flask import Flask, render_template, request
import sqlite3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Connect to the SQLite database
def connect_db(database_path):
    conn = sqlite3.connect(database_path)
    return conn

# Function to retrieve top 10 unique names based on query
def get_top_10_unique_names(query, conn, model):
    c = conn.cursor()
    query_embedding = model.encode([query])
    similarities = []
    c.execute("SELECT * FROM records")
    for row in c.fetchall():
        record_num, record_name, record_embeddings = row
        embeddings = json.loads(record_embeddings)
        embeddings = np.array(embeddings).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, embeddings)[0][0]
        similarities.append((record_name, similarity))
    
    sorted_names = [name for name, _ in sorted(similarities, key=lambda x: x[1], reverse=True)]
    unique_names = []
    for name in sorted_names:
        if name not in unique_names:
            unique_names.append(name)
            if len(unique_names) == 10:
                break
    
    return unique_names

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            conn = connect_db('D:/ML/Innomatics_Research_Lab_Internship/task8_SearchEngine/flask_webapp/chroma_2k.sqlite3')
            model = SentenceTransformer('bert-base-nli-mean-tokens')
            top_10_unique_names = get_top_10_unique_names(query, conn, model)
            conn.close()
            return render_template("index.html", documents=top_10_unique_names)
    return render_template("index.html", documents=None)

if __name__ == "__main__":
    app.run(debug=True)
