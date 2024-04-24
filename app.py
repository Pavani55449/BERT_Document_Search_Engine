from flask import Flask, request, jsonify, render_template
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Constants
CHROMA_DB_FILE = "subtitle_embeddings_5.db"

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Initialize BERT model
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def retrieve_documents(query_embedding, top_n=10):
    try:
        print("Connecting to the database...")
        chroma_conn = sqlite3.connect(CHROMA_DB_FILE)
        chroma_c = chroma_conn.cursor()
        print("Database connection successful.")

        print("Retrieving embeddings from the database...")
        chroma_c.execute("SELECT name, embedding FROM embeddings")
        rows = chroma_c.fetchall()
        print("Retrieved embeddings successfully.")

        # Calculate similarity scores
        scores = []
        for name, embedding_blob in tqdm(rows, desc="Calculating similarity scores"):
            # Convert the binary embedding data back to a NumPy array
            embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)

            # Calculate the cosine similarity between the query embedding and the current embedding
            score = cosine_similarity(query_embedding, embedding)

            # Append the document name and score to the list of scores
            scores.append((name, score))

        # Sort the scores by similarity in descending order
        scores.sort(key=lambda x: x[1], reverse=True)

        # Retrieve the top_n results while ensuring unique document names
        top_documents = []
        unique_names = set()  # Use a set to track unique document names

        for name, _ in scores:
            if name not in unique_names:
                top_documents.append(name)
                unique_names.add(name)

            if len(top_documents) == top_n:
                break

        print("Retrieval of documents successful.")

        # Close the connection
        chroma_conn.close()
        print("Database connection closed.")

        # Return the top 10 unique document names
        return top_documents

    except Exception as e:
        print("An error occurred:", e)
        return []


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_query = request.form['query']
        processed_query = preprocess_text(user_query)
        query_embedding = bert_model.encode([processed_query])
        top_documents = retrieve_documents(query_embedding)
        return render_template('index1.html', documents=top_documents)
    return render_template('index1.html', documents=[])  # Pass an empty list if no search has been performed yet


if __name__ == '__main__':
    app.run(debug=True)
