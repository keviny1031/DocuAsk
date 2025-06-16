from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import requests

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
id_to_doc = {}

def query_ollama(prompt, model_name="llama3"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def build_index(filepath):
    global index, id_to_doc
    loader = PyMuPDFLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    texts = [doc.page_content for doc in split_docs]
    embeddings = model.encode(texts)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings) # type: ignore

    id_to_doc = {i: doc for i, doc in enumerate(split_docs)}

@app.route("/", methods=["GET"])
def home():
    filename = request.args.get("filename")
    return render_template("index.html", filename=filename)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["pdf"]
    if file.filename == "":
        return redirect(url_for("home"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], str(file.filename))
    file.save(filepath)

    build_index(filepath)
    os.remove(filepath)
    return redirect(url_for("home", filename=file.filename))

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "")
    if not question or index is None:
        return jsonify({"answer": "Please upload a PDF and enter a valid question."})

    query_vec = model.encode([question])
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)

    D, I = index.search(np.array(query_vec), k=5) # type: ignore
    top_chunks = [id_to_doc[i].page_content for i in I[0]]

    context = "\n\n".join(top_chunks)
    prompt = f"Answer the question given the following context:\n\n{context}\n\nQuestion: {question}"
    answer = query_ollama(prompt)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
    