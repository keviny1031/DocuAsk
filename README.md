# DocuAsk

DocuAsk is a simple web application that allows you to upload a PDF document and ask questions about its content. The PDF is broken into chunks, embedded with a sentence-transformer model and indexed with the FAISS vector database. When you ask a question, the most relevant text chunks are retrieved and sent to a locally running LLM via [Ollama](https://github.com/ollama/ollama) to generate a concise answer.

## Features

- Upload a PDF from the browser.
- Question form with asynchronous requests.
- Retrieves top document chunks using vector similarity search.
- Uses a local Llama 3 model via the Ollama API for answers.

## Tech Stack

- **Python & Flask** – web framework and API routes (`app.py`).
- **Sentence Transformers** – generates embeddings (`all-MiniLM-L6-v2`).
- **Ollama LLM** – runs the LLM (`llama3`) locally for question answering.
- **FAISS Vector Database** – in-memory vector index for similarity search.
- **PyMuPDF** via `langchain_community.document_loaders.PyMuPDFLoader` – reads PDF files.
- **HTML/CSS/JS** – simple front‑end located in `templates/` and `static/`.

## Getting Started

1. Install dependencies (example using `pip`):
   ```bash
   pip install flask langchain-community sentence-transformers faiss-cpu numpy requests PyMuPDF
   ```
2. Ensure an Ollama server is running on `http://localhost:11434` with the `llama3` model available.
3. Start the Flask app:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser to upload a PDF and begin asking questions.

The uploaded file is temporarily saved in the `uploads/` directory and removed once the index is built. The example front end shows the upload status, allows you to type a question, and displays the model's answer.

## Repository Structure

```
app.py          - main Flask application
templates/      - HTML template(s)
static/         - CSS for the web UI
README.md       - project documentation
```
