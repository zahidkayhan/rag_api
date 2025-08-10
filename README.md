# RAG API - FastAPI Project Scaffold

**What this contains**
- A minimal, well-documented FastAPI scaffold that implements:
  - `/upload` endpoint to accept files (pdf, docx, txt, images, csv, sqlite .db)
  - `/query` endpoint to ask questions (optionally with image base64)
  - Modular code layout for ingestion, embeddings, OCR, and query handling
- Example `.env.example` and `requirements.txt`
- Instructions to run and extend

**Important**
- This scaffold uses `sentence-transformers` and `faiss` if available.
- If those are not installed, the code falls back to a simple in-memory cosine-similarity vector search.
- For OCR it expects `pytesseract` (and Tesseract engine installed on the system) and `Pillow` for image handling.

**Set your OpenAI Api Key in .env File.**
	Go to OpenAI API Keys page and log in.
    Click "Create new secret key"
	
## Quick Start (Development)

1. Create a virtual environment and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
