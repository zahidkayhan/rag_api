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
	Copy and paster the key on .env file replacing XXXXX, OPENAI_API_KEY=YOUR_APIKEY
	
## Quick Start (Development)

1. Create a virtual environment and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Run API on a Virtual Environment
```bash
.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
## Local Host API Endpoint

**API URL**
	http://127.0.0.1:8000/docs#/
	

# API I/O format

**Upload API**
- On Swagger Tryout the Upload api
- Select any pdf, docx, txt, images, csv, sqlite or .db file and execute.
- On a successful upload API will show Chunk size.

**Query API**
- After Uploading the file send your Query under question.
- set image _base64 value to "", if you have no image attached with your query.

- On Success OpneAPI should provide the respose.