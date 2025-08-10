from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64, os
from dotenv import load_dotenv

from .ingestion import parse_file, ocr_image_bytes
from .embeddings import EmbeddingStore, get_default_embedder
from .utils import chunk_text

# Load environment variables from .env
load_dotenv()

# OpenAI imports
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title='RAG FastAPI Scaffold')

# Initialize embedder + vector store
embedder = get_default_embedder()
store = EmbeddingStore(dim=embedder.dim)

class QueryRequest(BaseModel):
    question: str
    image_base64: str = None
    top_k: int = 4

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        texts = parse_file(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to parse file: {e}')

    chunks = []
    for i, t in enumerate(texts):
        chunks.extend(chunk_text(t, chunk_size=500, overlap=50))

    embeddings = embedder.encode(chunks)
    for i, vec in enumerate(embeddings):
        meta = {'filename': file.filename, 'chunk_index': i}
        store.add(vec, chunks[i], meta)

    return JSONResponse({'status': 'success', 'chunks_added': len(embeddings)})

@app.post('/query')
async def query(req: QueryRequest):
    # Optional OCR from image
    if req.image_base64:
        try:
            img_bytes = base64.b64decode(req.image_base64)
            ocr_text = ocr_image_bytes(img_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'OCR failed: {e}')
    else:
        ocr_text = None

    # Search vector store for relevant chunks
    results = store.search(req.question, top_k=req.top_k)
    context = '\n\n'.join([r['text'] for r in results])

    # Build RAG prompt
    prompt = f"""You are an AI assistant. Use the context below to answer the question accurately and concisely.
Context:
{context}

Question: {req.question}
"""
    if ocr_text:
        prompt += f"\nOCR extracted text:\n{ocr_text}"

    # Call OpenAI GPT model
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # You can change to gpt-4 or gpt-4o
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'LLM request failed: {e}')

    return JSONResponse({
        'question': req.question,
        'context': context,
        'answer': answer,
        'sources': [r['meta'] for r in results]
    })
