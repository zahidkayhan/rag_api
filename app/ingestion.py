import io, os, tempfile
from typing import List
from .ocr import ocr_image_bytes
import pdfplumber
import docx
from PIL import Image
import pandas as pd
import sqlite3

def parse_file(filename: str, content: bytes) -> List[str]:
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        return _parse_pdf(content)
    elif ext in ('docx', 'doc'):
        return _parse_docx(content)
    elif ext in ('jpg', 'jpeg', 'png', 'gif', 'bmp'):
        return [ocr_image_bytes(content)]
    elif ext == 'txt':
        return [content.decode('utf-8', errors='ignore')]
    elif ext == 'csv':
        df = pd.read_csv(io.BytesIO(content))
        return [df.to_csv(index=False)]
    elif ext in ('db', 'sqlite'):
        p = _save_tempfile(content, suffix='.db')
        return _parse_sqlite(p)
    else:
        raise ValueError(f'Unsupported file type: {ext}')

def _save_tempfile(content: bytes, suffix=''):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, content)
    os.close(fd)
    return path

def _parse_pdf(content: bytes):
    texts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ''
            if not text.strip():
                try:
                    x0 = p.to_image(resolution=150).original
                    texts.append(ocr_image_bytes(x0))
                except Exception:
                    texts.append('')
            else:
                texts.append(text)
    return texts

def _parse_docx(content: bytes):
    p = _save_tempfile(content, suffix='.docx')
    doc = docx.Document(p)
    texts = ['\n'.join([para.text for para in doc.paragraphs])]
    os.remove(p)
    return texts

def _parse_sqlite(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    outputs = []
    for t in tables:
        name = t[0]
        df = pd.read_sql_query(f'SELECT * FROM \"{name}\" LIMIT 1000', conn)
        outputs.append(df.to_csv(index=False))
    conn.close()
    return outputs
