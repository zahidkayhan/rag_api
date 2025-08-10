import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    sbert_available = True
except Exception:
    sbert_available = False
try:
    import faiss
    faiss_available = True
except Exception:
    faiss_available = False
from sklearn.metrics.pairwise import cosine_similarity

class SimpleEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.dim = 384
        if sbert_available:
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            self.model = None
    def encode(self, texts):
        if self.model:
            return self.model.encode(texts, show_progress_bar=False)
        return np.array([self._fake_embed(t) for t in texts])
    def _fake_embed(self, t):
        h = abs(hash(t)) % (10**8)
        vec = np.ones(self.dim) * (h % 1000) / 1000.0
        return vec

class EmbeddingStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.meta = []
        if faiss_available:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
    def add(self, vector, text, meta=None):
        v = np.array(vector).astype('float32').reshape(1, -1)
        if self.index is not None:
            self.index.add(v)
        self.meta.append({'text': text, 'meta': meta, 'vector': v})
    def search(self, query_or_vector, top_k=4):
        if isinstance(query_or_vector, str):
            emb = SimpleEmbedder().encode([query_or_vector])[0].astype('float32').reshape(1,-1)
        else:
            emb = np.array(query_or_vector).astype('float32').reshape(1,-1)
        results = []
        if self.index is not None:
            D, I = self.index.search(emb, top_k)
            for score, idx in zip(D[0], I[0]):
                if idx < len(self.meta):
                    m = self.meta[idx]
                    results.append({'score': float(score), 'text': m['text'], 'meta': m['meta']})
        else:
            vectors = np.vstack([m['vector'] for m in self.meta]) if self.meta else np.zeros((0,self.dim))
            if vectors.size == 0:
                return []
            sims = cosine_similarity(emb, vectors)[0]
            idxs = sims.argsort()[::-1][:top_k]
            for i in idxs:
                results.append({'score': float(sims[i]), 'text': self.meta[i]['text'], 'meta': self.meta[i]['meta']})
        return results

def get_default_embedder():
    return SimpleEmbedder()
