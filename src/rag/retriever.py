from sentence_transformers import CrossEncoder
from src.config import get_settings
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore


class CrossEncoderReranker:
    def __init__(self, model=None):
        self.model = model or CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if not hits:
            return []
        scores = self.model.predict([(query, h["text"]) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:top_k]]


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, reranker: CrossEncoderReranker | None = None):
        self.embedder, self.store = embedder, store
        self.reranker = reranker
        self.settings = get_settings()

    def retrieve(self, query: str) -> list[dict]:
        vector = self.embedder.encode([query])[0]
        hits = self.store.search(vector, self.settings.top_k)
        if self.reranker:
            return self.reranker.rerank(query, hits, self.settings.rerank_top_k)
        return hits[: self.settings.rerank_top_k]
