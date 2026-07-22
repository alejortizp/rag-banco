import uuid
from abc import ABC, abstractmethod
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import get_settings

class VectorStore(ABC):
    """Patrón Adapter: interfaz propia sobre el cliente concreto de la vector DB."""
    @abstractmethod
    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None: ...
    @abstractmethod
    def search(self, vector: list[float], top_k: int) -> list[dict]: ...

class QdrantVectorStore(VectorStore):
    def __init__(self, dim: int):
        s = get_settings()
        self.client = QdrantClient(url=s.qdrant_url)
        self.collection = s.collection_name
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=v, payload=c)
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points)

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        hits = self.client.query_points(self.collection, query=vector, limit=top_k).points
        return [{**h.payload, "score": h.score} for h in hits]
