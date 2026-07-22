import json
from pathlib import Path
from src.config import get_settings
from src.indexing.chunking import FixedSizeChunking
from src.indexing.embedder import Embedder
from src.indexing.vector_store import QdrantVectorStore

if __name__ == "__main__":
    s = get_settings()
    docs = [json.loads(l) for l in Path("data/processed/documents.jsonl").read_text(encoding="utf-8").splitlines()]
    strategy = FixedSizeChunking(s.chunk_size, s.chunk_overlap)
    chunks = [c for d in docs for c in strategy.chunk(d)]
    embedder = Embedder()
    store = QdrantVectorStore(dim=embedder.dim)
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        store.upsert(batch, embedder.encode([c["text"] for c in batch]))
    print(f"Indexados {len(chunks)} chunks de {len(docs)} documentos")
