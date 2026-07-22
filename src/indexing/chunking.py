from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    """Patrón Strategy: algoritmos de chunking intercambiables."""
    @abstractmethod
    def chunk(self, doc: dict) -> list[dict]: ...

class FixedSizeChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size, self.overlap = chunk_size, overlap

    def chunk(self, doc: dict) -> list[dict]:
        words = doc["text"].split()
        step = self.chunk_size - self.overlap
        chunks = []
        for i in range(0, max(len(words) - self.overlap, 1), step):
            text = " ".join(words[i:i + self.chunk_size])
            if text.strip():
                chunks.append({"text": text, "url": doc["url"], "title": doc["title"]})
        return chunks

class ParagraphChunking(ChunkingStrategy):
    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars

    def chunk(self, doc: dict) -> list[dict]:
        paras, current, chunks = doc["text"].split("\n"), "", []
        for p in paras:
            if len(current) + len(p) > self.max_chars and current:
                chunks.append({"text": current.strip(), "url": doc["url"], "title": doc["title"]})
                current = ""
            current += p + "\n"
        if current.strip():
            chunks.append({"text": current.strip(), "url": doc["url"], "title": doc["title"]})
        return chunks
