from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    """Patrón Strategy: algoritmos de chunking intercambiables."""
    @abstractmethod
    def chunk(self, doc: dict) -> list[dict]: ...

class FixedSizeChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if not 0 <= overlap < chunk_size:
            raise ValueError(f"overlap ({overlap}) debe ser >= 0 y menor que chunk_size ({chunk_size})")
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
            # If single paragraph exceeds max_chars, sub-split by words
            if len(p) > self.max_chars:
                # Flush current accumulation first
                if current.strip():
                    chunks.append({"text": current.strip(), "url": doc["url"], "title": doc["title"]})
                    current = ""
                # Split oversized paragraph by words into max_chars-bounded chunks
                words = p.split()
                word_chunk = ""
                for word in words:
                    if len(word_chunk) + len(word) + 1 > self.max_chars and word_chunk:
                        # Emit current word_chunk
                        chunks.append({"text": word_chunk.strip(), "url": doc["url"], "title": doc["title"]})
                        word_chunk = word
                    else:
                        if word_chunk:
                            word_chunk += " " + word
                        else:
                            word_chunk = word
                # Emit remaining word_chunk (note: single word > max_chars will be emitted alone)
                if word_chunk:
                    chunks.append({"text": word_chunk, "url": doc["url"], "title": doc["title"]})
            else:
                # Normal paragraph accumulation logic
                if len(current) + len(p) > self.max_chars and current:
                    chunks.append({"text": current.strip(), "url": doc["url"], "title": doc["title"]})
                    current = ""
                current += p + "\n"
        if current.strip():
            chunks.append({"text": current.strip(), "url": doc["url"], "title": doc["title"]})
        return chunks
