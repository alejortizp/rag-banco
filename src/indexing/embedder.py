from sentence_transformers import SentenceTransformer
from src.config import get_settings

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(get_settings().embedding_model)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
