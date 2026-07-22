from src.rag.retriever import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        # puntúa más alto el documento que contiene la query
        return [1.0 if p[0] in p[1] else 0.0 for p in pairs]


class FakeEmbedder:
    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.last_top_k = None

    def upsert(self, chunks, vectors):
        pass

    def search(self, vector, top_k):
        self.last_top_k = top_k
        return self.hits[:top_k]


def _hits(n):
    return [{"text": f"doc {i}", "url": f"http://x.co/{i}", "title": f"t{i}", "score": 1.0 - i * 0.1} for i in range(n)]


def test_reranker_ordena_por_relevancia():
    rr = CrossEncoderReranker(model=FakeCrossEncoder())
    hits = [{"text": "irrelevante"}, {"text": "tasas de interés hipotecario"}]
    out = rr.rerank("tasas de interés", hits, top_k=1)
    assert out[0]["text"] == "tasas de interés hipotecario"


def test_reranker_con_hits_vacios_devuelve_lista_vacia():
    rr = CrossEncoderReranker(model=FakeCrossEncoder())
    assert rr.rerank("cualquier query", [], top_k=3) == []


def test_retriever_busca_top_k_y_devuelve_rerank_top_k():
    from src.rag.retriever import Retriever
    store = FakeStore(_hits(8))
    r = Retriever(FakeEmbedder(), store, reranker=CrossEncoderReranker(model=FakeCrossEncoder()))
    out = r.retrieve("doc 5")
    assert store.last_top_k == 8          # pidió top_k al store
    assert len(out) == 3                  # devolvió rerank_top_k
    assert out[0]["text"] == "doc 5"      # el reranker (fake: 1.0 si query in text) subió el relevante


def test_retriever_sin_reranker_trunca_a_rerank_top_k():
    from src.rag.retriever import Retriever
    store = FakeStore(_hits(8))
    r = Retriever(FakeEmbedder(), store, reranker=None)
    out = r.retrieve("cualquier query")
    assert len(out) == 3
    assert out == _hits(8)[:3]            # orden del store preservado
