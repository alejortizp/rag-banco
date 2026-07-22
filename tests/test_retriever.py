from src.rag.retriever import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        # puntúa más alto el documento que contiene la query
        return [1.0 if p[0] in p[1] else 0.0 for p in pairs]


def test_reranker_ordena_por_relevancia():
    rr = CrossEncoderReranker(model=FakeCrossEncoder())
    hits = [{"text": "irrelevante"}, {"text": "tasas de interés hipotecario"}]
    out = rr.rerank("tasas de interés", hits, top_k=1)
    assert out[0]["text"] == "tasas de interés hipotecario"
