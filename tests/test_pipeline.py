from src.rag.pipeline import RAGPipeline
from src.memory.history import ConversationRepository

class FakeRetriever:
    def retrieve(self, q):
        return [{"text": "las tarjetas oro no tienen cuota", "url": "http://b.co/tarjetas", "title": "Tarjetas", "score": 0.9}]

class FakeLLM:
    def __init__(self):
        self.last_messages = None
    def chat(self, messages):
        self.last_messages = messages
        return "respuesta de prueba"

class FailingLLM:
    def chat(self, messages):
        raise RuntimeError("api caida")

def _pipeline(tmp_path, llm):
    return RAGPipeline(FakeRetriever(), llm, ConversationRepository(str(tmp_path / "h.db")))

def test_answer_devuelve_respuesta_y_fuentes(tmp_path):
    p = _pipeline(tmp_path, FakeLLM())
    out = p.answer("s1", "¿qué tarjetas hay?")
    assert out["answer"] == "respuesta de prueba"
    assert out["sources"] == ["http://b.co/tarjetas"]

def test_answer_persiste_pregunta_y_respuesta(tmp_path):
    p = _pipeline(tmp_path, FakeLLM())
    p.answer("s1", "hola")
    msgs = p.history.get_last_n("s1", 10)
    assert [m["role"] for m in msgs] == ["user", "assistant"]

def test_answer_incluye_historial_previo_en_los_mensajes(tmp_path):
    llm = FakeLLM()
    p = _pipeline(tmp_path, llm)
    p.answer("s1", "primera pregunta")
    p.answer("s1", "segunda pregunta")
    roles = [m["role"] for m in llm.last_messages]
    # system + (user, assistant de la 1ra ronda) + user actual
    assert roles == ["system", "user", "assistant", "user"]
    assert "primera pregunta" in llm.last_messages[1]["content"]

def test_error_del_llm_no_revienta_y_se_persiste(tmp_path):
    p = _pipeline(tmp_path, FailingLLM())
    out = p.answer("s1", "hola")
    assert "error" in out["answer"].lower()
    assert len(p.history.get_last_n("s1", 10)) == 2
