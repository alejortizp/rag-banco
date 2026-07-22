from src.config import get_settings
from src.rag.retriever import Retriever
from src.rag.llm import BaseLLM
from src.memory.history import ConversationRepository

SYSTEM_PROMPT = (
    "Eres un asistente del banco. Responde SOLO con base en el contexto proporcionado. "
    "Si la respuesta no está en el contexto, dilo honestamente. Responde en español."
)

class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: BaseLLM, history: ConversationRepository):
        self.retriever, self.llm, self.history = retriever, llm, history
        self.n = get_settings().history_window_n

    def answer(self, session_id: str, question: str) -> dict:
        hits = self.retriever.retrieve(question)
        context = "\n\n".join(f"[{h['title']}]({h['url']})\n{h['text']}" for h in hits)
        previous = self.history.get_last_n(session_id, self.n)
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + [{"role": m["role"], "content": m["content"]} for m in previous]
            + [{"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}]
        )
        try:
            answer = self.llm.chat(messages)
        except Exception as e:
            answer = f"Lo siento, ocurrió un error consultando el modelo: {e}"
        self.history.add_message(session_id, "user", question)
        self.history.add_message(session_id, "assistant", answer)
        return {"answer": answer, "sources": [h["url"] for h in hits]}
