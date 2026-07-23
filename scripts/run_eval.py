"""Evaluación RAGAS del pipeline RAG (offline).

Sistema bajo prueba: el pipeline real (retriever + reranker + LLM configurado en .env).
LLM juez: gpt-4o-mini (OpenAI) — independiente del proveedor bajo prueba.

Métricas sin ground truth:
  - faithfulness:        ¿la respuesta se sostiene solo en el contexto recuperado?
  - answer_relevancy:    ¿la respuesta atiende la pregunta?
  - context_utilization: ¿los chunks relevantes quedaron arriba del contexto?

Uso:  uv run --group eval python -m scripts.run_eval
Requiere: OPENAI_API_KEY en .env (solo para el juez) y Qdrant corriendo con datos indexados.
"""
import json
import sys
from pathlib import Path

from src.config import get_settings
from src.indexing.embedder import Embedder
from src.indexing.vector_store import QdrantVectorStore
from src.rag.llm import LLMFactory
from src.rag.pipeline import SYSTEM_PROMPT
from src.rag.retriever import CrossEncoderReranker, Retriever

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
RESULTS_PATH = Path("data/eval_results.json")
# Juez configurable: EVAL_JUDGE=openai (gpt-4o-mini, requiere crédito) | groq (free tier)
JUDGES = {
    "openai": {"model": "gpt-4o-mini", "base_url": None},
    "groq": {"model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
}


def build_answer(retriever, llm, question: str) -> dict:
    """Replica la generación del RAGPipeline sin escribir en el historial de conversaciones."""
    hits = retriever.retrieve(question)
    context = "\n\n".join(f"[{h['title']}]({h['url']})\n{h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"},
    ]
    return {"answer": llm.chat(messages), "contexts": [h["text"] for h in hits]}


def main() -> int:
    import os

    s = get_settings()
    judge_name = os.environ.get("EVAL_JUDGE", "openai")
    judge = JUDGES[judge_name]
    judge_key = s.openai_api_key if judge_name == "openai" else s.groq_api_key
    if not judge_key:
        print(f"ERROR: falta la API key para el juez '{judge_name}' en .env.")
        return 1

    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, context_utilization, faithfulness
    from ragas.run_config import RunConfig

    # strictness=1 → una sola completion por llamada (la API de Groq no soporta n>1)
    answer_relevancy = AnswerRelevancy(strictness=1)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    embedder = Embedder()
    retriever = Retriever(embedder, QdrantVectorStore(dim=embedder.dim), CrossEncoderReranker())
    llm = LLMFactory.create()

    print(f"Generando respuestas para {len(questions)} preguntas (LLM bajo prueba: {s.llm_provider}/{s.llm_model})...")
    rows = {"question": [], "answer": [], "contexts": []}
    for i, q in enumerate(questions, 1):
        result = build_answer(retriever, llm, q)
        rows["question"].append(q)
        rows["answer"].append(result["answer"])
        rows["contexts"].append(result["contexts"])
        print(f"  [{i}/{len(questions)}] {q}")

    print(f"\nEvaluando con RAGAS (juez: {judge_name}/{judge['model']})...")
    scores = evaluate(
        Dataset.from_dict(rows),
        metrics=[faithfulness, answer_relevancy, context_utilization],
        llm=ChatOpenAI(model=judge["model"], api_key=judge_key, base_url=judge["base_url"], temperature=0),
        # Embeddings locales (answer_relevancy): sin dependencia de cuota de APIs
        embeddings=HuggingFaceEmbeddings(model_name=s.embedding_model),
        # max_workers bajo: respeta los rate limits del free tier del juez
        run_config=RunConfig(max_workers=1, max_retries=10, timeout=120),
    )

    df = scores.to_pandas()
    summary = {
        "llm_bajo_prueba": f"{s.llm_provider}/{s.llm_model}",
        "juez": f"{judge_name}/{judge['model']}",
        "n_preguntas": len(questions),
        "faithfulness": round(float(df["faithfulness"].mean()), 3),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 3),
        "context_utilization": round(float(df["context_utilization"].mean()), 3),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail = df[["question", "faithfulness", "answer_relevancy", "context_utilization"]].to_dict(orient="records")
    RESULTS_PATH.write_text(
        json.dumps({"resumen": summary, "detalle": detail}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Resumen ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetalle por pregunta guardado en {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
