import streamlit as st
from src.config import get_settings
from src.indexing.embedder import Embedder
from src.indexing.vector_store import QdrantVectorStore
from src.rag.retriever import Retriever, CrossEncoderReranker
from src.rag.llm import LLMFactory
from src.rag.pipeline import RAGPipeline
from src.memory.history import ConversationRepository

st.set_page_config(page_title="Asistente RAG Banco", page_icon="🏦")
st.title("🏦 Asistente del sitio del banco")

@st.cache_resource
def get_pipeline() -> RAGPipeline:
    s = get_settings()
    embedder = Embedder()
    store = QdrantVectorStore(dim=embedder.dim)
    retriever = Retriever(embedder, store, CrossEncoderReranker())
    return RAGPipeline(retriever, LLMFactory.create(), ConversationRepository(s.db_path))

session_id = st.sidebar.text_input("ID de sesión", value="default")
pipeline = get_pipeline()

for m in pipeline.history.get_last_n(session_id, n=50):
    st.chat_message(m["role"]).write(m["content"])

if question := st.chat_input("Pregunta sobre el sitio del banco..."):
    st.chat_message("user").write(question)
    with st.spinner("Buscando..."):
        result = pipeline.answer(session_id, question)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        with st.expander("Fuentes"):
            for url in result["sources"]:
                st.write(url)
