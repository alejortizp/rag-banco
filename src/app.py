from datetime import datetime

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

pipeline = get_pipeline()

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"


def _cambiar_sesion(sid: str) -> None:
    st.session_state.session_id = sid


def _eliminar_sesion(sid: str) -> None:
    get_pipeline().history.delete_session(sid)
    if st.session_state.session_id == sid:
        st.session_state.session_id = "default"


st.sidebar.text_input("ID de sesión", key="session_id")
st.sidebar.button(
    "➕ Nueva sesión",
    on_click=_cambiar_sesion,
    args=(f"sesion-{datetime.now().strftime('%Y%m%d-%H%M%S')}",),
    use_container_width=True,
)

sesiones = pipeline.history.list_sessions()
if sesiones:
    st.sidebar.divider()
    st.sidebar.caption(f"Sesiones anteriores ({len(sesiones)})")
    for ses in sesiones:
        sid = ses["session_id"]
        activa = sid == st.session_state.session_id
        col_sesion, col_borrar = st.sidebar.columns([4, 1])
        col_sesion.button(
            f"{'▶ ' if activa else ''}{sid} · {ses['message_count']} msjs",
            key=f"ses-{sid}",
            on_click=_cambiar_sesion,
            args=(sid,),
            disabled=activa,
            use_container_width=True,
        )
        with col_borrar.popover("🗑", use_container_width=True):
            st.caption(f"¿Eliminar «{sid}» y sus {ses['message_count']} mensajes?")
            st.button(
                "Eliminar definitivamente",
                key=f"del-{sid}",
                on_click=_eliminar_sesion,
                args=(sid,),
                type="primary",
            )

session_id = st.session_state.session_id

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
