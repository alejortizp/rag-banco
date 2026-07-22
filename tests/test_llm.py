import pytest
from src.rag.llm import LLMFactory, GroqLLM, OllamaLLM, BaseLLM


def test_factory_rechaza_proveedor_desconocido(monkeypatch):
    """Factory debe rechazar proveedores no registrados."""
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "llm_provider", "desconocido")
    with pytest.raises(ValueError, match="desconocido"):
        LLMFactory.create()


def test_factory_registra_proveedores_esperados():
    """Factory debe tener registrados los proveedores groq y ollama."""
    assert set(LLMFactory._providers) == {"groq", "ollama"}
    assert issubclass(GroqLLM, BaseLLM) and issubclass(OllamaLLM, BaseLLM)
