from abc import ABC, abstractmethod
from src.config import get_settings


class BaseLLM(ABC):
    """Clase base abstracta para clientes LLM."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Envía mensajes en formato OpenAI y retorna la respuesta.

        Args:
            messages: Lista de dicts con formato {"role": "system"|"user"|"assistant", "content": str}

        Returns:
            str: Contenido de la respuesta del modelo
        """
        ...


class GroqLLM(BaseLLM):
    """Cliente LLM usando Groq API."""

    def __init__(self):
        from groq import Groq
        s = get_settings()
        self.client = Groq(api_key=s.groq_api_key)
        self.model = s.llm_model

    def chat(self, messages: list[dict]) -> str:
        """Envía mensajes a Groq API."""
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content


class OpenAILLM(BaseLLM):
    """Cliente LLM usando OpenAI API."""

    def __init__(self):
        from openai import OpenAI
        s = get_settings()
        self.client = OpenAI(api_key=s.openai_api_key)
        self.model = s.llm_model

    def chat(self, messages: list[dict]) -> str:
        """Envía mensajes a OpenAI API."""
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content


class OllamaLLM(BaseLLM):
    """Cliente LLM usando Ollama local."""

    def __init__(self):
        import requests
        s = get_settings()
        self.url = "http://ollama:11434/api/chat"
        self.model = s.llm_model
        self._requests = requests

    def chat(self, messages: list[dict]) -> str:
        """Envía mensajes a servidor Ollama local."""
        resp = self._requests.post(
            self.url,
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class LLMFactory:
    """Patrón Factory Method: crea el proveedor LLM según configuración."""

    _providers = {"groq": GroqLLM, "openai": OpenAILLM, "ollama": OllamaLLM}

    @classmethod
    def create(cls) -> BaseLLM:
        """Crea una instancia del cliente LLM configurado.

        Returns:
            BaseLLM: Instancia del cliente LLM especificado en settings

        Raises:
            ValueError: Si el proveedor no está registrado
        """
        provider = get_settings().llm_provider
        if provider not in cls._providers:
            raise ValueError(
                f"Proveedor LLM desconocido: {provider}. "
                f"Opciones: {list(cls._providers)}"
            )
        return cls._providers[provider]()
