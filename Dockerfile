FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /usr/local/bin/
WORKDIR /app

# uv instala las dependencias del lockfile directamente en el Python del sistema
# (sin venv dentro del contenedor): mismas versiones exactas que en desarrollo.
ENV UV_PROJECT_ENVIRONMENT=/usr/local
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Cachear modelos HF bajo /app para que sigan siendo legibles tras cambiar a usuario no-root
ENV HF_HOME=/app/.hf_cache
# pre-descargar modelos para que el contenedor arranque sin red al modelo
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser
# streamlit añade al path el dir del script (src/), no el del proyecto: sin esto falla `import src.*`
ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
