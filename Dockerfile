FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# Instalar torch CPU-only primero para evitar el paquete CUDA (mucho más pesado)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
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
