FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# Instalar torch CPU-only primero para evitar el paquete CUDA (mucho más pesado)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# pre-descargar modelos para que el contenedor arranque sin red al modelo
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
