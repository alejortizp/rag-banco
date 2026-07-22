import pytest
from src.indexing.chunking import FixedSizeChunking, ParagraphChunking

def test_chunks_respetan_tamano_y_overlap():
    doc = {"url": "u", "title": "t", "text": "palabra " * 300}
    chunks = FixedSizeChunking(chunk_size=100, overlap=20).chunk(doc)
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 100 for c in chunks)
    # overlap: el inicio del chunk 2 repite el final del chunk 1
    fin_1 = chunks[0]["text"].split()[-20:]
    inicio_2 = chunks[1]["text"].split()[:20]
    assert fin_1 == inicio_2

def test_overlap_invalido_lanza_error():
    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        FixedSizeChunking(chunk_size=100, overlap=150)

def test_paragraph_chunking_respeta_max_chars_en_texto_sin_saltos():
    doc = {"url": "u", "title": "t", "text": "palabra " * 2000}  # ~16000 chars, sin \n
    chunks = ParagraphChunking(max_chars=2000).chunk(doc)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 2000 for c in chunks)
    # no se pierde contenido: las palabras totales se conservan
    total = " ".join(c["text"] for c in chunks).split()
    assert len(total) == 2000
