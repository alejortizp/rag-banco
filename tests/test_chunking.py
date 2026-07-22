from src.indexing.chunking import FixedSizeChunking

def test_chunks_respetan_tamano_y_overlap():
    doc = {"url": "u", "title": "t", "text": "palabra " * 300}
    chunks = FixedSizeChunking(chunk_size=100, overlap=20).chunk(doc)
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 100 for c in chunks)
    # overlap: el inicio del chunk 2 repite el final del chunk 1
    fin_1 = chunks[0]["text"].split()[-20:]
    inicio_2 = chunks[1]["text"].split()[:20]
    assert fin_1 == inicio_2
