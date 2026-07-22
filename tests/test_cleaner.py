from src.scraping.cleaner import HtmlCleaner

def test_extrae_texto_sin_scripts_ni_estilos():
    html = "<html><head><title>Mi Banco</title><style>x{}</style></head><body><script>var a=1;</script><p>Hola cliente</p><nav>menu</nav></body></html>"
    doc = HtmlCleaner().clean_html(html, url="http://x.co")
    assert "Hola cliente" in doc["text"]
    assert "var a=1" not in doc["text"]
    assert doc["title"] == "Mi Banco"

def test_clean_all_sin_index_devuelve_lista_vacia(tmp_path, monkeypatch):
    import src.scraping.cleaner as cleaner_mod
    monkeypatch.setattr(cleaner_mod, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cleaner_mod, "PROCESSED_DIR", tmp_path / "processed")
    assert cleaner_mod.HtmlCleaner().clean_all() == []

def test_clean_all_ignora_archivos_faltantes(tmp_path, monkeypatch):
    import json
    import src.scraping.cleaner as cleaner_mod
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "ok.html").write_text("<html><title>T</title><body><p>" + "contenido util " * 20 + "</p></body></html>")
    index = [
        {"url": "http://x.co/ok", "path": str(raw / "ok.html"), "fetched_at": 1.0},
        {"url": "http://x.co/gone", "path": str(raw / "gone.html"), "fetched_at": 2.0},
    ]
    (raw / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(cleaner_mod, "RAW_DIR", raw)
    monkeypatch.setattr(cleaner_mod, "PROCESSED_DIR", tmp_path / "processed")
    docs = cleaner_mod.HtmlCleaner().clean_all()
    assert len(docs) == 1 and docs[0]["url"] == "http://x.co/ok"

def test_elimina_contenido_oculto():
    from src.scraping.cleaner import HtmlCleaner
    html = '<html><body><p>visible</p><div style="display: none">{}</div><span hidden>secreto</span></body></html>'
    doc = HtmlCleaner().clean_html(html, url="http://x.co")
    assert "visible" in doc["text"] and "{}" not in doc["text"] and "secreto" not in doc["text"]
