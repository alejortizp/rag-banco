from src.scraping.cleaner import HtmlCleaner

def test_extrae_texto_sin_scripts_ni_estilos():
    html = "<html><head><title>Mi Banco</title><style>x{}</style></head><body><script>var a=1;</script><p>Hola cliente</p><nav>menu</nav></body></html>"
    doc = HtmlCleaner().clean_html(html, url="http://x.co")
    assert "Hola cliente" in doc["text"]
    assert "var a=1" not in doc["text"]
    assert doc["title"] == "Mi Banco"
