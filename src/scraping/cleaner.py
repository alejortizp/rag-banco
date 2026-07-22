import json
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

class HtmlCleaner:
    REMOVE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "iframe"]

    def clean_html(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else ""
        for tag in soup(self.REMOVE_TAGS):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return {"url": url, "title": title, "text": text}

    def clean_all(self) -> list[dict]:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        index = json.loads((RAW_DIR / "index.json").read_text())
        docs = []
        for entry in index:
            html = Path(entry["path"]).read_text(encoding="utf-8")
            doc = self.clean_html(html, entry["url"])
            if len(doc["text"]) > 100:  # descartar páginas vacías
                docs.append(doc)
        out = PROCESSED_DIR / "documents.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return docs
