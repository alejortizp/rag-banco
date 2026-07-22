import json
import logging
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

logger = logging.getLogger(__name__)

class HtmlCleaner:
    REMOVE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "iframe"]

    def clean_html(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else ""
        for tag in soup(self.REMOVE_TAGS):
            tag.decompose()
        # Remove hidden DOM elements
        for tag in soup.find_all(style=True):
            # find_all() snapshots matches before this loop runs. When a
            # match contains a nested match (e.g. a hidden <div> wrapping
            # another hidden <div>, as seen on real Bancolombia pages),
            # decomposing the outer tag recursively clears the inner tag's
            # __dict__ too (bs4 Tag.decompose()), leaving tag.attrs == None
            # for that already-processed-by-cascade descendant. Skip it.
            if tag.decomposed or not tag.attrs:
                continue
            style = tag.attrs.get("style")
            if isinstance(style, str) and "display:none" in style.replace(" ", ""):
                tag.decompose()
        for tag in soup.find_all(hidden=True):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return {"url": url, "title": title, "text": text}

    def clean_all(self) -> list[dict]:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        index_path = RAW_DIR / "index.json"
        if not index_path.exists():
            logger.warning(f"Index file not found: {index_path}")
            return []
        index = json.loads(index_path.read_text())
        docs = []
        for entry in index:
            try:
                html = Path(entry["path"]).read_text(encoding="utf-8")
                doc = self.clean_html(html, entry["url"])
                if len(doc["text"]) > 100:  # descartar páginas vacías
                    docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to process {entry.get('path', '?')}: {e}")
                continue
        out = PROCESSED_DIR / "documents.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return docs
