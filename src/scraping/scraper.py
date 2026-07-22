import hashlib, json, time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")

class WebScraper:
    def __init__(self, base_url: str, max_pages: int = 50, delay: float = 1.0):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.delay = delay
        self.robots = RobotFileParser(urljoin(base_url, "/robots.txt"))
        try:
            self.robots.read()
        except Exception:
            self.robots = None

    def _allowed(self, url: str) -> bool:
        return self.robots is None or self.robots.can_fetch("*", url)

    def crawl(self) -> list[Path]:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        queue, seen, index, saved = [self.base_url], set(), [], []
        headers = {"User-Agent": "Mozilla/5.0 (prueba-tecnica-rag)"}
        while queue and len(saved) < self.max_pages:
            url = queue.pop(0)
            if url in seen or not self._allowed(url):
                continue
            seen.add(url)
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"[WARN] fallo {url}: {e}")
                continue
            name = hashlib.sha1(url.encode()).hexdigest()[:16] + ".html"
            path = RAW_DIR / name
            path.write_text(resp.text, encoding="utf-8")
            saved.append(path)
            index.append({"url": url, "path": str(path), "fetched_at": time.time()})
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                nxt = urljoin(url, a["href"]).split("#")[0]
                if urlparse(nxt).netloc == self.domain and nxt not in seen:
                    queue.append(nxt)
            time.sleep(self.delay)
        (RAW_DIR / "index.json").write_text(json.dumps(index, indent=2))
        return saved
