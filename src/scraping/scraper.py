import hashlib, json, logging, time
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlsplit
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")

BINARY_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip",
    ".doc", ".docx", ".xls", ".xlsx", ".mp4", ".webp", ".ico",
    ".css", ".js",
)

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """Strip fragment, querystring, and a single trailing slash (except bare root)."""
    parts = urlsplit(url)
    parts = parts._replace(query="", fragment="")
    path = parts.path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    parts = parts._replace(path=path)
    return parts.geturl()


class WebScraper:
    def __init__(self, base_url: str, max_pages: int = 50, delay: float = 1.0):
        self.base_url = _normalize_url(base_url)
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

    def _is_binary_path(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return path.endswith(BINARY_EXTENSIONS)

    def crawl(self) -> list[Path]:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        index_path = RAW_DIR / "index.json"
        queue, seen, index, saved = [self.base_url], {self.base_url}, [], []
        headers = {"User-Agent": "Mozilla/5.0 (prueba-tecnica-rag)"}
        while queue and len(saved) < self.max_pages:
            url = queue.pop(0)
            if not self._allowed(url):
                seen.add(url)
                continue

            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue

                name = hashlib.sha1(url.encode()).hexdigest()[:16] + ".html"
                path = RAW_DIR / name
                path.write_text(resp.text, encoding="utf-8")
                saved.append(path)
                index.append({"url": url, "path": str(path), "fetched_at": time.time()})
                index_path.write_text(json.dumps(index, indent=2))

                soup = BeautifulSoup(resp.text, "lxml")
                for a in soup.find_all("a", href=True):
                    nxt = _normalize_url(urljoin(url, a["href"]))
                    if self._is_binary_path(nxt):
                        continue
                    if urlparse(nxt).netloc == self.domain and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            except requests.RequestException as e:
                logger.warning("fallo de red en %s: %s", url, e)
                continue
            except Exception as e:
                logger.warning("fallo al procesar/parsear %s: %s", url, e)
                continue
            finally:
                time.sleep(self.delay)

        index_path.write_text(json.dumps(index, indent=2))
        return saved
