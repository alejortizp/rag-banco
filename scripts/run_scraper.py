from src.config import get_settings
from src.scraping.scraper import WebScraper
from src.scraping.cleaner import HtmlCleaner

if __name__ == "__main__":
    s = get_settings()
    pages = WebScraper(s.scrape_base_url, s.scrape_max_pages).crawl()
    print(f"Scrapeadas {len(pages)} páginas en data/raw/")
    docs = HtmlCleaner().clean_all()
    print(f"Limpiados {len(docs)} documentos en data/processed/")
