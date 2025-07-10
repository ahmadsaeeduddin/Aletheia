import os
import hashlib
import json
from urllib.parse import urlparse
from scraper2 import ContentScraper
from fact_check import FactChecker

# 📁 Ensure folder exists
os.makedirs("knowledge_base", exist_ok=True)

def clean_filename(url: str) -> str:
    """Create a safe filename using domain + short hash"""
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "_")
    short_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{domain}_{short_hash}.json"

def save_json(data: dict, filename: str):
    path = os.path.join("knowledge_base", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {path}")

def build_knowledge_base(claim_text: str, urls: list):
    print(f"\n🧠 Building knowledge base for claim: \"{claim_text}\"\n")

    fc = FactChecker()
    scraper = ContentScraper()

    for url in urls:
        try:
            print(f"🔍 Processing: {url}")
            if "snopes.com" in url:
                data = fc.scrape_snopes(url)
            elif "politifact.com" in url:
                data = fc.scrape_politifact(url)
            else:
                data = scraper.scrape_content(url)

            filename = clean_filename(url)
            save_json(data, filename)

        except Exception as e:
            print(f"❌ Failed to process {url}: {e}")
