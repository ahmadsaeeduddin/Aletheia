from generate_claim import Generate_claim
from scraper import ContentScraper
from fetching_related_URLs import Fetch

url = input('Enter The URL : ')
scraper = ContentScraper()

print(f"\nScraping: {url}")
result = scraper.scrape_content(url)

# Print summary
print(f"Platform: {result['platform']}")
print(f"Title: {result['title'][:100]}...")
print(f"Text length: {len(result['text'])} characters")
print(f"Author: {result['author']}")
print(f"Published: {result['published_date']}")
article_text = result['text']

# Claim Generation
article_claim = Generate_claim.convert_paragraph_to_claim(article_text)
print(f"\n📢 Extracted Claim:\n✅ {article_claim}")

#  Evidence Finding
related_urls = Fetch.search_duckduckgo_lite(article_claim)
print("\n🔗 Extracted URLs:")
for i, url in enumerate(related_urls, 1):
    print(f"{i}. {url}")