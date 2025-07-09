import time
from scraper_2 import ContentScraper
from text_cleaner import TextCleaner
from claim_extraction import ClaimExtractor

def main():
    # Step 1: Scrape content
    url = "https://www.arabnews.com/node/2607266/middle-east"
    scraper = ContentScraper(url)
    result = scraper.scrape_content(url)
    
    scraped_text = result['title'] + " " + result['text']
    print(f"Original text length: {len(scraped_text)} characters")
    print("=" * 60)
    
    # Step 2: Clean and normalize text
    cleaner = TextCleaner()
    clean_text = cleaner.clean_text(scraped_text)
    print(f"Cleaned text length: {len(clean_text)} characters")
    print("Sample cleaned text:")
    print(clean_text[:500] + "...")
    print("=" * 60)
    
    # Step 3: Vectorize text (for potential ML use)
    tfidf_vectors = cleaner.vectorize_text([clean_text])
    print(f"TF-IDF vectors shape: {tfidf_vectors.shape}")
    print("=" * 60)
    
    # Step 4: Extract claims
    extractor = ClaimExtractor()
    claims = extractor.process_article(scraped_text)  # Using original text for better entity recognition
    
    print("\nExtracted Claims:")
    for i, claim in enumerate(claims, 1):
        print(f"\nClaim {i}:")
        print(f"Sentence: {claim['sentence']}")
        print(f"Sentiment: {claim['sentiment']:.2f}")
        
        print("\nEntities:")
        for entity in claim['entities']:
            print(f"- {entity['text']} ({entity['label']})")
        
        print("\nTriples:")
        for triple in claim['triples']:
            print(f"- Subject: {triple['subject']}")
            print(f"  Predicate: {triple['relation']}")
            print(f"  Object: {triple['object']}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()