import os
from dotenv import load_dotenv
from scraper2 import ContentScraper
from groq_claim import GroqClaimGenerator
from fact_check import FactChecker
from scraper import google_search, duckduckgo_search
from build_knowledge_base import KnowledgeBaseBuilder


# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY_4")

def step_1_scrape_article(url: str, save_file="data.json"):
    print(f"\n🔍 Scraping article: {url}")
    scraper = ContentScraper()
    result = scraper.scrape_content(url)
    scraper.save_to_json(result, save_file)
    return result

def step_2_generate_claims(text: str):
    print("\n✍️ Generating claims...")
    claim_gen = GroqClaimGenerator(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    claims = claim_gen.generate_claims_from_text(text)
    filtered = claim_gen.filter_similar_claims(claims)
    ranked = claim_gen.score_claims_nlp(filtered)
    return ranked

def step_3_search_claim(claim_text: str, num_results=5):
    print(f"\n🌐 Searching evidence for claim: \"{claim_text}\"")
    google_results = google_search(claim_text, num_results)
    duck_snopes = duckduckgo_search(claim_text, site="snopes.com")
    duck_politifact = duckduckgo_search(claim_text, site="politifact.com")
    return google_results + duck_snopes + duck_politifact

def step_4_check_fact(url: str):
    fc = FactChecker()
    if "snopes.com" in url:
        return fc.scrape_snopes(url)
    elif "politifact.com" in url:
        return fc.scrape_politifact(url)
    else:
        # fallback
        print("ℹ️ Unknown fact site. Falling back to article scraping.")
        content_scraper = ContentScraper()
        return content_scraper.scrape_content(url)

def load_text_from_json(path="data.json"):
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    title = data.get("title", "")
    text = data.get("text", "")
    return f"{title}. {text}" if title and title not in text else text

def main():
    # Step 1: Scrape the main article
    input_url = input("🖋️ Enter the article URL: ").strip()
    article_data = step_1_scrape_article(input_url)
    
    # Step 2: Generate top claims from the article
    article_text = load_text_from_json("data.json")
    ranked_claims = step_2_generate_claims(article_text)

    if not ranked_claims:
        print("🚫 No valid claims found.")
        return

    print("\n🏆 Top Claims:")
    for i, (claim, score) in enumerate(ranked_claims[:3], 1):
        print(f"{i}. (Score: {score:.2f}) {claim}")

    # Ask user to pick a claim to validate
    choice = input("\nChoose a claim to fact-check (1/2/3): ").strip()
    try:
        idx = int(choice) - 1
        selected_claim = ranked_claims[idx][0]
    except:
        print("⚠️ Invalid choice. Exiting.")
        return

    # Step 3: Search for related evidence
    evidence_urls = step_3_search_claim(selected_claim)

    if not evidence_urls:
        print("❌ No evidence URLs found.")
        return

    print("\n🔗 Evidence URLs Found:")
    for i, url in enumerate(evidence_urls, 1):
        print(f"{i}. {url}")

    # Step 4: Build knowledge base for all related URLs
    kb_builder = KnowledgeBaseBuilder()
    kb_builder.build(selected_claim, evidence_urls)


if __name__ == "__main__":
    main()
