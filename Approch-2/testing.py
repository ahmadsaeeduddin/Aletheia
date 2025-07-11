import os
from dotenv import load_dotenv
from scraper2 import ContentScraper
from groq_claim import GroqClaimGenerator
from text_cleaner import TextCleaner
import json

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY_4")

def step_1_scrape_article(url: str, save_file="data.json"):
    print(f"\nScraping article: {url}")
    scraper = ContentScraper()
    result = scraper.scrape_content(url)
    scraper.save_to_json(result, save_file)
    return result

def step_2_generate_claims(text: str):
    print("\nGenerating claims from cleaned text...")
    claim_gen = GroqClaimGenerator(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    claims = claim_gen.generate_claims_from_text(text)
    filtered = claim_gen.filter_similar_claims(claims)
    ranked = claim_gen.score_claims_nlp(filtered)
    return ranked

def load_text_from_json(path="data.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    title = data.get("title", "")
    text = data.get("text", "")
    return f"{title}. {text}" if title and title not in text else text

def debug_clean_text_steps(text):
    cleaner = TextCleaner()
    print("\nOriginal Text:")
    print(text[:500], '...\n')

    text = cleaner._remove_html_tags(text)
    print("After Removing HTML Tags:")
    print(text[:500], '...\n')

    text = cleaner._remove_urls(text)
    print("After Removing URLs:")
    print(text[:500], '...\n')

    text = cleaner._remove_punctuation(text)
    print("After Removing Punctuation:")
    print(text[:500], '...\n')

    text = cleaner._remove_numbers(text)
    print("After Removing Numbers:")
    print(text[:500], '...\n')

    text = cleaner._to_lowercase(text)
    print("After Lowercasing:")
    print(text[:500], '...\n')

    tokens = cleaner._tokenize(text)
    print("After Tokenizing:")
    print(tokens[:50], '...\n')

    tokens = cleaner._remove_stopwords(tokens)
    print("After Removing Stopwords:")
    print(tokens[:50], '...\n')

    tokens = cleaner._lemmatize(tokens)
    print("After Lemmatization:")
    print(tokens[:50], '...\n')

    tokens = cleaner._stem(tokens)
    print("After Stemming:")
    print(tokens[:50], '...\n')

    clean_text = ' '.join(tokens)
    print("Final Cleaned Text:")
    print(clean_text[:500], '...\n')

    return clean_text

def main():
    # Step 1: Scrape article
    input_url = input("Enter the article URL: ").strip()
    article_data = step_1_scrape_article(input_url)

    # Step 2: Load raw article text
    raw_text = load_text_from_json("data.json")

    # Step 2.5: Clean the text and show intermediate steps
    clean_text = debug_clean_text_steps(raw_text)

    # Step 3: Generate claims from the cleaned text
    ranked_claims = step_2_generate_claims(clean_text)

    if not ranked_claims:
        print("No valid claims found.")
        return

    print("\nTop Claims:")
    for i, (claim, score) in enumerate(ranked_claims[:3], 1):
        print(f"{i}. (Score: {score:.2f}) {claim}")

if __name__ == "__main__":
    main()
