import os
from dotenv import load_dotenv
from scraper2 import ContentScraper
from groq_claim import GroqClaimGenerator
from text_cleaner import TextCleaner
from query_extractor import DynamicKeyPhraseExtractor
from search_engine import WebSearcher
from build_knowledge_base import KnowledgeBaseBuilder
import json

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY_4")

def step_1_scrape_article(url: str, save_file="data.json"):
    print(f"\nScraping article: {url}")
    scraper = ContentScraper()
    result = scraper.scrape_content(url)
    scraper.save_to_json(result, save_file)
    return result

def load_text_from_json(path="data.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    title = data.get("title", "")
    text = data.get("text", "")
    return f"{title}. {text}" if title and title not in text else text

def step_2_clean_text(text):
    cleaner = TextCleaner()
    
    print("\nOriginal Text:")
    print(text[:500], '...\n')

    text = cleaner._remove_html_tags(text)
    text = cleaner._expand_contractions(text)
    text = cleaner._remove_urls(text)
    text = cleaner._to_lowercase(text)
    tokens = cleaner._tokenize(text)
    tokens = cleaner._lemmatize(tokens)

    clean_text = ' '.join(tokens)
    print("Final Cleaned Text:")
    print(clean_text[:500], '...\n')

    return clean_text

def step_3_generate_claims(text: str):
    print("\nGenerating claims from cleaned text...")
    claim_gen = GroqClaimGenerator(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    claims = claim_gen.generate_claims_from_text(text)
    filtered = claim_gen.filter_similar_claims(claims)
    ranked = claim_gen.score_claims_nlp(filtered)
    return ranked

def step_4_extract_keywords(claim):
    extractor = DynamicKeyPhraseExtractor()
    keyphrases = extractor.get_meaningful_phrases(claim)
    print(f"\nExtracted keyphrases for claim:\n\"{claim}\"")
    for phrase, score in keyphrases:
        print(f"- {phrase} (Score: {score:.4f})")
    return keyphrases

def step_5_search_related_links(query, input_url, save_file="related_urls.txt"):
    searcher = WebSearcher(save_file=save_file)
    google_links = searcher.google_search(query)
    duck_links = searcher.duckduckgo_search(query)
    all_links = list(set(google_links + duck_links))
    #all_links = list(set(duck_links))


    # Remove input_url if present
    all_links = [url for url in all_links if url.strip() != input_url.strip()]

    # Rewrite the file to exclude the original URL
    with open(save_file, "w", encoding="utf-8") as f:
        for url in all_links:
            f.write(url + "\n")

    print(f"\n✅ Final {len(all_links)} links saved (excluding the input URL).")


def step_6_build_knowledge_base(claim_text, url_file="related_urls.txt"):
    kb_builder = KnowledgeBaseBuilder()
    urls = kb_builder.load_unique_urls(url_file)

    if not urls:
        print("[WARNING] No URLs found to build knowledge base.")
        return

    kb_builder.build(claim_text, urls)


def main():
    # Step 1: Scrape article
    input_url = input("Enter the article URL: ").strip()
    article_data = step_1_scrape_article(input_url)
    raw_text = load_text_from_json("data.json")

    # Step 2: Clean the text
    clean_text = step_2_clean_text(raw_text)

    # Step 3: Generate claims
    ranked_claims = step_3_generate_claims(clean_text)

    if not ranked_claims:
        print("No valid claims found.")
        return

    print("\nFiltered & Ranked Claims:")
    for idx, (claim, score) in enumerate(ranked_claims, 1):
        print(f"{idx}. (Score: {score:.2f}) {claim}")

    print("\nEnter the number of the claim you want to extract keyphrases for.")
    print("Or type 'all' to extract keyphrases for all claims.")
    user_input = input("Your choice: ").strip().lower()

    if user_input == "all":
        for idx, (claim, _) in enumerate(ranked_claims, 1):
            print(f"\nProcessing claim #{idx}...")
            
            # Step 4: Extract Keywords
            keyphrases = step_4_extract_keywords(claim)
            if keyphrases:
                top_phrase = keyphrases[0][0]

                # Step 5: Search Related Links
                step_5_search_related_links(top_phrase, input_url)

                # Step 6: Buld Knowledge Base
                step_6_build_knowledge_base(claim)

    elif user_input.isdigit():
        selected_index = int(user_input) - 1
        if 0 <= selected_index < len(ranked_claims):
            selected_claim = ranked_claims[selected_index][0]

            # Step 4: Extract Keywords
            keyphrases = step_4_extract_keywords(selected_claim)
            if keyphrases:
                top_phrase = keyphrases[0][0]

                # Step 5: Search Related Links
                step_5_search_related_links(top_phrase, input_url)

                # Step 6: Buld Knowledge Base
                step_6_build_knowledge_base(selected_claim)
        else:
            print("Invalid selection. Index out of range.")
    else:
        print("Invalid input. Please enter a valid number or 'all'.")

if __name__ == "__main__":
    print('The system starts ..........')
    main()
