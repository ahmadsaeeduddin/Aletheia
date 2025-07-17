import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from groq import Groq

# Load your SerpAPI key
load_dotenv()

# -----------------------------
# 🔍 Google Search using SerpAPI
# -----------------------------
def google_search(query, num_results=10):
    print(f"\n🔍 Google Search for: \"{query}\"")
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("❌ Missing SERPAPI_KEY in .env file.")
        return []

    params = {
        "q": query,
        "num": num_results,
        "api_key": api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    links = []

    for result in results.get("organic_results", []):
        link = result.get("link")
        if link:
            links.append(link)

    return links

# -----------------------------
# 🦆 DuckDuckGo Lite search for site-specific fact-checks
# -----------------------------
def duckduckgo_search(query, date_filter="", site=""):
    url = "https://lite.duckduckgo.com/lite"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # Example: "Trump Gaza site:snopes.com"
    data = {
        "q": f"{query} site:{site}",
        "df": date_filter
    }

    results = []
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all("a", class_="result-link")
        results = [link["href"] for link in links if link.has_attr("href")]
    except Exception as e:
        print(f"❌ DuckDuckGo search failed: {e}")

    return results


def generate_urls_for_query(query):
    """
    Use Groq's LLaMA 3 model to generate top 10 website URLs related to a search query.
    """
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY_4"),
    )
    prompt = f"""
You are a helpful assistant. Your task is to provide a list of the top 10 website URLs that are most relevant to the following user query.

Query: "{query}"

Only return 10 valid URLs, starting with "https://"

Only return the 10 URLs in a numbered list format. Do not include any additional explanation.
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# ---------------------------------
# 🧪 Quick test
# ---------------------------------
# if __name__ == "__main__":
#     query = input("🔎 Enter your search query: ").strip()

#     google_results = google_search(query, num_results=5)
#     print("\n🌐 Google Search Results:")
#     if google_results:
#         for i, url in enumerate(google_results, 1):
#             print(f"{i}. {url}")
#     else:
#         print("⚠️ No Google results found.")

#     duck_results_snopes = duckduckgo_search(query, site="snopes.com")
#     print("\n🦆 DuckDuckGo Snopes Results:")
#     if duck_results_snopes:
#         for i, url in enumerate(duck_results_snopes, 1):
#             print(f"{i}. {url}")
#     else:
#         print("⚠️ No DuckDuckGo Snopes results found.")

#     duck_results_politifact = duckduckgo_search(query, site="politifact.com")
#     print("\n🦆 DuckDuckGo Politifact Results:")
#     if duck_results_politifact:
#         for i, url in enumerate(duck_results_politifact, 1):
#             print(f"{i}. {url}")
#     else:
#         print("⚠️ No DuckDuckGo Politifact results found.")
        
#     duck_results = duckduckgo_search(query, site="")
#     print("\n🦆 DuckDuckGo Politifact Results:")
#     if duck_results:
#         for i, url in enumerate(duck_results, 1):
#             print(f"{i}. {url}")
#     else:
#         print("⚠️ No DuckDuckGo results found.")
