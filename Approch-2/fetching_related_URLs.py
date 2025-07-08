from googlesearch import search as google_search_api
import requests
from bs4 import BeautifulSoup

def google_search(query, num_results=10):
    print(f"\n🔍 Top {num_results} Google results for: \"{query}\"\n")
    results = []
    try:
        for i, url in enumerate(google_search_api(query, stop=num_results), 1):
            results.append(url)
    except Exception as e:
        print(f"❌ Google search failed: {e}")
    return results


def duckduckgo_search(query, date_filter="" , site = ""):
    url = "https://lite.duckduckgo.com/lite"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "q": query  + " site: " + site,
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


if __name__ == "__main__":
    query = input("🔎 Enter your search query: ").strip()
    filter_input = input("📅 Time filter for DuckDuckGo (empty for 'Any Time', options: d/w/m/y): ").strip().lower()
    
    

    google_results = google_search(query, num_results=10)
    print("\n🌐 Google Search Results: ")
    if google_results:
        for i, url in enumerate(google_results, 1):
            print(f"{i}. {url}")
    else:
        print("⚠️ No results from Google.")

    duck_results_snopes = duckduckgo_search(query, date_filter=filter_input, site="snopes.com")
    print("\n🦆 DuckDuckGo Search Results: [ --- Snopes --- ]")
    if duck_results_snopes:
        for i, url in enumerate(duck_results_snopes, 1):
            print(f"{i}. {url}")
    else:
        print("⚠️ No results from DuckDuckGo.")
        
    duck_results_politifact = duckduckgo_search(query, date_filter=filter_input, site="politifact.com")
    print("\n🦆 DuckDuckGo Search Results: [ --- Politifact --- ]")
    if duck_results_politifact:
        for i, url in enumerate(duck_results_politifact, 1):
            print(f"{i}. {url}")
    else:
        print("⚠️ No results from DuckDuckGo.")
        
    
# import requests
# from bs4 import BeautifulSoup

# def search_duckduckgo_snopes(claim, date_filter=""):
#     # Use DuckDuckGo Lite interface
#     url = "https://lite.duckduckgo.com/lite"
#     headers = {
#         "User-Agent": "Mozilla/5.0",
#         "Content-Type": "application/x-www-form-urlencoded"
#     }

#     # Combine query with site restriction for Snopes fact-checks
#     query = f"{claim} site:snopes.com/fact-check"

#     data = {
#         "q": query,
#         "df": date_filter  # "" (Any Time), "d", "w", "m", "y"
#     }

#     response = requests.post(url, headers=headers, data=data)
#     response.raise_for_status()
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Extract all <a class="result-link"> with Snopes fact-check URLs only
#     result_links = soup.find_all("a", class_="result-link")
#     snopes_urls = [
#         link["href"]
#         for link in result_links
#         if link.has_attr("href") and "snopes.com/fact-check" in link["href"]
#     ]

#     return snopes_urls

# # 🔍 User input
# claim = input("Enter your claim or question: ").strip()
# time_filter = input("Time filter (empty for Any Time, options: d=Day, w=Week, m=Month, y=Year): ").strip().lower()

# # 🔎 Search and extract Snopes URLs
# snopes_results = search_duckduckgo_snopes(claim, time_filter)

# # 📤 Output
# if snopes_results:
#     print("\n✅ Snopes Fact-Check URLs found:\n")
#     for i, url in enumerate(snopes_results, 1):
#         print(f"{i}. {url}")
# else:
#     print("\n❌ No Snopes fact-check URLs found.")