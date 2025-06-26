import requests
from bs4 import BeautifulSoup

def search_duckduckgo_lite(query, date_filter=""):
    url = "https://lite.duckduckgo.com/lite"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # Add filter in the form: df = "" for Any Time, "d" = Past Day, "w" = Past Week, "m" = Past Month, "y" = Past Year.
    data = {
        "q": query,
        "df": date_filter  # Can be "d", "w", "m", "y" or "" for Any Time
    }

    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all("a", class_="result-link")
    return [link["href"] for link in links if link.has_attr("href")]

# ✅ Example usage
search_term = input("Enter your search query: ").strip()
# Optional: ask for time filter
filter_input = input("Time filter (empty for 'Any Time', options: d/w/m/y): ").strip().lower()

results = search_duckduckgo_lite(search_term, date_filter=filter_input)

print("\n🔗 Extracted URLs:")
for i, url in enumerate(results, 1):
    print(f"{i}. {url}")



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