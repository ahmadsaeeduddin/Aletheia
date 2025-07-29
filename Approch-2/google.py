from googlesearch import search

def google_search(query, num_results=15):
    print(f"\nTop {num_results} Google results for: \"{query}\"\n")
    for i, url in enumerate(search(query, num_results=num_results), 1):
        print(f"{i}. {url}")

if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    google_search(user_query)