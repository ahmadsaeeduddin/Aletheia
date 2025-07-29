from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Prompt user for query
query = input("Enter your Google News search query: ")

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--start-maximized")

# Optional: run headless
# chrome_options.add_argument("--headless")

# Setup Chrome driver
service = Service()
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Open Google
    driver.get("https://www.google.com")

    # Accept cookies if you get a popup (optional)
    # Example: click 'I agree' if needed here

    # Find the textarea input and enter query
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "APjFqb"))
    )
    search_box.clear()
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    # Wait for results page to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "search"))
    )

    # Click the 'News' tab
    news_tab = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//div[text()='News']"))
    )
    news_tab.click()

    # Wait for news results to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.SoaBEf"))
    )

    # Get all news result links
    news_results = driver.find_elements(By.CSS_SELECTOR, "div.SoaBEf a[jsname='YKoRaf']")

    urls = []
    for result in news_results:
        link = result.get_attribute("href")
        if link and link.startswith("http"):
            urls.append(link)

    print("\nFetched News URLs:")
    for url in urls:
        print(url)

finally:
    driver.quit()