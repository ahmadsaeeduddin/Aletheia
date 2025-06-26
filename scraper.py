import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from urllib.parse import urlparse, urljoin
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import dateutil.parser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentScraper:
    def __init__(self, headless=True, wait_time=10):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.headless = headless
        self.wait_time = wait_time
        self.driver = None
        
    def _setup_driver(self):
        """Setup Selenium WebDriver for dynamic content"""
        if self.driver is None:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(self.wait_time)
    
    def _close_driver(self):
        """Close Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _identify_platform(self, url):
        """Identify the platform/website type"""
        domain = urlparse(url).netloc.lower()
        
        social_platforms = {
            'twitter.com': 'Twitter',
            'x.com': 'Twitter',
            'facebook.com': 'Facebook',
            'instagram.com': 'Instagram',
            'linkedin.com': 'LinkedIn',
            'youtube.com': 'YouTube',
            'tiktok.com': 'TikTok',
            'reddit.com': 'Reddit'
        }
        
        for platform_domain, platform_name in social_platforms.items():
            if platform_domain in domain:
                return platform_name, True
        
        return domain, False
    
    def _extract_with_requests(self, url):
        """Try to extract content using requests + BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.warning(f"Requests extraction failed: {e}")
            return None
    
    def _extract_with_selenium(self, url):
        """Extract content using Selenium for dynamic content"""
        try:
            self._setup_driver()
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, self.wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(3)
            
            return BeautifulSoup(self.driver.page_source, 'html.parser')
        except Exception as e:
            logger.warning(f"Selenium extraction failed: {e}")
            return None
    
    def _extract_meta_tags(self, soup):
        """Extract metadata from meta tags"""
        meta_data = {}
        
        # Common meta tags
        meta_mappings = {
            'og:title': 'title',
            'twitter:title': 'title',
            'og:description': 'description',
            'twitter:description': 'description',
            'og:site_name': 'site_name',
            'og:url': 'canonical_url',
            'article:published_time': 'published_time',
            'article:author': 'author',
            'og:type': 'content_type'
        }
        
        for meta_tag in soup.find_all('meta'):
            property_val = meta_tag.get('property') or meta_tag.get('name')
            content = meta_tag.get('content')
            
            if property_val and content:
                if property_val in meta_mappings:
                    meta_data[meta_mappings[property_val]] = content
        
        return meta_data
    
    def _extract_article_content(self, soup, platform):
        """Extract article content based on common patterns"""
        content_selectors = [
            'article',
            '[role="main"]',
            '.post-content',
            '.entry-content',
            '.article-body',
            '.story-body',
            '.content',
            'main'
        ]
        
        text_content = ""
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    # Remove script and style elements
                    for script in element(["script", "style", "nav", "header", "footer", "aside"]):
                        script.decompose()
                    
                    # Extract text
                    element_text = element.get_text(strip=True, separator=' ')
                    if len(element_text) > len(text_content):
                        text_content = element_text
                
                if text_content:
                    break
        
        # Fallback to body content if no specific content found
        if not text_content:
            body = soup.find('body')
            if body:
                for script in body(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()
                text_content = body.get_text(strip=True, separator=' ')
        
        return text_content
    
    def _extract_twitter_content(self, soup):
        """Extract Twitter-specific content"""
        data = {}
        
        # Twitter selectors (may need updates as Twitter changes)
        tweet_selectors = [
            '[data-testid="tweetText"]',
            '.tweet-text',
            '[data-testid="tweet"]'
        ]
        
        for selector in tweet_selectors:
            elements = soup.select(selector)
            if elements:
                data['text'] = ' '.join([elem.get_text(strip=True) for elem in elements])
                break
        
        # Extract author
        author_selectors = [
            '[data-testid="User-Name"]',
            '.username',
            '.fullname'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                data['author'] = element.get_text(strip=True)
                break
        
        return data
    
    def _extract_facebook_content(self, soup):
        """Extract Facebook-specific content"""
        data = {}
        
        # Facebook post content
        post_selectors = [
            '[data-ad-preview="message"]',
            '.userContent',
            '[data-testid="post_message"]'
        ]
        
        for selector in post_selectors:
            element = soup.select_one(selector)
            if element:
                data['text'] = element.get_text(strip=True)
                break
        
        return data
    
    def _parse_date(self, date_string):
        """Parse various date formats"""
        if not date_string:
            return None
        
        try:
            # Try parsing with dateutil (handles most formats)
            parsed_date = dateutil.parser.parse(date_string)
            return parsed_date.isoformat()
        except:
            # Try common patterns
            patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}',
                r'\w+ \d{1,2}, \d{4}'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, date_string)
                if match:
                    try:
                        parsed_date = dateutil.parser.parse(match.group())
                        return parsed_date.isoformat()
                    except:
                        continue
        
        return date_string  # Return original if parsing fails
    
    def scrape_content(self, url):
        """Main scraping function"""
        logger.info(f"Starting to scrape: {url}")
        
        # Identify platform
        platform, is_social = self._identify_platform(url)
        
        # Try requests first, then Selenium if needed
        soup = self._extract_with_requests(url)
        
        if not soup or is_social:
            logger.info("Using Selenium for dynamic content extraction")
            soup = self._extract_with_selenium(url)
        
        if not soup:
            raise Exception("Failed to extract content from URL")
        
        # Extract meta data
        meta_data = self._extract_meta_tags(soup)
        
        # Initialize result structure
        result = {
            'url': url,
            'platform': platform,
            'is_social_media': is_social,
            'scraped_at': datetime.now().isoformat(),
            'title': '',
            'text': '',
            'author': '',
            'publisher': '',
            'published_date': None,
            'site_name': platform
        }
        
        # Extract title
        title_sources = [
            meta_data.get('title'),
            soup.find('title').get_text(strip=True) if soup.find('title') else None,
            soup.find('h1').get_text(strip=True) if soup.find('h1') else None
        ]
        
        for title_source in title_sources:
            if title_source and title_source.strip():
                result['title'] = title_source.strip()
                break
        
        # Platform-specific extraction
        if platform == 'Twitter':
            twitter_data = self._extract_twitter_content(soup)
            result.update(twitter_data)
        elif platform == 'Facebook':
            facebook_data = self._extract_facebook_content(soup)
            result.update(facebook_data)
        else:
            # Generic article extraction
            result['text'] = self._extract_article_content(soup, platform)
        
        # Extract author/publisher
        if not result['author']:
            author_selectors = [
                '.author',
                '.byline',
                '[rel="author"]',
                '.post-author',
                '.article-author'
            ]
            
            for selector in author_selectors:
                element = soup.select_one(selector)
                if element:
                    result['author'] = element.get_text(strip=True)
                    break
        
        # For non-social media, try to find publisher
        if not is_social:
            result['publisher'] = meta_data.get('site_name') or platform
        
        # Extract date
        date_sources = [
            meta_data.get('published_time'),
            soup.select_one('time')['datetime'] if soup.select_one('time') and soup.select_one('time').get('datetime') else None,
            soup.select_one('.date, .publish-date, .post-date').get_text(strip=True) if soup.select_one('.date, .publish-date, .post-date') else None
        ]
        
        for date_source in date_sources:
            if date_source:
                parsed_date = self._parse_date(date_source)
                if parsed_date:
                    result['published_date'] = parsed_date
                    break
        
        # Clean up driver
        self._close_driver()
        
        return result
    
    def save_to_json(self, data, filename=None):
        """Save scraped data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scraped_content_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")
        return filename

def scrape_url(url, output_file=None):
    """Convenience function to scrape a single URL"""
    scraper = ContentScraper()
    try:
        result = scraper.scrape_content(url)
        filename = scraper.save_to_json(result, output_file)
        return result, filename
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Example URLs for testing
    test_urls = [
        "https://theintercept.com/2025/06/24/zohran-mamdani-andrew-cuomo-nyc-mayor/"
    ]
    
    scraper = ContentScraper()
    
    for url in test_urls:
        try:
            print(f"\nScraping: {url}")
            result = scraper.scrape_content(url)
            
            # Print summary
            print(f"Platform: {result['platform']}")
            print(f"Title: {result['title'][:100]}...")
            print(f"Text length: {len(result['text'])} characters")
            print(f"Author: {result['author']}")
            print(f"Published: {result['published_date']}")
            
            # Save to JSON
            scraper.save_to_json(result)
            
        except Exception as e:
            print(f"Error: {e}")