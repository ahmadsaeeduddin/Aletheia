import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

class TextCleaner:
    def __init__(self, max_features=5000):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')

    def clean_text(self, text):
        """Perform all cleaning steps on raw text"""
        text = self._remove_html_tags(text)
        text = self._remove_urls(text)
        text = self._remove_punctuation(text)
        text = self._remove_numbers(text)
        text = self._to_lowercase(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)
        tokens = self._stem(tokens)
        clean_text = ' '.join(tokens)
        return clean_text

    def _remove_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def _remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)

    def _remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def _remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def _to_lowercase(self, text):
        return text.lower()

    def _tokenize(self, text):
        return word_tokenize(text)

    def _remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def _lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def _stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def vectorize_text(self, clean_texts):
        """Convert cleaned text to TF-IDF vectors"""
        tfidf_vectors = self.vectorizer.fit_transform(clean_texts)
        return tfidf_vectors