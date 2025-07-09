import spacy
from textblob import TextBlob
from openie import StanfordOpenIE
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

class ClaimExtractor:
    def __init__(self):
        self.client = StanfordOpenIE()
        
    def identify_checkworthy_claims(self, text):
        """Identify check-worthy claims using anomaly detection"""
        sentences = self._split_sentences(text)
        anomalies = self._detect_anomalies(sentences)
        return anomalies
    
    def _split_sentences(self, text):
        doc = nlp(text)
        return [sent.text for sent in doc.sents]
    
    def _detect_anomalies(self, sentences):
        # Get embeddings for all sentences
        embeddings = sentence_model.encode(sentences)
        
        # Calculate pairwise cosine similarities
        sim_matrix = cosine_similarity(embeddings)
        
        # Calculate average similarity for each sentence
        avg_similarities = np.mean(sim_matrix, axis=1)
        
        # Find sentences with low average similarity (anomalies)
        threshold = np.percentile(avg_similarities, 25)  # Bottom 25% as anomalies
        anomalies = [sent for sent, sim in zip(sentences, avg_similarities) if sim < threshold]
        
        # Further filter by sentiment (high absolute polarity)
        filtered_anomalies = []
        for sent in anomalies:
            polarity = TextBlob(sent).sentiment.polarity
            if abs(polarity) > 0.5:  # Highly positive or negative
                filtered_anomalies.append(sent)
        
        return filtered_anomalies
    
    def extract_claim_triples(self, text):
        """Extract SPO triples from text using OpenIE"""
        triples = self.client.extract(text)
        return triples
    
    def process_article(self, text):
        """Full pipeline: identify check-worthy sentences and extract triples"""
        checkworthy = self.identify_checkworthy_claims(text)
        results = []
        
        for sentence in checkworthy:
            triples = self.extract_claim_triples(sentence)
            if triples:
                results.append({
                    'sentence': sentence,
                    'triples': triples,
                    'entities': self._extract_entities(sentence),
                    'sentiment': TextBlob(sentence).sentiment.polarity
                })
        
        return results
    
    def _extract_entities(self, text):
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities