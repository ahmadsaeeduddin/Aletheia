import os
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from text_cleaner import TextCleaner


class EvidenceExtractor:
    def __init__(self, kb_folder="knowledge_base", similarity_threshold=0.3):
        self.kb_folder = kb_folder
        self.similarity_threshold = similarity_threshold
        self.cleaner = TextCleaner()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def extract_evidence_for_claim(self, claim):
        evidence_list = []

        print(f"\n🔍 Extracting evidence for claim:\n\"{claim}\"\n")

        # Iterate over all files in the knowledge base folder
        for filename in os.listdir(self.kb_folder):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(self.kb_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load {filename}: {e}")
                    continue

            platform = data.get("platform", "unknown")
            is_social_media = data.get("is_social_media", False)
            raw_text = data.get("text", "")

            # Clean and segment the text
            cleaned_text = self.cleaner.clean_text(raw_text)
            sentences = self.cleaner.segment_sentences(cleaned_text)

            if not sentences:
                continue

            # Compute similarity between claim and all sentences
            similarity, matched_index = self._find_best_matching_index(claim, sentences)

            if similarity >= self.similarity_threshold:
                context_sentences = self._get_context(sentences, matched_index)

                evidence_entry = {
                    "platform": platform,
                    "is_social_media": is_social_media,
                    "evidence": context_sentences
                }
                evidence_list.append(evidence_entry)
                print(f"✅ Match found in: {filename} (similarity: {similarity:.2f})")
            else:
                print(f"❌ No match >= {self.similarity_threshold} in: {filename}")

        return evidence_list

    def _find_best_matching_index(self, claim, sentences):
        all_sentences = [claim] + sentences
        tfidf_matrix = self.vectorizer.fit_transform(all_sentences)

        claim_vector = tfidf_matrix[0]
        sentence_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(claim_vector, sentence_vectors).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        return best_score, best_idx

    def _get_context(self, sentences, idx):
        # Fetch one sentence before and one after, if they exist
        context = []
        if idx > 0:
            context.append(sentences[idx - 1])
        context.append(sentences[idx])
        if idx < len(sentences) - 1:
            context.append(sentences[idx + 1])
        return context

    def save_evidence_to_json(self, evidence_list, output_path="evidences.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evidence_list, f, indent=4, ensure_ascii=False)
        print(f"\n💾 Saved evidence to: {output_path}")


# Run independently for testing
if __name__ == "__main__":
    claim_input = input("Enter claim: ").strip()
    extractor = EvidenceExtractor()
    evidence = extractor.extract_evidence_for_claim(claim_input)
    extractor.save_evidence_to_json(evidence)
