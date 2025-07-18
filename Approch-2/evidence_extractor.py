import os
import json
from sentence_transformers import SentenceTransformer, util
from text_cleaner import TextCleaner


class EvidenceExtractor:
    def __init__(self, kb_folder="knowledge_base", similarity_threshold=0.5, top_k=3):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb_folder = kb_folder
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.cleaner = TextCleaner()

    def extract_evidence_for_claim(self, claim):
        evidence_list = []
        seen_evidence_keys = set()

        print(f"\n🔍 Extracting evidence for claim:\n\"{claim}\"\n")

        for filename in os.listdir(self.kb_folder):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(self.kb_folder, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load {filename}: {e}")
                continue

            platform = data.get("platform", "unknown")
            is_social_media = data.get("is_social_media", False)
            raw_text = data.get("text", "")

            cleaned_text = self.cleaner.clean_text(raw_text)
            sentences = self.cleaner.segment_sentences(cleaned_text)

            if not sentences:
                continue

            top_matches = self._find_top_k_matches(claim, sentences)

            for score, idx in top_matches:
                if score < self.similarity_threshold:
                    continue

                context_sentences = self._get_context(sentences, idx)
                evidence_text = " ".join(context_sentences)

                # Avoid duplicates using hash of the evidence text
                if evidence_text in seen_evidence_keys:
                    continue
                seen_evidence_keys.add(evidence_text)

                evidence_entry = {
                    "platform": platform,
                    "is_social_media": is_social_media,
                    "similarity_score": round(score, 3),
                    "evidence": context_sentences
                }
                evidence_list.append(evidence_entry)

                print(f"✅ Match found in: {filename} (score: {score:.2f})")

        if not evidence_list:
            print("🚫 No relevant evidence found.")

        return evidence_list

    def _find_top_k_matches(self, claim, sentences, top_k=None):
        top_k = top_k or self.top_k
        k = min(top_k, len(sentences))  # Don't ask for more than available

        if k == 0:
            return []

        claim_embedding = self.model.encode(claim, convert_to_tensor=True)
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(claim_embedding, sentence_embeddings)

        if similarities.ndim == 0:  # Extremely rare case
            similarities = similarities.unsqueeze(0)

        similarities = similarities[0]  # Get similarities as a flat vector
        top_k_result = similarities.topk(k)

        # ✅ Convert top indices to list
        top_indices = top_k_result.indices.tolist()
        top_scores = top_k_result.values.tolist()

        return list(zip(top_scores, top_indices))




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


# # Run independently for testing
# if __name__ == "__main__":
#     claim_input = input("Enter claim: ").strip()
#     extractor = EvidenceExtractor()
#     evidence = extractor.extract_evidence_for_claim(claim_input)
#     extractor.save_evidence_to_json(evidence)

