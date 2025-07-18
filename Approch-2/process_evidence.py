import json
from collections import defaultdict

class EvidenceFilter:
    def __init__(self, top_n=10, social_media_penalty=0.6):
        """
        :param top_n: Number of top unique platform evidences to keep
        :param social_media_penalty: Weight multiplier for social media sources (default = 0.6)
        """
        self.top_n = top_n
        self.social_media_penalty = social_media_penalty

    def compute_importance(self, similarity_score, is_social_media):
        """
        Compute importance of evidence based on source type.
        """
        weight = self.social_media_penalty if is_social_media else 1.0
        return round(similarity_score * weight, 3)

    def filter(self, evidence_list):
        """
        Filter evidence to retain only top-N most relevant and unique platform entries.
        :param evidence_list: List of dicts, each with 'platform', 'similarity_score', 'evidence', 'is_social_media'
        :return: List of filtered evidence entries (dicts)
        """
        platform_best = {}
        seen_hashes = set()

        for entry in evidence_list:
            platform = entry.get("platform", "").strip().lower()
            score = entry.get("similarity_score", 0)
            is_social = entry.get("is_social_media", False)
            evidence_sentences = entry.get("evidence", [])
            evidence_text = " ".join(evidence_sentences).strip()

            if not platform or not evidence_text:
                continue  # skip empty

            evidence_hash = hash(evidence_text)
            if evidence_hash in seen_hashes:
                continue  # avoid duplicates

            importance = self.compute_importance(score, is_social)

            new_entry = {
                "platform": platform,
                "is_social_media": is_social,
                "similarity_score": round(score, 3),
                "importance": importance,
                "evidence": evidence_sentences,
                "evidence_text": evidence_text
            }

            if platform not in platform_best or score > platform_best[platform]["similarity_score"]:
                platform_best[platform] = new_entry
                seen_hashes.add(evidence_hash)

        # Sort by importance descending
        sorted_filtered = sorted(platform_best.values(), key=lambda x: x["importance"], reverse=True)
        return sorted_filtered[:self.top_n]

    def save_to_json(self, filtered_evidence, output_path="filtered_evidence.json"):
        """
        Save filtered evidence to a JSON file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_evidence, f, indent=4, ensure_ascii=False)
        print(f"💾 Saved filtered evidence to: {output_path}")


def main():
    input_path = "evidences.json"
    output_path = "filtered_evidence.json"

    # Step 1: Load raw evidence
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            evidence_list = json.load(f)
        print(f"Loaded {len(evidence_list)} evidence entries from: {input_path}")
    except FileNotFoundError:
        print(f"Error: File not found - {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_path}")
        return

    # Step 2: Filter top evidence
    filterer = EvidenceFilter(top_n=10, social_media_penalty=0.6)
    filtered_evidence = filterer.filter(evidence_list)
    print(f"✅ Selected {len(filtered_evidence)} unique platform evidences.")

    # Step 3: Save filtered results
    filterer.save_to_json(filtered_evidence, output_path)


if __name__ == "__main__":
    main()
