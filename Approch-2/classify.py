import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load the API key from your .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY_6")

class ClaimClassifier:
    def __init__(self, api_key=GROQ_API_KEY, model_name="llama3-8b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model_name

    def format_prompt(self, claim, evidence_blocks):
        evidence_str = "\n\n".join(
            [f"[{i+1}] ({e['platform']}): {e['evidence_text']}" for i, e in enumerate(evidence_blocks)]
        )

        prompt = f"""Classify the given claim based on provided statements into one of:
1. 'Supported' if there is sufficient evidence indicating that the claim is legitimate. 
2. 'Refuted' if there is any evidence contradicting the claim.
3. 'Not Enough Evidence' if you cannot find any conclusive factual evidence either supporting or refuting the claim.
4. 'Conflicting Evidence/Cherrypicking' if there is factual evidence both supporting and refuting the claim.

Given Claim: {claim}

Evidence:
{evidence_str}

Respond in the format:
Classification: <Your answer>
Reason: <Brief justification>
"""
        return prompt

    def classify_claim(self, claim, evidence_file="filtered_evidence.json"):
        try:
            with open(evidence_file, "r", encoding="utf-8") as f:
                evidence_data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {evidence_file}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {evidence_file}")
            return None

        prompt = self.format_prompt(claim, evidence_data)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400,
                top_p=1.0
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error from Groq API: {e}")
            return None


if __name__ == "__main__":
    classifier = ClaimClassifier()

    # Prompt user for claim input only
    print("🔍 Claim Classifier")
    claim_text = input("Enter the claim you'd like to classify:\n> ").strip()

    # Always use the default evidence file
    evidence_file = "filtered_evidence.json"

    result = classifier.classify_claim(claim=claim_text, evidence_file=evidence_file)

    if result:
        print("\n📢 Classification Result:\n")
        print(result)
    else:
        print("⚠️ Unable to classify the claim.")
