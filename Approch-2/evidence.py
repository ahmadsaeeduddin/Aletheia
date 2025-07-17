import os
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from groq import Groq  # Groq official client
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from generate_question import ClaimQuestionGenerator

# === Load environment variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY_1")

# === Initialize Groq client ===
groq_client = Groq(api_key=GROQ_API_KEY)

class FactChecker:
    def __init__(self, kb_chunks: List[str], threshold: float = 0.6):
        """
        kb_chunks: List of text chunks from your knowledge base.
        threshold: Similarity threshold to consider a chunk as evidence.
        """
        self.kb_chunks = kb_chunks
        self.threshold = threshold
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Or your choice

    def find_evidence(self, claim: str) -> List[str]:
        """
        Find relevant evidence chunks for a claim by similarity.
        """
        claim_emb = self.embedder.encode([claim])
        kb_embs = self.embedder.encode(self.kb_chunks)
        sims = cosine_similarity(claim_emb, kb_embs)[0]

        evidence = []
        for chunk, sim in zip(self.kb_chunks, sims):
            if sim >= self.threshold:
                evidence.append(chunk)
        return evidence

    def query_llm(self, claim: str, question: str, evidence: List[str]) -> Dict:
        """
        Query the Groq LLM with claim, question, and selected evidence.
        """
        context = "\n".join(evidence)
        prompt = (
            f"Claim: {claim}\n"
            f"Question: {question}\n"
            f"Evidence:\n{context}\n\n"
            "Answer the question with Yes/No/Neutral. "
            "Provide justification and categorize with one of: "
            "Conflicting Evidence, False, True, Not enough Evidence."
        )

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Replace with your Groq model
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content
        return {
            "claim": claim,
            "question": question,
            "evidence": evidence,
            "llm_output": answer
        }

    def fact_check(self, claims_questions: List[Dict[str, str]]) -> List[Dict]:
        """
        Full pipeline for a batch of claims/questions.
        Each dict must have 'claim' and 'question' keys.
        """
        results = []
        for item in claims_questions:
            claim = item['claim']
            question = item['question']
            evidence = self.find_evidence(claim)
            result = self.query_llm(claim, question, evidence)
            results.append(result)
        return results

    def load_kb_chunks_from_json(kb_dir: str = "knowledge_base", max_words: int = 8000) -> List[str]:
        """
        Loads text & title from all JSON files in a directory, then chunks and cleans them.
        Returns a list of chunks across all articles.
        """
        all_chunks = []

        for filename in os.listdir(kb_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(kb_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = data.get("title", "")
                    text = data.get("text", "")
                    full_text = f"{title}. {text}" if title and title not in text else text

                    # Clip to max words
                    words = word_tokenize(full_text)
                    if len(words) > max_words:
                        full_text = " ".join(words[:max_words])

                    # Sentence-based chunking
                    sentences = sent_tokenize(full_text)
                    chunk_size = max(1, len(sentences) // 8)  # 8 chunks max
                    for i in range(0, len(sentences), chunk_size):
                        chunk = " ".join(sentences[i:i + chunk_size])
                        all_chunks.append(chunk)

        print(f"📚 Loaded {len(all_chunks)} total chunks from {kb_dir}")
        return all_chunks
    
    def generate_claim_question_pairs(claims: List[str]) -> List[Dict[str, str]]:
        """
        For each claim, generate one neutral yes/no question.
        Returns: List of {"claim": ..., "question": ...} dictionaries.
        """
        claim_question_pairs = []

        for claim in claims:
            try:
                questions = ClaimQuestionGenerator().generate_questions(claim)
                best_question = questions[0]  # You can switch to questions[1] if preferred
                claim_question_pairs.append({
                    "claim": claim.strip(),
                    "question": best_question.strip()
                })
            except Exception as e:
                print(f"❌ Error generating question for claim: '{claim}'\n{e}")

        return claim_question_pairs


def main():
    # === 1) Step: Load chunks from knowledge base ===
    kb_chunks = FactChecker.load_kb_chunks_from_json("knowledge_base")
    
    # === 2) Step: Input your generated claims ===
    claims = [
       "Jota played in 26 of Liverpool's 38 league games, scoring six goals and providing four assists.",
       "The car crash that killed Diogo Jota and his brother was caused by a 'burst tire while overtaking' on the A52 road in Zamora, northwestern Spain.",
       "Diogo Jota was 28 years old when he died in a car crash in Spain.",
       "The vehicle that Jota and his brother were in left the road and subsequently caught fire.",
       "The crash is being investigated as 'a possible speeding incident'."
    ]

    # === 3) Step: Generate claim-question pairs ===
    print("\n🧠 Generating neutral Yes/No questions for claims...")
    claim_question_pairs = FactChecker.generate_claim_question_pairs(claims)

    # === 4) Step: Initialize FactChecker ===
    fact_checker = FactChecker(kb_chunks, threshold=0.6)

    # === 5) Step: Perform fact-checking ===
    print("\n🔍 Running fact-checking against knowledge base...")
    results = fact_checker.fact_check(claim_question_pairs)

    # === 6) Step: Show output ===
    print("\n🧾 Final Results:\n")
    for res in results:
        print("=" * 50)
        print(f"Claim    : {res['claim']}")
        print(f"Question : {res['question']}")
        print(f"Evidence :\n- " + "\n- ".join(res['evidence']))
        print(f"\nLLM Output:\n{res['llm_output']}")
        print("=" * 50)


# if __name__ == "__main__":
#     main()
