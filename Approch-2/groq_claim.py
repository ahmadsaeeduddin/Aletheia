import os
import faiss
import json
import re
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import spacy


# Load .env file
load_dotenv()

class GroqClaimGenerator:
    def __init__(self, api_key, model_name="", k=3):
        self.model_name = model_name
        self.k = k
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = Groq(api_key=api_key)

    def preprocess_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\[[^\]]*\]", "", text)
        return text.strip()

    # def adaptive_chunk_text(self, text):
    #     words = text.split()
    #     total_words = len(words)

    #     # Decide number of chunks
    #     if total_words <= 5000:
    #         num_chunks = 5
    #     elif total_words <= 10000:
    #         num_chunks = 8
    #     elif total_words <= 20000:
    #         num_chunks = 10
    #     elif total_words <= 35000:
    #         num_chunks = 12
    #     else:
    #         num_chunks = 15

    #     chunk_size = total_words // num_chunks
    #     chunks = []

    #     for i in range(num_chunks):
    #         start = i * chunk_size
    #         end = (i + 1) * chunk_size if i < num_chunks - 1 else total_words
    #         chunk = words[start:end]
    #         if chunk:
    #             chunks.append(" ".join(chunk))

    #     print(f"🔹 Total words: {total_words} | Chunk size: ~{chunk_size} | Total chunks: {len(chunks)}")
    #     return chunks
    
    def adaptive_chunk_text(self, text):
        words = text.split()
        total_words = len(words)

        # Limit to first 10,000 words
        if total_words > 10000:
            words = words[:10000]
            total_words = 10000
            print("⚠️ Trimming to first 10,000 words.")

        chunk_size = 400
        chunks = []

        for i in range(0, total_words, chunk_size):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))

        print(f"🔹 Total words used: {total_words} | Chunk size: {chunk_size} | Total chunks: {len(chunks)}")
        return chunks

    def build_faiss_index(self, chunks):
        print("📦 Indexing text...")
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings

    def retrieve_context(self, query, index, chunks, embeddings):
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, self.k)
        return [chunks[i] for i in indices[0]]

    def generate_claim(self, context):
        prompt = f"""
You are an expert fact-checking assistant. Your task is to read the following context and convert it into one fact-checkable, concise, and specific claim in a single sentence.

Only give the claim and nothing else.

Context:
\"\"\"{context}\"\"\"

Claim:
""".strip()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    def generate_claims_from_text(self, raw_text):
        text = self.preprocess_text(raw_text)
        chunks = self.adaptive_chunk_text(text)
        index, embeddings = self.build_faiss_index(chunks)

        print(f"\n🔍 Generating claims from {len(chunks)} chunks sequentially...\n")

        claims = []
        for i, chunk in enumerate(chunks):
            context = " ".join(self.retrieve_context(chunk, index, chunks, embeddings))
            print(f"➡️ [{i + 1}/{len(chunks)}] Generating claim...")
            try:
                claim = self.generate_claim(context)
                if claim:
                    claims.append(claim)
            except Exception as e:
                print(f"❌ Error generating claim for chunk {i + 1}: {e}")
                continue
        return claims

    def filter_similar_claims(self, claims, threshold=0.85):
        print("\n🧹 Filtering similar claims...")
        unique_claims = []
        embeddings = self.embedder.encode(claims, convert_to_numpy=True)

        for i, emb in enumerate(embeddings):
            if i == 0:
                unique_claims.append(claims[i])
                continue

            sims = cosine_similarity([emb], embeddings[:i])[0]
            max_sim = max(sims)

            if max_sim < threshold:
                unique_claims.append(claims[i])
            else:
                print(f"⚠️ Removed similar claim: \"{claims[i]}\" (similarity: {max_sim:.2f})")

        return unique_claims
    
    def score_claims_nlp(self, claims):
        print("\n📊 Scoring claims using NLP (Named Entity Recognition + length)...")
        nlp = spacy.load("en_core_web_sm")
        scored_claims = []

        for claim in claims:
            doc = nlp(claim)
            num_entities = len(doc.ents)
            num_tokens = len(doc)
            score = num_entities + (0.1 * num_tokens)
            scored_claims.append((claim, score))

        # Sort by score descending
        return sorted(scored_claims, key=lambda x: x[1], reverse=True)



def load_article_text(filepath="data.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    title = data.get("title", "").strip()
    text = data.get("text", "").strip()
    return f"{title}. {text}" if title and title not in text else text


if __name__ == "__main__":
    article_text = load_article_text()
    if not article_text:
        print("🚫 No article text found.")
        exit()

    api_key = os.getenv("GROQ_API_KEY_4")
    if not api_key:
        print("🚫 No API key found in .env file.")
        exit()

    generator = GroqClaimGenerator(api_key=api_key, model_name="llama3-8b-8192")
    claims = generator.generate_claims_from_text(article_text)
    final_claims = generator.filter_similar_claims(claims)
    scored_claims = generator.score_claims_nlp(final_claims)

    print("\n🏆 Top Factful Claims (via NLP):")
    for i, (claim, score) in enumerate(scored_claims, 1):
        print(f"{i}. (Score: {score:.2f}) {claim}")

