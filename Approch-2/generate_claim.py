import os
import faiss
import json
import re
import time
from sentence_transformers import SentenceTransformer
from groq import Groq  # pip install groq
from sklearn.metrics.pairwise import cosine_similarity

class GroqClaimGenerator:
    def __init__(self, model_name="", k=3):
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name
        self.k = k
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def preprocess_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\[[^\]]*\]", "", text)
        return text.strip()

    def chunk_text(self, text, max_words=100):
        words = text.split()
        chunks, chunk = [], []
        for word in words:
            chunk.append(word)
            if len(chunk) >= max_words:
                chunks.append(" ".join(chunk))
                chunk = []
        if chunk:
            chunks.append(" ".join(chunk))
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
\"\"\"
{context}
\"\"\"

Claim:
""".strip()

        response = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    def generate_claims_from_text(self, raw_text):
        text = self.preprocess_text(raw_text)
        chunks = self.chunk_text(text)
        index, embeddings = self.build_faiss_index(chunks)

        claims = []
        print(f"\n🔍 Generating claims from {len(chunks)} chunks using Groq...\n")
        for i, chunk in enumerate(chunks):
            try:
                context = " ".join(self.retrieve_context(chunk, index, chunks, embeddings))
                print(f"➡️ [{i+1}/{len(chunks)}] Generating claim...")
                claim = self.generate_claim(context)
                if claim and claim not in claims:
                    claims.append(claim)
                time.sleep(0.5)  # prevent hitting rate limits
            except Exception as e:
                print(f"❌ Error: {e}")
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

    generator = GroqClaimGenerator(model_name="llama3-8b-8192")
    claims = generator.generate_claims_from_text(article_text)

    # ➕ Filter out similar claims
    final_claims = generator.filter_similar_claims(claims)

    print("\n✅ Final Unique Claims:")
    for i, c in enumerate(final_claims, 1):
        print(f"{i}. {c}")
