import os
import json
import time
import fitz  # PyMuPDF
import faiss
import numpy as np
from fpdf import FPDF
from sentence_transformers import SentenceTransformer, util
from groq import Groq

from dotenv import load_dotenv
load_dotenv()


class ClaimFactChecker:
    def __init__(self, json_folder, pdf_path, groq_api_key):
        self.json_folder = json_folder
        self.original_pdf_path = pdf_path
        self.pdf_path = ""
        self.groq_api_key = groq_api_key
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.index = None

    def clean_text(self, text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    def merge_json_to_pdf(self):
        print("📄 Merging JSON files into PDF...")
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for filename in os.listdir(self.json_folder):
            if filename.endswith('.json'):
                with open(os.path.join(self.json_folder, filename), 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        title = data.get("title", "No Title")
                        text = data.get("text", "No Text")

                        pdf.set_font("Arial", 'B', size=14)
                        pdf.multi_cell(0, 10, self.clean_text(f"Title: {title}"))
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, self.clean_text(text))
                        pdf.ln(10)
                    except json.JSONDecodeError:
                        print(f"⚠️ Warning: Could not decode {filename}")
                        continue

        # Create timestamped PDF name
        timestamp = int(time.time())
        self.pdf_path = self.original_pdf_path.replace(".pdf", f"_{timestamp}.pdf")
        pdf.output(self.pdf_path)
        print(f"✅ PDF created at: {self.pdf_path}")

    def extract_text_chunks(self, chunk_size=500):
        print("📑 Extracting and chunking text from PDF...")
        doc = fitz.open(self.pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        self.chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    def retrieve_context(self, claim, top_k=3):
        print("🔍 Retrieving relevant context for the claim...")
        query_emb = self.model.encode([claim], batch_size=1, convert_to_tensor=True)
        chunk_embs = self.model.encode(self.chunks, batch_size=32, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, chunk_embs)[0]
        top_k_idx = np.argsort(-scores.cpu().numpy())[:top_k]

        top_chunks = [self.chunks[i] for i in top_k_idx]
        return "\n\n---\n\n".join(top_chunks)

    def classify_claim_with_groq(self, claim, context):
        client = Groq(api_key=self.groq_api_key)

        prompt = f"""
You are a professional fact-checking AI.

Below is a CLAIM and several CONTEXT EXCERPTS retrieved from a knowledge base of trusted news sources. Your task is to:
- Determine if the claim is supported, refuted, or not addressed by the context.
- Explain why, using specific parts of the context as evidence.

CLAIM:
"{claim}"

CONTEXT:
{context}

Respond in the following format:
Label: <True / Refuted / Conflicting Evidence / Lack of Evidence>
Justification: <Short explanation with specific references to the context>
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in fact-checking and logical reasoning."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    def run_pipeline(self, claim):
        self.merge_json_to_pdf()
        self.extract_text_chunks()
        context = self.retrieve_context(claim)

        print("\n🤖 Classifying claim using Groq...")
        result = self.classify_claim_with_groq(claim, context)

        print("\n🔎 Claim:", claim)
        print("\n✅ Classification Result:\n", result)


# if __name__ == "__main__":
#     json_folder = "C:\\Users\\Saeed\\Desktop\\Genesys-Lab\\FakeNewsDetection\\Approch-2\\knowledge_base"
#     pdf_path = "knowledge_base.pdf"
#     groq_api_key = os.getenv("GROQ_API_KEY_4")

#     checker = ClaimFactChecker(json_folder, pdf_path, groq_api_key)
#     checker.run_pipeline("ronaldo won the world cup in 2022")
