# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# # ✅ Load model better suited for longer article text
# model_name = "facebook/bart-large-cnn"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# class Generate_claim:
#     def convert_paragraph_to_claim(text, max_length=60):
#         prompt = f"Convert the following into one fact-checkable claim in 1 sentence:\n{text}"

#         inputs = tokenizer([prompt], max_length=1024, truncation=True, return_tensors="pt")

#         output_ids = model.generate(
#             inputs["input_ids"],
#             max_length=max_length,
#             min_length=5,
#             length_penalty=2.0,
#             num_beams=4,
#             early_stopping=True
#         )

#         return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# # ✅ Example: Article paragraph input
# article_text = """
# In a major environmental policy shift, the European Union has voted to phase out the sale of all new gasoline and diesel cars by 2035. The legislation, passed by a wide majority in the European Parliament, aims to push automakers toward producing only zero-emission vehicles, primarily electric and hydrogen-powered. The policy is part of the EU’s broader Fit for 55 climate package, targeting a 55% reduction in greenhouse gas emissions by 2030. Car manufacturers like Volkswagen and Volvo have already announced full electric transition plans in line with the EU roadmap.
# """

# claim = Generate_claim.convert_paragraph_to_claim(article_text)
# fianl_claim = Generate_claim.convert_paragraph_to_claim(claim)

# print(f"\n📢 Extracted Claim:\n✅ {fianl_claim}")









import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


class GenerateClaim:
    @staticmethod
    def convert_paragraph_to_claim(text, max_length=60):
        prompt = f"Convert the following into one fact-checkable claim in 1 sentence:\n{text}"

        inputs = tokenizer([prompt], max_length=1024, truncation=True, return_tensors="pt")

        output_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=5,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @staticmethod
    def generate_claims_from_large_text(text, max_chunk_tokens=900):
        # Split text into paragraph-like chunks
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            combined = f"{current_chunk} {p}".strip()
            if len(tokenizer.encode(combined)) <= max_chunk_tokens:
                current_chunk = combined
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = p
        if current_chunk:
            chunks.append(current_chunk)

        # Generate claims from each chunk
        claims = []
        for chunk in chunks:
            claim = GenerateClaim.convert_paragraph_to_claim(chunk)
            if claim and claim not in claims:
                claims.append(claim)

        return claims


def load_article_text():
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data.get("title", "").strip()
    text = data.get("text", "").strip()

    if title and title not in text:
        article_text = f"{title}. {text}"
    else:
        article_text = text

    return article_text


if __name__ == "_main_":
    article_text = load_article_text()

    if not article_text:
        print("No article text found in data.json")
    else:
        claims = GenerateClaim.generate_claims_from_large_text(article_text)

        print("\nExtracted Claims:")
        for idx, claim in enumerate(claims, 1):
            print(f"{idx}. {claim}")
