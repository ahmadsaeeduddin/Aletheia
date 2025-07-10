import json
from groq import Groq

# === User input ===
claim = input("Enter the claim: ").strip()
evidence = input("Enter the evidence: ").strip()

# === Groq API setup ===
groq_key = "API_KEY"
client = Groq(api_key=groq_key)

# === Step 1: Extract relevant evidence ===
extract_messages = [
    {"role": "system", "content": "You are strictly prohibited from generating any text of your own."},
    {"role": "system", "content": "Limit your answer to 50 words."},
    {"role": "system", "content": "Your task is to extract a part of the given text that directly answers the given question. The extracted information should contain a conclusive answer to the question and should be either positive or negative in relation to the question. It should also be concise without irrelevant words"},
    {"role": "user", "content": f"Question: {claim}"},
    {"role": "user", "content": "Pay more attention to the later parts of the evidences, as the initial sentences are only the introduction"},
    {"role": "user", "content": f"Evidence: {evidence}"},
    {"role": "system", "content": "You do not need to explain your answer. Only return the extracted sentence and follow the system instructions strictly"},
]

extract_response = client.chat.completions.create(
    model="llama3-8b-8192",  # ✅ Use this model instead of the deprecated Mixtral
    messages=extract_messages,
    temperature=0.0
)
relevant_sentence = extract_response.choices[0].message.content.strip()

# === Step 2: Classification ===
classification_messages = [
    {"role": "system", "content": "You are tasked with classifying a given claim based on provided statements into one of the following four categories:"},
    {"role": "system", "content": "1. Supported: If there is sufficient evidence indicating that the claim is legitimate, classify it as Supported."},
    {"role": "system", "content": "2. Refuted: If there is any evidence contradicting the claim, classify it as Refuted."},
    {"role": "system", "content": "3. Not Enough Evidence: If you cannot find any conclusive factual evidence either supporting or refuting the claim, classify it as Not Enough Evidence."},
    {"role": "system", "content": "4. Conflicting Evidence/Cherrypicking: If there is factual evidence both supporting and refuting the claim, classify it as Conflicting Evidence/Cherrypicking."},
    {"role": "system", "content": "Examples:"},
    {"role": "system", "content": "Example 1:\nClaim: The new drug is effective in treating diabetes.\nStatements: ['Clinical trials have shown no significant reduction in blood sugar levels among patients.', 'Several patients reported no change in their blood sugar levels after using the drug', 'The drug has not been widely tested in clinical trials.']\nFinal Classification: Refuted"},
    {"role": "system", "content": "Example 2:\nClaim: The new drug is effective in treating diabetes.\nStatements: ['Clinical trials have shown a significant reduction in blood sugar levels among patients.', 'Many patients reported positive results in managing their blood sugar levels.', 'Experts in the field have praised the effectiveness of the drug.']\nFinal Classification: Supported"},
    {"role": "user", "content": f"Claim: {claim}"},
    {"role": "user", "content": f"Statements: ['{relevant_sentence}']"},
    {"role": "system", "content": "Pick only one from ['Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherry-picking']"},
    {"role": "system", "content": "Do not print anything other than the final classification."},
]

classification_response = client.chat.completions.create(
    model="llama3-8b-8192",  # ✅ Use supported model
    messages=classification_messages,
    temperature=0.0
)
final_label = classification_response.choices[0].message.content.strip()

# === Output ===
print("\n--- Result ---")
print("Claim:", claim)
print("Extracted Evidence:", relevant_sentence)
print("Final Classification:", final_label)
