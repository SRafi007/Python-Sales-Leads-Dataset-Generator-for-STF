import os
import json
import time
import hashlib
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
import torch
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, util

load_dotenv()


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

GENERATION_PROMPT = """
You are generating realistic inbound B2B sales inquiries for a SaaS company.

Rules:
- Output EXACTLY 10 distinct sales messages.
- Each message should be 1–5 sentences.
- Messages must sound human-written, informal or semi-formal.
- Include typos, incomplete sentences, or casual phrasing occasionally.
- Vary:
  - company size (startup, SMB, enterprise, individual)
  - intent (demo, purchase, inquiry, support, spam)
  - urgency (low, medium, high)
  - clarity of budget
- Some messages should be ambiguous or unclear.
- DO NOT number the messages.
- DO NOT include explanations.

Return as a JSON array of strings.
"""

def normalize(text: str) -> str:
    return " ".join(text.lower().split())


class DuplicateChecker:
    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self.texts = []
        self.embeddings = []

    def is_duplicate(self, text: str) -> bool:
        if not self.texts:
            return False

        emb = embedder.encode(text, convert_to_tensor=True)
        # Stack embeddings list into a tensor
        embeddings_tensor = torch.stack(self.embeddings)
        scores = util.cos_sim(emb, embeddings_tensor)
        return scores.max().item() >= self.threshold

    def add(self, text: str):
        emb = embedder.encode(text, convert_to_tensor=True)
        self.texts.append(text)
        self.embeddings.append(emb)


def generate_batch(model_name: str) -> List[str]:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=GENERATION_PROMPT),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        raw = response.text
        messages = json.loads(raw)
        
        if isinstance(messages, list):
             # Ensure we return strings
             return [str(m).strip() for m in messages]
        return []
    except Exception as e:
        print(f"Error generating batch with {model_name}: {e}")
        return []


def generate_dataset(
    output_file="raw_sales_messages.json",
    sleep_time=8.0  # Enforce ~7.5 calls/min limit
):
    checker = DuplicateChecker()
    dataset = []

    # Resume if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            dataset = json.load(f)
        for item in dataset:
            checker.add(item)
    
    # Models to rotate through
    models = ['gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']
    batches_per_model = 19
    
    total_expected = len(dataset) + (len(models) * batches_per_model * 10) # Approx
    pbar = tqdm(total=total_expected, initial=len(dataset))

    for model in models:
        print(f"\nSwitching to model: {model}")
        for i in range(batches_per_model):
            batch = generate_batch(model)
            
            # If batch is empty due to error, we should still sleep to avoid hammering
            if not batch:
                 time.sleep(sleep_time)
                 continue

            for msg in batch:
                norm = normalize(msg)
                if checker.is_duplicate(norm):
                    continue

                checker.add(norm)
                dataset.append(msg)
                pbar.update(1)

            # Build incremental save
            with open(output_file, "w") as f:
                json.dump(dataset, f, indent=2)
            
            time.sleep(sleep_time)

    pbar.close()
    print(f"Saved {len(dataset)} samples to {output_file}")


if __name__ == "__main__":
    generate_dataset(
        output_file="raw_sales_messages.json"
    )


