import json
import os
import time
from tqdm import tqdm
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# =====================
# CONFIGURATION
# =====================
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

TARGET_SAMPLES = 800
BATCH_SIZE = 10
OUTPUT_FILE = "synthetic_sales_leads.json"

SIMILARITY_THRESHOLD = 0.90
SLEEP_BETWEEN_CALLS = 2

LEAD_TYPES = ["enterprise", "startup", "smb", "individual", "unknown"]
INTENTS = ["purchase", "demo", "inquiry", "support", "spam"]
BUDGETS = ["low", "medium", "high", "unknown"]
URGENCY = ["low", "medium", "high"]

# =====================
# INIT
# =====================
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =====================
# LOAD EXISTING DATA
# =====================
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        dataset = json.load(f)
else:
    dataset = []

existing_texts = [item["input"] for item in dataset]
existing_embeddings = embedder.encode(existing_texts, convert_to_tensor=True) if existing_texts else None

# =====================
# VALIDATION FUNCTIONS
# =====================
def validate_schema(item):
    try:
        o = item["output"]
        return (
            o["lead_type"] in LEAD_TYPES and
            o["intent"] in INTENTS and
            o["budget_range"] in BUDGETS and
            o["urgency"] in URGENCY and
            isinstance(o["recommended_action"], str)
        )
    except Exception:
        return False


def is_similar(text, existing_embeds):
    if existing_embeds is None:
        return False
    emb = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(emb, existing_embeds)
    return scores.max().item() > SIMILARITY_THRESHOLD


# =====================
# PROMPT TEMPLATE
# =====================
PROMPT = """
Generate 10 realistic and diverse inbound B2B sales messages.

Requirements:
- Each item must be different in tone, company type, urgency, and intent
- Avoid repeating phrasing or structure
- Include ambiguity, informal language, or typos where appropriate
- Mix strong buying signals, weak signals, support requests, and spam

Output format:
A valid JSON array of objects, where each object has:
- "input": string
- "output": {
    "lead_type": one of [enterprise, startup, smb, individual, unknown],
    "intent": one of [purchase, demo, inquiry, support, spam],
    "budget_range": one of [low, medium, high, unknown],
    "urgency": one of [low, medium, high],
    "recommended_action": string
}

Output ONLY the JSON array. No explanations.
"""

# =====================
# MAIN GENERATION LOOP
# =====================
with tqdm(total=TARGET_SAMPLES, initial=len(dataset)) as pbar:
    while len(dataset) < TARGET_SAMPLES:
        response = model.generate_content(PROMPT)
        try:
            batch = json.loads(response.text)
        except json.JSONDecodeError:
            continue

        accepted = []

        for item in batch:
            if not validate_schema(item):
                continue
            if is_similar(item["input"], existing_embeddings):
                continue
            accepted.append(item)

        if not accepted:
            continue

        dataset.extend(accepted)

        # Update embeddings
        new_texts = [x["input"] for x in accepted]
        new_embeds = embedder.encode(new_texts, convert_to_tensor=True)

        if existing_embeddings is None:
            existing_embeddings = new_embeds
        else:
            existing_embeddings = util.cat((existing_embeddings, new_embeds), dim=0)

        # Save incrementally
        with open(OUTPUT_FILE, "w") as f:
            json.dump(dataset, f, indent=2)

        pbar.update(len(accepted))
        time.sleep(SLEEP_BETWEEN_CALLS)

print(f"Dataset generation complete: {len(dataset)} samples saved.")
