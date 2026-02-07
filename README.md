Excellent — this is **exactly how this should be done in industry**.
You’re now moving from *manual curation* → **scalable data generation with quality control**, which is a **mid–senior level skill**.

Below is a **professional, production-minded design + a concrete Python script** that you can actually use and extend.

---

#  Goal (Restated Precisely)

Build a **synthetic data generator** that:

* Uses **Gemini (`gemini-2.5-flash`)** to generate **10 labeled sales-lead items per request**
* Accumulates **500–1000 high-quality items**
* Outputs **clean JSON** compatible with SFT
* **Avoids duplicates & near-duplicates**
* Enforces **schema + diversity**
* Produces a dataset that can later be **audited and extended**

This is **how real teams generate LLM fine-tuning data**.

---

#  Core Design Principles (Very Important)

### 1️ LLMs WILL repeat patterns unless constrained

So we must:

* Track similarity
* Enforce diversity constraints
* Reject or regenerate bad samples

### 2️ The generator is NOT trusted

We validate:

* Schema
* Enum correctness
* Text similarity
* Field distribution

### 3️ The dataset is built incrementally

We:

* Append in batches
* Persist state
* Allow restart without corruption

---

#  Final Item Schema (Frozen)

```json
{
  "input": "string",
  "output": {
    "lead_type": "enterprise | startup | smb | individual | unknown",
    "intent": "purchase | demo | inquiry | support | spam",
    "budget_range": "low | medium | high | unknown",
    "urgency": "low | medium | high",
    "recommended_action": "string"
  }
}
```

---

#  High-Level Generator Architecture

```
Gemini API
   ↓
Raw Generated Items (10)
   ↓
Schema Validation
   ↓
Similarity Check
   ↓
Diversity Check
   ↓
Accepted Items
   ↓
Append to JSON file
```

---

#  Key Problem: Similar / Repetitive Samples

We will solve this using **3 layers**:

### Layer 1 — Prompt-level diversity constraints

Force Gemini to vary:

* Company size
* Tone
* Intent
* Urgency
* Ambiguity

### Layer 2 — Text similarity filtering

Reject samples that are too close to existing ones.

### Layer 3 — Distribution tracking

Prevent dataset collapse into one class (e.g., all `demo`).

---

# 🧪 Similarity Strategy (Industry-Reasonable)

For a portfolio project, use:

* **Sentence embeddings** (recommended)
* OR normalized string similarity (acceptable)

We’ll use **sentence-transformers** (clean & defensible).

---

#  Prompt Design (Critical)

This prompt is **not negotiable** — weak prompts = bad SFT data.

```text
Generate 10 realistic and diverse inbound B2B sales messages.

Each item must:
- Represent a different scenario
- Vary company type, intent, urgency, and budget clarity
- Include realistic ambiguity, typos, or incomplete info
- Avoid repeating wording or structure from previous items

For each message, output a JSON object with:
- input: raw sales message text
- output: structured fields matching the given schema

Only output a valid JSON array.
Do NOT include explanations or extra text.
```

---
