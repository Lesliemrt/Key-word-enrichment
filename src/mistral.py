from mistralai import Mistral
import json
import re
import time
from tqdm import tqdm
from src.utils.config import API_KEY

client = Mistral(api_key=API_KEY)

def call_mistral_with_retry(messages, max_retries=5):
    delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model="mistral-medium-latest",
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )
            return response

        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit → sleeping {delay}s...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                raise e

    raise RuntimeError("Too many retries")


def batch_extract(texts, batch_size=20, sleep_between=0.5):
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        formatted = "\n".join(
            [f"{idx}. {t}" for idx, t in enumerate(batch)]
        )
        prompt = f"""
You are a biology expert.

For EACH text, extract taxons.

Return ONLY a JSON ARRAY (not object).
The array must contain EXACTLY {len(batch)} elements.
Each element:
- list of taxon names
- or null

Example:
[
  ["Homo sapiens"],
  null,
  ["Canis lupus"]
]

Texts:
{formatted}
"""
        messages = [{"role": "user", "content": prompt}]

        response = call_mistral_with_retry(messages)

        data = json.loads(response.choices[0].message.content)

        if not isinstance(data, list):
            data = [None] * len(batch)

        if len(data) < len(batch):
            data += [None] * (len(batch) - len(data))

        if len(data) > len(batch):
            data = data[:len(batch)]

        results.extend(data)

        time.sleep(sleep_between)

    return results
