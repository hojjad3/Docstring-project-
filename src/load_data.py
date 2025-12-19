import json
import os
import time

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Error: GROQ_API_KEY not found in .env file!")
    exit()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

MODEL_NAME = "llama-3.3-70b-versatile"

output_file = "groq_dpo_dataset.jsonl"


def generate_with_groq(code_snippet, retries=3):
    system_prompt = """
    You are an expert Python engineer.
    You must output strictly valid JSON.
    Do not add any markdown formatting like ```json ... ```.
    Just raw JSON.
    """

    user_prompt = f"""
    Task:
    1. "chosen": Write a professional, high-quality Google-style docstring for the code below.
    2. "rejected": Write a poor, vague, or overly verbose docstring.

    Input Code:
    {code_snippet}

    Output Format:
    {{"chosen": "...", "rejected": "..."}}
    """

    wait_time = 2
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            content = completion.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            print(f" Error (Attempt {attempt + 1}): {e}")
            if "429" in str(e):
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                return None
    return None


ds = load_dataset(
    "calum/the-stack-smol-python-docstrings", split="train", streaming=True
)

print(f" Starting Pipeline with Groq ({MODEL_NAME})...")

with open(output_file, "a", encoding="utf-8") as f:
    counter = 0

    for row in ds:
        if counter >= 500:
            break

        code = row.get("body_without_docstring")
        if not code or len(code) < 50:
            continue

        data = generate_with_groq(code)

        if data:
            entry = {
                "prompt": f"Write a docstring for:\n{code}",
                "chosen": data.get("chosen"),
                "rejected": data.get("rejected"),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

            counter += 1
            print(f"Generated: {counter}")

        time.sleep(20)

print("Done! Dataset is ready.")
