import argparse
import os
import time

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are a general practitioner working in a clinic."

USER_PROMPT = """
Answer the following question from a patient.
Provide a CONCISE and SPECIFIC answer in two to three sentences.
Question: {question}
"""


def clean_generated_answer(text):
    if not text:
        return None

    cleaned = text.strip()

    stop_markers = [
        "### Answer",
        "Assistant:",
        "Question:",
        "Answer the following question from a patient.",
    ]
    for marker in stop_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()

    # Keep only the first paragraph and cap to a concise length.
    cleaned = cleaned.split("\n\n")[0].strip()
    if not cleaned:
        return None

    words = cleaned.split()
    if len(words) > 70:
        cleaned = " ".join(words[:70]).rstrip(" ,;:") + "."

    return cleaned


def generate_answer_meditron(model, tokenizer, question, retries=2):
    prompt = f"System: {SYSTEM_PROMPT}\n" f"Patient question: {question}\n" "Doctor answer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    for attempt in range(1, retries + 1):
        try:
            with torch.inference_mode():
                raw = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            ans = tokenizer.decode(raw[0][input_len:], skip_special_tokens=True)
            ans = clean_generated_answer(ans)
            if ans:
                return ans

            print(f"    [empty output attempt {attempt}/{retries}]")
            if attempt == retries:
                return None
            time.sleep(1)

        except Exception as e:
            print(f"    [generation error attempt {attempt}/{retries}]: {e}")
            if attempt == retries:
                return None
            time.sleep(1)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    load_dotenv(".env")
    login(token=os.environ["HF_TOKEN"])

    qa_df = pd.read_csv(args.input)

    model_id = "epfl-llm/meditron-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    raw_limit = input("How many rows to process? (press Enter for all): ").strip()
    if raw_limit:
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError("Row count must be an integer or empty.") from exc
        if limit < 1:
            raise ValueError("Row count must be at least 1.")
        qa_df = qa_df.head(limit)

    num_samples = len(qa_df)

    meditron_qa = []
    n_failed = 0
    n_success = 0

    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Generating Answers"):
        a = generate_answer_meditron(model, tokenizer, row["question"], retries=2)

        if a is None:
            n_failed += 1
            continue

        record = {"question": row["question"], "answer": row["answer"], "generated_answer": a}
        meditron_qa.append(record)
        n_success += 1

        if n_failed + n_success == num_samples:
            break

    tqdm.write(f"Processed {len(meditron_qa)} chains. Success: {n_success}, Failed: {n_failed}")

    output_path = "meditron_results.csv"
    pd.DataFrame(meditron_qa).to_csv(output_path, index=False)
    print(f"Saved {len(meditron_qa)} results to {output_path}")


if __name__ == "__main__":
    main()
