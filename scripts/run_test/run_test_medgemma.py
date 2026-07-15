import argparse
import os
import time

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

SYSTEM_PROMPT = "You are a general practitioner working in a clinic."

USER_PROMPT = """
Answer the following question from a patient.
Provide a CONCISE and SPECIFIC answer in two to three sentences.
Question: {question}
"""


def generate_answer_medgemma(model, processor, question, retries):
    user_prompt_question = USER_PROMPT.format(question=question)
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}, {"role": "user", "content": [{"type": "text", "text": user_prompt_question}]}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    for attempt in range(1, retries + 1):
        try:
            with torch.inference_mode():
                raw = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                ans = processor.decode(raw[0][input_len:], skip_special_token=True).replace("<end_of_turn>", "").strip()

            if ans:
                return ans

            print(f"    [empty output attempt {attempt}/{retries}]")
            if attempt == retries:
                return None
            time.sleep(2**attempt)

        except Exception as e:
            print(f"    [API error attempt {attempt}/{retries}]: {e}")
            if attempt == retries:
                return None
            time.sleep(2**attempt)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    load_dotenv(".env")
    login(token=os.environ["HF_TOKEN"])

    qa_df = pd.read_csv(args.input)

    model_id = "google/medgemma-4b-it"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

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

    medgemma_qa = []
    n_failed = 0
    n_success = 0

    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Generating Answers"):
        a = generate_answer_medgemma(model, processor, row["question"], 3)

        if a is None:
            n_failed += 1
            continue

        record = {"question": row["question"], "answer": row["answer"], "generated_answer": a}
        medgemma_qa.append(record)
        n_success += 1

        if n_failed + n_success == num_samples:
            break

    tqdm.write(f"Processed {len(medgemma_qa)} chains. Success: {n_success}, Failed: {n_failed}")

    output_path = "medgemma_results.csv"
    pd.DataFrame(medgemma_qa).to_csv(output_path, index=False)
    print(f"Saved {len(medgemma_qa)} results to {output_path}")


if __name__ == "__main__":
    main()
