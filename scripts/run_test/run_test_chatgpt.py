import argparse
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = "You are a general practitioner working in a clinic."

USER_PROMPT = """
Answer the following question from a patient.
Provide a CONCISE and SPECIFIC answer in two to three sentences.
Question: {question}
"""


def generate_answer_chatgpt(client, question, retries):
    user_prompt_question = USER_PROMPT.format(question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_question},
    ]

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=256,
            )
            ans = response.choices[0].message.content if response.choices else None

            if ans:
                ans = ans.strip()
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

    qa_df = pd.read_csv(args.input)

    client = OpenAI()

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

    chatgpt_qa = []
    n_failed = 0
    n_success = 0

    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Generating Answers"):
        a = generate_answer_chatgpt(client, row["question"], 3)

        if a is None:
            n_failed += 1
            continue

        record = {"question": row["question"], "answer": row["answer"], "generated_answer": a}
        chatgpt_qa.append(record)
        n_success += 1

        if n_failed + n_success == num_samples:
            break

    tqdm.write(f"Processed {len(chatgpt_qa)} chains. Success: {n_success}, Failed: {n_failed}")

    output_path = "chatgpt_results.csv"
    pd.DataFrame(chatgpt_qa).to_csv(output_path, index=False)
    print(f"Saved {len(chatgpt_qa)} results to {output_path}")


if __name__ == "__main__":
    main()
