import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

sys.path.insert(0, str(Path(__file__).parents[1]))
from engine import (build_category_guardrail_context,
                    category_guardrail_trigger_metadata,
                    guardrail_category_prompt, normalize_guardrail_categories)

SYSTEM_PROMPT = "You are a general practitioner working in a clinic."

USER_PROMPT = """
Answer the following question from a patient.
Provide a CONCISE and SPECIFIC answer in two to three sentences.
Question: {question}
"""


def classify_guardrail_category(pipeline, question):
    messages = [{"role": "user", "content": guardrail_category_prompt(question)}]
    inputs = pipeline.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(pipeline.model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        raw = pipeline.model.generate(
            **inputs,
            max_new_tokens=20,
            eos_token_id=pipeline.tokenizer.eos_token_id,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            do_sample=False,
        )
    text = pipeline.tokenizer.decode(raw[0][input_len:], skip_special_tokens=True)
    return normalize_guardrail_categories(text)


def generate_answer_openbio(pipeline, question, guardrail_category, retries):
    user_prompt_question = USER_PROMPT.format(question=question)
    system_prompt = SYSTEM_PROMPT + build_category_guardrail_context(guardrail_category, question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_question},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    inputs = pipeline.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(pipeline.model.device)
    input_len = inputs["input_ids"].shape[-1]

    for attempt in range(1, retries + 1):
        try:
            with torch.inference_mode():
                raw = pipeline.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    pad_token_id=pipeline.tokenizer.eos_token_id,
                    do_sample=False,
                )
                ans = pipeline.tokenizer.decode(raw[0][input_len:], skip_special_tokens=True)

            if ans:
                ans = ans.strip()
                if ans:
                    return ans

            print(f"    [empty output attempt {attempt}/{retries}]")
            if attempt == retries:
                return None
            torch.cuda.empty_cache()
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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    LLAMA3_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] | trim }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"""
    model_id = "aaditya/Llama3-OpenBioLLM-8B"
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    model_pipeline = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"quantization_config": bnb_config},
        device_map="auto",
    )
    model_pipeline.tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

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

    openbio_qa = []
    n_failed = 0
    n_success = 0

    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Generating Answers"):
        guardrail_category = classify_guardrail_category(model_pipeline, row["question"])
        a = generate_answer_openbio(model_pipeline, row["question"], guardrail_category, 3)

        if a is None:
            n_failed += 1
            continue

        record = {
            "question": row["question"],
            "answer": row["answer"],
            "generated_answer": a,
            **category_guardrail_trigger_metadata(guardrail_category, row["question"]),
        }
        openbio_qa.append(record)
        n_success += 1

        if n_failed + n_success == num_samples:
            break

    tqdm.write(f"Processed {len(openbio_qa)} chains. Success: {n_success}, Failed: {n_failed}")

    output_path = "openbio_results.csv"
    pd.DataFrame(openbio_qa).to_csv(output_path, index=False)
    print(f"Saved {len(openbio_qa)} results to {output_path}")


if __name__ == "__main__":
    main()
