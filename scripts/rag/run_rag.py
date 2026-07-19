import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from agentic_rag_workflow import DEFAULT_MAX_ITERATIONS, AgenticRagWorkflow
from generator_adapters import ADAPTER_BUILDERS
from llamaindex_retriever import DEFAULT_INDEX_DIR, DEFAULT_MIN_SCORE, LlamaIndexRagRetriever
from rag_prompts import load_qa_rows, make_output_record, sleep_before_retry

DEFAULT_INPUT_PATH = Path("dataset/guidelines/guidelines_qa_pairs.csv")
DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 7000

OUTPUT_FILENAMES = {
    "chatgpt": "rag_chatgpt_guidelines_qa.csv",
    "meditron": "rag_meditron_guidelines_qa.csv",
    "medalpaca": "rag_medalpaca_guidelines_qa.csv",
    "openbio": "rag_openbio_guidelines_qa.csv",
    "medgemma": "rag_medgemma_guidelines_qa.csv",
    "iimedical": "rag_iimedical_guidelines_qa.csv",
}

# Models loaded via huggingface_hub (i.e. everything except the OpenAI-backed chatgpt).
HF_MODELS = {"meditron", "medalpaca", "openbio", "medgemma", "iimedical"}


class NoVerifyLLM:
    """Stand-in verifier used with --no-verify: always reports 'passed' without
    making any API call, so the workflow's verify step is a no-op."""

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"passed": True, "reasoning": "verification skipped (--no-verify)"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate agentic RAG answers for guideline QA pairs.")
    parser.add_argument("--model", required=True, choices=sorted(ADAPTER_BUILDERS.keys()), help="Generator backend to use.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help=f"Input QA CSV path. Default: {DEFAULT_INPUT_PATH}")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path. Default: dataset/rag/rag_<model>_guidelines_qa.csv")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_DIR, help=f"LlamaIndex FAISS storage directory. Default: {DEFAULT_INDEX_DIR}")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of chunks to retrieve per question. Default: {DEFAULT_TOP_K}")
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Cosine-similarity floor; retrieved chunks scoring below this are dropped instead of forced into context. Default: {DEFAULT_MIN_SCORE}",
    )
    parser.add_argument("--rows", type=int, default=None, help="Limit generation to the first N rows.")
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS, help=f"Maximum context characters sent to the LLM. Default: {DEFAULT_MAX_CONTEXT_CHARS}")
    parser.add_argument("--no-verify", action="store_true", help="Skip the verify step entirely (retrieve+draft only, no verifier call).")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help=f"Max draft->verify passes per question before falling back to the last draft. Default: {DEFAULT_MAX_ITERATIONS}")
    parser.add_argument("--retries", type=int, default=3, help="Retries if a row's final answer comes back empty. Default: 3")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if not 0 <= args.min_score <= 1:
        raise SystemExit("--min-score must be between 0 and 1")
    if args.rows is not None and args.rows < 1:
        raise SystemExit("--rows must be at least 1")
    if args.max_context_chars < 1000:
        raise SystemExit("--max-context-chars must be at least 1000")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be at least 1")


async def generate_for_row(workflow: AgenticRagWorkflow, question: str, retries: int):
    for attempt in range(1, retries + 1):
        result = await workflow.run(question=question)
        if result.final_answer.strip():
            return result
        print(f"    [empty output attempt {attempt}/{retries}]")
        if attempt == retries:
            return None
        sleep_before_retry(attempt)
    return None


async def main_async() -> None:
    args = parse_args()
    validate_args(args)
    load_dotenv(".env")

    if args.model in HF_MODELS:
        login(token=os.environ["HF_TOKEN"])

    output_path = args.output or Path("dataset/rag") / OUTPUT_FILENAMES[args.model]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qa_df = load_qa_rows(args.input, rows=args.rows)

    retriever = LlamaIndexRagRetriever(index_dir=args.index, min_score=args.min_score)
    generator = ADAPTER_BUILDERS[args.model]()
    verifier_llm = NoVerifyLLM() if args.no_verify else generator
    workflow = AgenticRagWorkflow(
        retriever=retriever,
        generator=generator,
        verifier_llm=verifier_llm,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_iterations=args.max_iterations,
        timeout=None,
    )

    n_success = 0
    n_failed = 0
    write_header = True
    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc=f"Generating RAG answers ({args.model})"):
        result = await generate_for_row(workflow, row["question"], retries=args.retries)
        if result is None:
            n_failed += 1
            continue
        record = make_output_record(
            row,
            result.final_answer,
            result.retrieved_chunks,
            args.max_context_chars,
            verification_passed=result.verification_passed,
            verification_reasoning=result.verification_reasoning,
            verification_iterations=result.verification_iterations,
        )
        pd.DataFrame([record]).to_csv(output_path, mode="w" if write_header else "a", header=write_header, index=False)
        write_header = False
        n_success += 1

    print(f"Saved {n_success} results to {output_path}. Failed: {n_failed}.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
