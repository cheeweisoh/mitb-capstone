import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from agentic_rag_workflow import AgenticRagWorkflow
from generator_adapters import ADAPTER_BUILDERS
from llamaindex_retriever import (DEFAULT_FUSION_CANDIDATES, DEFAULT_INDEX_DIR,
                                  DEFAULT_RRF_K, LlamaIndexRagRetriever)
from rag_prompts import (SYSTEM_PROMPT, build_user_prompt, load_qa_rows,
                         make_output_record, sleep_before_retry)

DEFAULT_INPUT_PATH = Path("dataset/guidelines/guidelines_qa_pairs.csv")
DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 7000
DEFAULT_MAX_ITERATIONS = 2
DEFAULT_BATCH_SIZE = 8

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
        return json.dumps({"passed": True, "reasoning": "verification skipped (--no-verify)", "reformulated_query": None})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate agentic RAG answers for guideline QA pairs.")
    parser.add_argument("--model", required=True, choices=sorted(ADAPTER_BUILDERS.keys()), help="Generator backend to use.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help=f"Input QA CSV path. Default: {DEFAULT_INPUT_PATH}")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path. Default: dataset/rag/rag_<model>_guidelines_qa.csv")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_DIR, help=f"LlamaIndex FAISS storage directory. Default: {DEFAULT_INDEX_DIR}")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of chunks to retrieve per question. Default: {DEFAULT_TOP_K}")
    parser.add_argument("--rows", type=int, default=None, help="Limit generation to the first N rows.")
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS, help=f"Maximum context characters sent to the LLM. Default: {DEFAULT_MAX_CONTEXT_CHARS}")
    parser.add_argument("--hybrid", action="store_true", help="Fuse dense FAISS and BM25 rankings before sending top-k chunks to the LLM.")
    parser.add_argument("--fusion-candidates", type=int, default=DEFAULT_FUSION_CANDIDATES, help=f"Number of dense and BM25 candidates to fuse. Default: {DEFAULT_FUSION_CANDIDATES}")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help=f"Reciprocal rank fusion smoothing constant. Default: {DEFAULT_RRF_K}")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help=f"Max retrieve/draft/verify loop iterations. Default: {DEFAULT_MAX_ITERATIONS}")
    parser.add_argument("--no-verify", action="store_true", help="Skip the verify/re-retrieve loop entirely (single-pass retrieve+draft, no verifier calls).")
    parser.add_argument("--retries", type=int, default=3, help="Retries if a row's final answer comes back empty. Default: 3")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Rows per generate() call when the generator supports batching (requires --no-verify and --max-iterations 1). Default: {DEFAULT_BATCH_SIZE}")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if args.rows is not None and args.rows < 1:
        raise SystemExit("--rows must be at least 1")
    if args.max_context_chars < 1000:
        raise SystemExit("--max-context-chars must be at least 1000")
    if args.fusion_candidates < 1:
        raise SystemExit("--fusion-candidates must be at least 1")
    if args.rrf_k < 1:
        raise SystemExit("--rrf-k must be at least 1")
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be at least 1")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")


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


async def generate_batch(
    retriever: LlamaIndexRagRetriever,
    generator: Any,
    questions: list[str],
    top_k: int,
    max_context_chars: int,
    retries: int,
) -> tuple[list[list[dict]], list[str | None]]:
    """Bypasses the per-row agentic workflow for the retrieve-once/draft-once
    case (--no-verify --max-iterations 1) so the whole batch goes through the
    generator's model.generate() in a single call instead of one at a time."""
    chunks_per_question = [retriever.retrieve(q, top_k=top_k) for q in questions]
    prompts = [build_user_prompt(q, chunks, max_context_chars) for q, chunks in zip(questions, chunks_per_question)]

    answers: list[str | None] = [None] * len(questions)
    remaining = list(range(len(questions)))
    for attempt in range(1, retries + 1):
        if not remaining:
            break
        results = await generator.agenerate_batch(SYSTEM_PROMPT, [prompts[i] for i in remaining])
        still_empty = []
        for i, answer in zip(remaining, results):
            if answer:
                answers[i] = answer
            else:
                still_empty.append(i)
        remaining = still_empty
        if remaining and attempt < retries:
            print(f"    [{len(remaining)} empty output(s) in batch, retry {attempt}/{retries}]")
            sleep_before_retry(attempt)

    return chunks_per_question, answers


async def main_async() -> None:
    args = parse_args()
    validate_args(args)
    load_dotenv(".env")

    if args.model in HF_MODELS:
        login(token=os.environ["HF_TOKEN"])

    output_path = args.output or Path("dataset/rag") / OUTPUT_FILENAMES[args.model]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qa_df = load_qa_rows(args.input, rows=None)

    done_questions: set[str] = set()
    write_header = True
    if output_path.exists():
        existing = pd.read_csv(output_path)
        done_questions = set(existing["question"].tolist())
        write_header = False
        print(f"Resuming: {len(done_questions)} rows already in {output_path}, skipping them.")

    pending = qa_df[~qa_df["question"].isin(done_questions)]
    if args.rows is not None:
        pending = pending.head(args.rows)

    if pending.empty:
        print("All rows already processed.")
        return

    retriever = LlamaIndexRagRetriever(
        index_dir=args.index,
        hybrid=args.hybrid,
        fusion_candidates=args.fusion_candidates,
        rrf_k=args.rrf_k,
    )
    generator = ADAPTER_BUILDERS[args.model]()

    n_success = 0
    n_failed = 0

    if args.no_verify and args.max_iterations == 1 and hasattr(generator, "agenerate_batch"):
        with tqdm(total=len(pending), desc=f"Generating RAG answers ({args.model}, batched)") as pbar:
            for start in range(0, len(pending), args.batch_size):
                batch = pending.iloc[start : start + args.batch_size]
                chunks_per_row, answers = await generate_batch(
                    retriever,
                    generator,
                    batch["question"].tolist(),
                    top_k=args.top_k,
                    max_context_chars=args.max_context_chars,
                    retries=args.retries,
                )
                for (_, row), chunks, answer in zip(batch.iterrows(), chunks_per_row, answers):
                    if answer is None:
                        n_failed += 1
                        pbar.update(1)
                        continue
                    record = make_output_record(
                        row,
                        answer,
                        chunks,
                        args.max_context_chars,
                        verification_passed=None,
                        verification_reasoning="verification skipped (--no-verify)",
                        retrieval_iterations=1,
                        retrieval_queries_used=[row["question"]],
                    )
                    pd.DataFrame([record]).to_csv(output_path, mode="a", header=write_header, index=False)
                    write_header = False
                    n_success += 1
                    pbar.update(1)
    else:
        verifier_llm = NoVerifyLLM() if args.no_verify else generator
        workflow = AgenticRagWorkflow(
            retriever=retriever,
            generator=generator,
            verifier_llm=verifier_llm,
            max_iterations=args.max_iterations,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            timeout=None,
        )
        for _, row in tqdm(pending.iterrows(), total=len(pending), desc=f"Generating RAG answers ({args.model})"):
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
                retrieval_iterations=result.retrieval_iterations,
                retrieval_queries_used=result.retrieval_queries_used,
            )
            pd.DataFrame([record]).to_csv(output_path, mode="a", header=write_header, index=False)
            write_header = False
            n_success += 1

    print(f"Saved {n_success} new results to {output_path}. Failed this run: {n_failed}. Total in file: {len(done_questions) + n_success}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
