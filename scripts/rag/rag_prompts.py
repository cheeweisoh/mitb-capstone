import json
import time
from typing import Any

import pandas as pd

SYSTEM_PROMPT = (
    "You are a general practitioner working in a clinic. "
    "Use the provided clinical guideline excerpts only when they are directly relevant. "
    "Return the final patient-facing answer only."
)

USER_PROMPT = """\
Answer the following question from a patient.
Provide a concise and specific answer in two to three sentences.
First decide whether the excerpts directly answer the question.
Use an excerpt only if its patient group, condition, and intervention match the question.
Do not apply a recommendation from one indication to another.
Do not add dates, targets, follow-up intervals, organizations, or thresholds unless they are explicitly stated in the excerpts.
If the excerpts are partially relevant, synthesize the relevant practical advice instead of saying the answer is not covered.
If the excerpts are unrelated, say they do not directly cover the question and give safe general advice.
Do not explain your reasoning, mention the retrieval process, or summarize the excerpts.
Include citation labels only for recommendations you actually use, such as [C1] or [C2].

Question:
{question}

Guideline excerpts:
{context}
"""

VERIFY_SYSTEM_PROMPT = (
    "You are a clinical safety reviewer checking a draft answer against the guideline excerpts "
    "it was supposedly based on. You are strict about groundedness and patient safety, but you "
    "also recognize when an answer legitimately falls back to safe general advice because the "
    "excerpts do not cover the question."
)

VERIFY_USER_PROMPT = """\
Question from a patient:
{question}

Guideline excerpts the draft answer had access to:
{context}

Draft answer to review:
{draft_answer}

Judge the draft answer on three points:
1. Groundedness: does it only state dates, targets, follow-up intervals, organizations, or thresholds
   that are explicitly present in the excerpts (or none at all)? It must not apply a recommendation
   from one condition/indication to a different one implied by the question.
2. Safety: does it avoid missing an escalation/red-flag that the excerpts call for, and avoid giving
   dangerous or contraindicated advice?
3. Responsiveness: does it actually answer the patient's question rather than being a non-answer?

Respond with a single JSON object and nothing else, in this exact shape:
{{"passed": true or false, "reasoning": "<one sentence explaining the verdict>"}}
"""


def format_context(results: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    remaining_chars = max_chars

    for rank, result in enumerate(results, start=1):
        label = f"C{rank}"
        header = f"[{label}] {result['source_file']} " f"p{result['start_page']}-{result['end_page']} | {result['heading']}"
        text = str(result["text"]).strip()
        block_budget = remaining_chars - len(header) - 2
        if block_budget <= 0:
            break
        if len(text) > block_budget:
            text = text[: block_budget - 3].rstrip() + "..."
        block = f"{header}\n{text}"
        blocks.append(block)
        remaining_chars -= len(block) + 2

    return "\n\n".join(blocks)


def build_user_prompt(question: str, retrieved_chunks: list[dict[str, Any]], max_context_chars: int) -> str:
    return USER_PROMPT.format(
        question=question,
        context=format_context(retrieved_chunks, max_chars=max_context_chars),
    )


def build_verify_user_prompt(question: str, retrieved_chunks: list[dict[str, Any]], draft_answer: str, max_context_chars: int) -> str:
    return VERIFY_USER_PROMPT.format(
        question=question,
        context=format_context(retrieved_chunks, max_chars=max_context_chars),
        draft_answer=draft_answer,
    )


def citations_json(retrieved_chunks: list[dict[str, Any]]) -> str:
    citations = []
    for rank, result in enumerate(retrieved_chunks, start=1):
        citations.append(
            {
                "label": f"C{rank}",
                "chunk_id": result["chunk_id"],
                "source_file": result["source_file"],
                "start_page": result["start_page"],
                "end_page": result["end_page"],
                "heading": result["heading"],
                "score": result["score"],
            }
        )
    return json.dumps(citations, ensure_ascii=False)


def chunk_ids_json(retrieved_chunks: list[dict[str, Any]]) -> str:
    return json.dumps([chunk["chunk_id"] for chunk in retrieved_chunks], ensure_ascii=False)


def scores_json(retrieved_chunks: list[dict[str, Any]]) -> str:
    return json.dumps([chunk["score"] for chunk in retrieved_chunks])


def make_output_record(
    row: pd.Series,
    generated_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    max_context_chars: int,
    verification_passed: bool | None = None,
    verification_reasoning: str = "",
) -> dict[str, Any]:
    record = row.to_dict()
    record["generated_answer"] = generated_answer
    record["retrieved_context"] = format_context(
        retrieved_chunks,
        max_chars=max_context_chars,
    )
    record["citations"] = citations_json(retrieved_chunks)
    record["top_k_chunk_ids"] = chunk_ids_json(retrieved_chunks)
    record["top_k_scores"] = scores_json(retrieved_chunks)
    record["verification_passed"] = verification_passed
    record["verification_reasoning"] = verification_reasoning
    return record


def load_qa_rows(path, rows: int | None) -> pd.DataFrame:
    qa_df = pd.read_csv(path)
    required_columns = {"question", "answer"}
    missing_columns = required_columns - set(qa_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")
    if rows is not None:
        qa_df = qa_df.head(rows)
    return qa_df


def sleep_before_retry(attempt: int) -> None:
    time.sleep(2**attempt)
