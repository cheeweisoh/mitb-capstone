import json
import time
from typing import Any

import pandas as pd

SYSTEM_PROMPT = (
    "You are a general practitioner working in a clinic. "
    "Use the provided clinical guideline excerpts only when they are directly relevant. "
    "Return the final patient-facing answer only."
)

ROUTE_SYSTEM_PROMPT = (
    "You are a fast triage classifier for a clinical guideline retrieval system. Given a patient "
    "question and a list of available guideline topics, decide which topic(s), if any, the question "
    "is actually about. Be conservative: a question merely mentioning a symptom that happens to "
    "overlap with a guideline's wording is not enough -- only pick a topic if the question's core "
    "concern matches it."
)

ROUTE_USER_TEMPLATE = """\
Patient question:
{question}

Available guideline topics (label: what it covers):
{topic_list}
{feedback_block}
Respond with a single JSON object and nothing else, in this exact shape:
{{"labels": ["<0 to 2 topic labels from the list above -- just the label before the colon, exactly as written>"], "reasoning": "<one sentence>"}}
Use an empty list for "labels" if none of the topics are actually about this question.
"""

ROUTE_RETRY_FEEDBACK_TEMPLATE = """
Your previous pick ({previous_labels}) turned up no strong matches when searched: {previous_reasoning}
Pick a different topic if another one is plausible, or return an empty list if none of the available \
topics actually cover this question.
"""


def build_route_user_prompt(question: str, topic_labels: list[str], feedback: tuple[list[str], str] | None = None) -> str:
    topic_list = "\n".join(f"- {label}" for label in topic_labels)
    if feedback:
        previous_labels, previous_reasoning = feedback
        feedback_block = ROUTE_RETRY_FEEDBACK_TEMPLATE.format(previous_labels=", ".join(previous_labels) or "none", previous_reasoning=previous_reasoning)
    else:
        feedback_block = ""
    return ROUTE_USER_TEMPLATE.format(question=question, topic_list=topic_list, feedback_block=feedback_block)


DESCRIBE_TOPIC_SYSTEM_PROMPT = (
    "You summarize a clinical guideline document in one short sentence for a triage classifier "
    "that will match patient questions against it. Name the specific conditions, symptoms, "
    "complications, and clinical terms this guideline covers -- including less-obvious terms a "
    "patient might use that don't share vocabulary with the topic's own name (e.g. a specific "
    "symptom, complication, or alternate name for the condition)."
)

DESCRIBE_TOPIC_USER_TEMPLATE = """\
Guideline topic label: {label}

Section headings from this guideline:
{headings}

Write one sentence (max ~25 words), plain text only, no JSON or formatting: what conditions, \
symptoms, and clinical terms does this guideline cover?
"""


def build_describe_topic_user_prompt(label: str, headings: str) -> str:
    return DESCRIBE_TOPIC_USER_TEMPLATE.format(label=label, headings=headings)


def parse_route_response(text: str) -> tuple[list[str] | None, str]:
    """Parse the router's JSON verdict. Returns (labels, reasoning).

    labels is None on a parse failure -- distinct from an explicit empty list, which means the
    router confidently decided no topic covers the question (skip retrieval). A parse error is
    unknown, not a confident "not covered" verdict, so callers should fall back to unfiltered
    retrieval on None, the same fail-safe posture as the verifier's fail-open behavior on error."""
    try:
        payload = json.loads(text)
        labels = [str(label) for label in payload.get("labels", [])]
        reasoning = str(payload.get("reasoning", ""))
        return labels, reasoning
    except Exception:
        return None, "route_parse_error"

USER_PROMPT = """\
Answer the following question from a patient.
Give a concise, patient-facing answer: normally two to three sentences. Identify which excerpt -- or \
which part of an excerpt -- most directly answers the specific thing the patient is asking about, not \
just which excerpt covers the same condition, and base your answer on that. If that excerpt states a \
numeric threshold, timeframe, dosing/monitoring detail, or escalation/red-flag criterion for the specific \
point the patient raised, include it -- do not drop it for brevity. Do not pull in a threshold or \
criterion from a different retrieved excerpt just because it's about the same condition; a recommendation \
about a different clinical decision (e.g. surgical referral criteria) is not relevant to a question about \
something else on the same condition (e.g. starting exercise), even if both were retrieved. Only add a \
specific number, threshold, or timeframe if you can point to where the excerpts state it for the point \
being asked about; if the excerpts don't give a specific for this recommendation, say so in general terms \
instead of supplying one from your own medical knowledge -- a correct-sounding number you can't source from \
the excerpts is exactly as much a failure as an incorrect one.
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

REDRAFT_USER_PROMPT = """\
Answer the following question from a patient.
Give a concise, patient-facing answer: normally two to three sentences. Identify which excerpt -- or \
which part of an excerpt -- most directly answers the specific thing the patient is asking about, not \
just which excerpt covers the same condition, and base your answer on that. If that excerpt states a \
numeric threshold, timeframe, dosing/monitoring detail, or escalation/red-flag criterion for the specific \
point the patient raised, include it -- do not drop it for brevity. Do not pull in a threshold or \
criterion from a different retrieved excerpt just because it's about the same condition; a recommendation \
about a different clinical decision (e.g. surgical referral criteria) is not relevant to a question about \
something else on the same condition (e.g. starting exercise), even if both were retrieved. Only add a \
specific number, threshold, or timeframe if you can point to where the excerpts state it for the point \
being asked about; if the excerpts don't give a specific for this recommendation, say so in general terms \
instead of supplying one from your own medical knowledge -- a correct-sounding number you can't source from \
the excerpts is exactly as much a failure as an incorrect one.
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

Your previous answer was rejected by a clinical safety reviewer:
Previous answer:
{prior_draft}

Reviewer's reasoning:
{feedback}

Write a corrected answer that fixes the issue the reviewer raised, using only the excerpts above.
"""

VERIFY_SYSTEM_PROMPT = (
    "You are a clinical safety reviewer checking a draft answer against the guideline excerpts "
    "it was supposedly based on. You are strict about groundedness and patient safety, but you "
    "also recognize when an answer legitimately falls back to safe general advice because the "
    "excerpts do not cover the question. Only fail groundedness over a patient-group, condition, or "
    "intervention mismatch if using that excerpt actually changed the recommendation given -- a safe, "
    "generic statement that happens to be loosely inspired by a not-quite-matching excerpt is not a "
    "failure by itself."
)

VERIFY_USER_PROMPT = """\
Question from a patient:
{question}

Guideline excerpts the draft answer had access to:
{context}

Draft answer to review:
{draft_answer}

Judge the draft answer on four points. Decide each one independently -- a strong answer on one \
point does not excuse a failure on another.
1. Groundedness: does it only state dates, targets, follow-up intervals, organizations, or thresholds
   that are explicitly present in the excerpts (or none at all)? It must not apply a recommendation
   from one condition/indication to a different one implied by the question in a way that changes what
   the patient is actually told to do. A generic, safe statement that is only loosely inspired by a
   not-quite-matching excerpt is not a failure by itself -- only fail this point if the mismatch
   changes the substance of the recommendation given.
2. Safety: does it avoid missing an escalation/red-flag that the excerpts call for, and avoid giving
   dangerous or contraindicated advice?
3. Responsiveness: does it actually answer the patient's question rather than being a non-answer?
4. Coverage: when the excerpt that specifically answers what the patient asked states a numeric
   threshold, timeframe, dosing/monitoring detail, or escalation/red-flag criterion for that specific
   point, does the draft actually include it, rather than giving only the generic recommendation and
   leaving the specific out? Only fail this point over a specific from the excerpt answering the exact
   thing the patient asked about -- a threshold from a different recommendation on the same condition
   (e.g. surgical referral criteria when the patient asked about starting exercise) does not count, do
   not fail it for details the excerpts never mention, and do not fail it if the excerpts have nothing
   more specific to give for that point.

Respond with a single JSON object and nothing else, in this exact shape:
{{"groundedness_passed": true or false, "groundedness_reasoning": "<one sentence>",
"safety_passed": true or false, "safety_reasoning": "<one sentence>",
"responsiveness_passed": true or false, "responsiveness_reasoning": "<one sentence>",
"coverage_passed": true or false, "coverage_reasoning": "<one sentence>"}}
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


def build_redraft_user_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    max_context_chars: int,
    prior_draft: str,
    feedback: str,
) -> str:
    return REDRAFT_USER_PROMPT.format(
        question=question,
        context=format_context(retrieved_chunks, max_chars=max_context_chars),
        prior_draft=prior_draft,
        feedback=feedback,
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
    verification_iterations: int = 1,
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
    record["verification_iterations"] = verification_iterations
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
