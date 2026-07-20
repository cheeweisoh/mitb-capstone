import json
import re
from dataclasses import dataclass
from typing import Any

from llama_index.core.workflow import (Context, Event, StartEvent, StopEvent,
                                       Workflow, step)

from rag_prompts import (SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT,
                         build_redraft_user_prompt, build_user_prompt,
                         build_verify_user_prompt)

# Non-instruction-tuned-for-JSON generators (e.g. meditron/medalpaca) sometimes
# wrap the verdict in stray commentary; grab the first {...} span rather than
# requiring the whole response to be valid JSON.
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)

DEFAULT_MAX_ITERATIONS = 2


def _build_requery(question: str, feedback: str) -> str:
    """Fold the verifier's rejection reasoning into the original question so
    the follow-up retrieval call is a genuinely different query, not the same
    one asked twice (which would return the same ranked results)."""
    return f"{question}\n\n{feedback}"


def _merge_and_rerank_chunks(existing: list[dict[str, Any]], new: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Union existing and newly-retrieved chunks, deduplicated by chunk_id
    (keeping the higher score on a duplicate), sorted by score, and capped to
    top_k. Ties keep their first-seen order, so `existing` chunks win over
    `new` chunks scored identically."""
    by_id: dict[str, dict[str, Any]] = {}
    for chunk in existing + new:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in by_id or chunk["score"] > by_id[chunk_id]["score"]:
            by_id[chunk_id] = chunk
    ranked = sorted(by_id.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]


@dataclass
class VerifierVerdict:
    passed: bool | None  # None = verifier itself errored (fail-open, tri-state)
    reasoning: str


@dataclass
class AgenticRagResult:
    final_answer: str
    retrieved_chunks: list[dict[str, Any]]
    verification_passed: bool | None
    verification_reasoning: str
    verification_iterations: int


class DraftEvent(Event):
    query: str
    retrieved_chunks: list[dict[str, Any]]
    iteration: int
    prior_draft: str | None = None
    feedback: str | None = None


class VerifyEvent(Event):
    query: str
    retrieved_chunks: list[dict[str, Any]]
    draft_answer: str
    iteration: int


class AgenticRagWorkflow(Workflow):
    """Retrieve, then loop draft -> verify -> (re-retrieve +) redraft until the
    verifier passes the answer or max_iterations is reached.

    On a failed verification, the verifier's reasoning is fed back both into
    a follow-up retrieval call (query + reasoning, merged with the original
    chunks and re-ranked by score) and into the next draft prompt alongside
    the rejected answer. This lets the loop recover from bad retrieval, not
    just bad phrasing: a chunk the first query missed can still surface if
    the verifier's feedback names what's missing. A verifier error (fail-open,
    passed=None) is treated the same as a pass, since it doesn't indicate the
    answer is actually wrong.
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        verifier_llm: Any,
        top_k: int = 5,
        max_context_chars: int = 7000,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.retriever = retriever
        self.generator = generator
        self.verifier_llm = verifier_llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.max_iterations = max_iterations

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> DraftEvent:
        question = ev.get("question")
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        return DraftEvent(query=question, retrieved_chunks=chunks, iteration=1)

    @step
    async def draft(self, ctx: Context, ev: DraftEvent) -> VerifyEvent:
        if ev.prior_draft is None:
            user_prompt = build_user_prompt(ev.query, ev.retrieved_chunks, self.max_context_chars)
        else:
            user_prompt = build_redraft_user_prompt(
                ev.query, ev.retrieved_chunks, self.max_context_chars, ev.prior_draft, ev.feedback or ""
            )
        answer = await self.generator.agenerate(SYSTEM_PROMPT, user_prompt)
        return VerifyEvent(query=ev.query, retrieved_chunks=ev.retrieved_chunks, draft_answer=answer or "", iteration=ev.iteration)

    async def _run_verifier(self, ev: VerifyEvent) -> VerifierVerdict:
        prompt = build_verify_user_prompt(ev.query, ev.retrieved_chunks, ev.draft_answer, self.max_context_chars)
        try:
            content = await self.verifier_llm.agenerate(VERIFY_SYSTEM_PROMPT, prompt)
            match = _JSON_OBJECT_RE.search(content or "")
            payload = json.loads(match.group(0) if match else content)
            if not {"passed", "reasoning"}.issubset(payload.keys()):
                raise ValueError(f"verifier response missing required keys: {payload}")
            return VerifierVerdict(passed=bool(payload["passed"]), reasoning=str(payload["reasoning"]))
        except Exception as e:
            return VerifierVerdict(passed=None, reasoning=f"verifier_error: {e}")

    @step
    async def verify(self, ctx: Context, ev: VerifyEvent) -> DraftEvent | StopEvent:
        verdict = await self._run_verifier(ev)
        if verdict.passed is False and ev.iteration < self.max_iterations:
            requery = _build_requery(ev.query, verdict.reasoning)
            new_chunks = self.retriever.retrieve(requery, top_k=self.top_k)
            retrieved_chunks = _merge_and_rerank_chunks(ev.retrieved_chunks, new_chunks, self.top_k)
            return DraftEvent(
                query=ev.query,
                retrieved_chunks=retrieved_chunks,
                iteration=ev.iteration + 1,
                prior_draft=ev.draft_answer,
                feedback=verdict.reasoning,
            )
        return StopEvent(
            result=AgenticRagResult(
                final_answer=ev.draft_answer,
                retrieved_chunks=ev.retrieved_chunks,
                verification_passed=verdict.passed,
                verification_reasoning=verdict.reasoning,
                verification_iterations=ev.iteration,
            )
        )
