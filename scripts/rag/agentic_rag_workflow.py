import json
import re
from dataclasses import dataclass
from typing import Any

from llama_index.core.workflow import (Context, Event, StartEvent, StopEvent,
                                       Workflow, step)

from rag_prompts import (SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT,
                         build_user_prompt, build_verify_user_prompt)

# Non-instruction-tuned-for-JSON generators (e.g. meditron/medalpaca) sometimes
# wrap the verdict in stray commentary; grab the first {...} span rather than
# requiring the whole response to be valid JSON.
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


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


class DraftEvent(Event):
    query: str
    retrieved_chunks: list[dict[str, Any]]


class VerifyEvent(Event):
    query: str
    retrieved_chunks: list[dict[str, Any]]
    draft_answer: str


class AgenticRagWorkflow(Workflow):
    """Single-pass retrieve -> draft -> verify.

    The verifier's verdict is reported alongside the answer for downstream
    analysis, but it never triggers a re-retrieve: there is exactly one
    retrieval and one draft per question.
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        verifier_llm: Any,
        top_k: int = 5,
        max_context_chars: int = 7000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.retriever = retriever
        self.generator = generator
        self.verifier_llm = verifier_llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> DraftEvent:
        question = ev.get("question")
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        return DraftEvent(query=question, retrieved_chunks=chunks)

    @step
    async def draft(self, ctx: Context, ev: DraftEvent) -> VerifyEvent:
        user_prompt = build_user_prompt(ev.query, ev.retrieved_chunks, self.max_context_chars)
        answer = await self.generator.agenerate(SYSTEM_PROMPT, user_prompt)
        return VerifyEvent(query=ev.query, retrieved_chunks=ev.retrieved_chunks, draft_answer=answer or "")

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
    async def verify(self, ctx: Context, ev: VerifyEvent) -> StopEvent:
        verdict = await self._run_verifier(ev)
        return StopEvent(
            result=AgenticRagResult(
                final_answer=ev.draft_answer,
                retrieved_chunks=ev.retrieved_chunks,
                verification_passed=verdict.passed,
                verification_reasoning=verdict.reasoning,
            )
        )
