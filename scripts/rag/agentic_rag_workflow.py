import json
import re
from dataclasses import dataclass, field
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
    reformulated_query: str | None = None


@dataclass
class AgenticRagResult:
    final_answer: str
    retrieved_chunks: list[dict[str, Any]]
    verification_passed: bool | None
    verification_reasoning: str
    retrieval_iterations: int
    retrieval_queries_used: list[str] = field(default_factory=list)


class RetrieveEvent(Event):
    query: str
    iteration: int


class DraftEvent(Event):
    query: str
    search_query: str
    iteration: int
    retrieved_chunks: list[dict[str, Any]]


class VerifyEvent(Event):
    query: str
    search_query: str
    iteration: int
    retrieved_chunks: list[dict[str, Any]]
    draft_answer: str


class AgenticRagWorkflow(Workflow):
    """Bounded retrieve -> draft -> verify -> (re-retrieve | stop) loop.

    Deliberately not a ReAct/FunctionAgent: the control flow (retrieve, then
    draft, then verify, then a binary re-retrieve-or-stop decision) is fully
    fixed and deterministic, so an open-ended tool-selection agent would just
    add unreliable LLM-mediated indirection on top of a decision we already
    know how to make in plain Python.
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        verifier_llm: Any,
        max_iterations: int = 2,
        top_k: int = 5,
        max_context_chars: int = 7000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.retriever = retriever
        self.generator = generator
        self.verifier_llm = verifier_llm
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    @step
    async def start_retrieve(self, ctx: Context, ev: StartEvent) -> RetrieveEvent:
        await ctx.store.set("queries_used", [])
        return RetrieveEvent(query=ev.get("question"), iteration=1)

    @step
    async def retrieve(self, ctx: Context, ev: RetrieveEvent) -> DraftEvent:
        chunks = self.retriever.retrieve(ev.query, top_k=self.top_k)
        queries_used = await ctx.store.get("queries_used")
        queries_used.append(ev.query)
        await ctx.store.set("queries_used", queries_used)
        return DraftEvent(
            query=ev.query if ev.iteration == 1 else await ctx.store.get("original_question"),
            search_query=ev.query,
            iteration=ev.iteration,
            retrieved_chunks=chunks,
        )

    @step
    async def draft(self, ctx: Context, ev: DraftEvent) -> VerifyEvent:
        if ev.iteration == 1:
            await ctx.store.set("original_question", ev.query)
        original_question = ev.query if ev.iteration == 1 else await ctx.store.get("original_question")
        user_prompt = build_user_prompt(original_question, ev.retrieved_chunks, self.max_context_chars)
        answer = await self.generator.agenerate(SYSTEM_PROMPT, user_prompt)
        return VerifyEvent(
            query=original_question,
            search_query=ev.search_query,
            iteration=ev.iteration,
            retrieved_chunks=ev.retrieved_chunks,
            draft_answer=answer or "",
        )

    async def _run_verifier(self, ev: VerifyEvent) -> VerifierVerdict:
        prompt = build_verify_user_prompt(ev.query, ev.retrieved_chunks, ev.draft_answer, self.max_context_chars)
        try:
            content = await self.verifier_llm.agenerate(VERIFY_SYSTEM_PROMPT, prompt)
            match = _JSON_OBJECT_RE.search(content or "")
            payload = json.loads(match.group(0) if match else content)
            if not {"passed", "reasoning"}.issubset(payload.keys()):
                raise ValueError(f"verifier response missing required keys: {payload}")
            return VerifierVerdict(
                passed=bool(payload["passed"]),
                reasoning=str(payload["reasoning"]),
                reformulated_query=payload.get("reformulated_query") or None,
            )
        except Exception as e:
            return VerifierVerdict(passed=None, reasoning=f"verifier_error: {e}", reformulated_query=None)

    @step
    async def verify(self, ctx: Context, ev: VerifyEvent) -> RetrieveEvent | StopEvent:
        verdict = await self._run_verifier(ev)
        queries_used = await ctx.store.get("queries_used")

        if verdict.passed is not False or ev.iteration >= self.max_iterations:
            return StopEvent(
                result=AgenticRagResult(
                    final_answer=ev.draft_answer,
                    retrieved_chunks=ev.retrieved_chunks,
                    verification_passed=verdict.passed,
                    verification_reasoning=verdict.reasoning,
                    retrieval_iterations=ev.iteration,
                    retrieval_queries_used=queries_used,
                )
            )

        next_query = verdict.reformulated_query or ev.query
        return RetrieveEvent(query=next_query, iteration=ev.iteration + 1)
