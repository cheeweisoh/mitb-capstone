import json
import re
from pathlib import Path
from typing import Any

from llamaindex_retriever import DEFAULT_CHUNKS_PATH, load_chunk_rows
from rag_prompts import (DESCRIBE_TOPIC_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT,
                         build_describe_topic_user_prompt, build_route_user_prompt,
                         parse_route_response)

DEFAULT_DESCRIPTIONS_PATH = Path("dataset/rag/topic_descriptions.json")
# Section headings fed into the one-time description prompt, capped to keep
# that (one-off, per-document, cached-to-disk) call cheap even for a huge
# document like Stroke Rehabilitation Guidelines (~180 chunks).
_MAX_HEADING_CHARS = 2000

_MONTH_PAREN_RE = re.compile(r"\((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}\)", re.IGNORECASE)
_BARE_YEAR_PAREN_RE = re.compile(r"\(\d{4}\)")
_DUPLICATE_MARKER_RE = re.compile(r"\(\d+\)\s*$")
_HASH_SUFFIX_RE = re.compile(r"[a-f0-9]{16,}")


def _label_from_source_file(source_file: str) -> str:
    """Turn a guideline PDF filename into a short, readable topic label for the
    router prompt -- cosmetic cleanup only, not used for matching (the caller
    keeps its own label -> source_file mapping)."""
    name = Path(source_file).stem
    name = re.sub(r"^Singapore\s+", "", name, flags=re.IGNORECASE)
    name = name.replace("[PDF]", "")
    name = _MONTH_PAREN_RE.sub("", name)
    name = _BARE_YEAR_PAREN_RE.sub("", name)
    name = _DUPLICATE_MARKER_RE.sub("", name)
    name = _HASH_SUFFIX_RE.sub("", name)
    name = name.replace("-", " ").replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name else source_file


def build_topic_index(chunks_path: Path = DEFAULT_CHUNKS_PATH) -> tuple[list[str], dict[str, list[str]]]:
    """Build (topic_labels, label_to_source_files) once per run from the chunk
    corpus's unique source_file values. Reused for every question -- this is
    not re-derived per-call."""
    rows = load_chunk_rows(chunks_path)
    source_files = sorted({row["source_file"] for row in rows})
    label_to_sources: dict[str, list[str]] = {}
    for source_file in source_files:
        label = _label_from_source_file(source_file)
        label_to_sources.setdefault(label, []).append(source_file)
    return list(label_to_sources.keys()), label_to_sources


class TopicRouter:
    """Wraps the one-shot routing decision: which guideline topic(s), if any,
    a question is about. Built once per run; route() is called per question
    (and, from the workflow's bounded retry loop, up to max_route_iterations
    times per question with feedback from a weak first pick)."""

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        descriptions_path: Path = DEFAULT_DESCRIPTIONS_PATH,
    ) -> None:
        self.chunks_path = chunks_path
        self.descriptions_path = descriptions_path
        self.topic_labels, self.label_to_sources = build_topic_index(chunks_path)
        # Populated by ensure_descriptions(); label -> short description used
        # to enrich the router prompt beyond a bare filename-derived label.
        # Falls back to an empty description (just the label) if never called.
        self.descriptions: dict[str, str] = {}

    async def ensure_descriptions(self, generator: Any) -> None:
        """One-time, disk-cached generation of a short description per topic
        label -- e.g. "National Dengue Clinical Guideline" alone doesn't tell
        the router that guideline covers "hypotensive shock, fluid
        resuscitation", but a description built from its section headings
        does. Call once after construction, before the first route() call;
        cheap and safe to call on every run since it only makes an LLM call
        for labels missing from the cache file (new documents added to the
        corpus), reusing cached descriptions for everything else."""
        cached: dict[str, str] = {}
        if self.descriptions_path.exists():
            cached = json.loads(self.descriptions_path.read_text())

        missing_labels = [label for label in self.topic_labels if label not in cached]
        if not missing_labels:
            self.descriptions = {label: cached[label] for label in self.topic_labels}
            return

        headings_by_source: dict[str, list[str]] = {}
        for row in load_chunk_rows(self.chunks_path):
            headings_by_source.setdefault(row["source_file"], []).append(row["heading"])

        for label in missing_labels:
            headings: list[str] = []
            for source_file in self.label_to_sources[label]:
                headings.extend(headings_by_source.get(source_file, []))
            seen: set[str] = set()
            unique_headings = [h for h in headings if not (h in seen or seen.add(h))]
            heading_text = "; ".join(unique_headings)[:_MAX_HEADING_CHARS]

            prompt = build_describe_topic_user_prompt(label, heading_text)
            description = await generator.agenerate(DESCRIBE_TOPIC_SYSTEM_PROMPT, prompt)
            cached[label] = (description or label).strip()

        self.descriptions_path.parent.mkdir(parents=True, exist_ok=True)
        self.descriptions_path.write_text(json.dumps(cached, indent=2))
        self.descriptions = {label: cached[label] for label in self.topic_labels}

    def _topic_entries(self) -> list[str]:
        return [f"{label}: {self.descriptions[label]}" if label in self.descriptions else label for label in self.topic_labels]

    async def route(
        self,
        generator: Any,
        question: str,
        feedback: tuple[list[str], str] | None = None,
    ) -> tuple[list[str] | None, str]:
        """Returns (source_file_filter, reasoning):
        - non-empty list: restrict retrieval to these source_files
        - empty list []: router confidently decided no available topic covers this question
        - None: router response couldn't be parsed -- caller should fall back to unfiltered retrieval
        """
        prompt = build_route_user_prompt(question, self._topic_entries(), feedback=feedback)
        response = await generator.agenerate(ROUTE_SYSTEM_PROMPT, prompt)
        labels, reasoning = parse_route_response(response or "")
        if labels is None:
            return None, reasoning

        source_files: list[str] = []
        for label in labels:
            source_files.extend(self.label_to_sources.get(label, []))
        return source_files, reasoning
