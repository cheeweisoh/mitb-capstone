import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path("dataset/guidelines/local/extraction_cache.json")
DEFAULT_OUTPUT_PATH = Path("dataset/rag/rag_chunks.csv")
DEFAULT_MIN_WORDS = 30
DEFAULT_MAX_WORDS = 700
DEFAULT_OVERLAP_WORDS = 100
DEFAULT_MERGE_UNDER_WORDS = 100

SKIP_HEADING_RE = re.compile(
    r"^("
    r"\d+(?:\.\d+)*\.?\s+references?|"
    r"references?|"
    r"expert group|"
    r"foreword|"
    r"date of reference to source guideline|"
    r"ministry of health singapore|"
    r"citation|"
    r"implementation|"
    r"about this guideline|"
    r"message to healthcare professionals|"
    r"about the agency|"
    r"secretariat|"
    r"suggested citation|"
    r"find out more about ace|"
    r"acknowledgements?|"
    r"appendix|"
    r"annex"
    r")\b",
    re.IGNORECASE,
)

SKIP_TEXT_RE = re.compile(
    r"("
    r"name\s*\|\s*organisation\s*\|\s*role\s*\|\s*contribution|"
    r"all rights reserved|"
    r"reproduction of this publication|"
    r"the ministry of health, singapore disclaims|"
    r"agency for care effectiveness.*college road|"
    r"project lead|"
    r"chairperson\s+|"
    r"\bmembers\s+"
    r")",
    re.IGNORECASE,
)

TABLE_CELL_RE = re.compile(
    r"^(strong for|weak for|strong against|weak against|nil|yes|no|for|against)$",
    re.IGNORECASE,
)

CLINICAL_SHORT_HEADING_RE = re.compile(
    r"^("
    r"mri is (not |may be )?indicated|"
    r"cxr is (not |may not be )?indicated|"
    r"red flags?|"
    r"risk of harm|"
    r"referral considerations|"
    r"specialist referral|"
    r"emergency|"
    r"urgent"
    r")\b",
    re.IGNORECASE,
)

STRONG_HEADING_RE = re.compile(
    r"^("
    r"\d+(?:\.\d+)*\.?\s+references?|"
    r"recommendation\s+\d+[a-z]?|"
    r"practice point|"
    r"clinical practice point|"
    r"objective|"
    r"scope|"
    r"target audience|"
    r"foreword|"
    r"date of reference to source guideline|"
    r"ministry of health singapore|"
    r"citation|"
    r"implementation|"
    r"about this guideline|"
    r"message to healthcare professionals|"
    r"references?|"
    r"expert group|"
    r"about the agency|"
    r"secretariat|"
    r"suggested citation"
    r")\b",
    re.IGNORECASE,
)

TABLE_OF_CONTENTS_RE = re.compile(
    r"\btable of contents\b|" r"\bcontents\b\s+\d+\.\s+\w+.*\d+\.\s+\w+",
    re.IGNORECASE,
)

FRONT_MATTER_RE = re.compile(
    r"\b(first published|last updated|published):\s+\d{1,2}\s+\w+\s+\d{4}\b|" r"\bwww\.ace-hta\.gov\.sg\b",
    re.IGNORECASE,
)


@dataclass
class Page:
    source_file: str
    page_num: int
    text: str


@dataclass
class Section:
    source_file: str
    heading: str
    start_page: int
    end_page: int
    text: str


def normalize_text(text: str) -> str:
    """Collapse noisy whitespace while preserving readable text."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_lines(text: str) -> list[str]:
    """Clean extracted page text while preserving line boundaries for headings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def word_count(text: str) -> int:
    return len(text.split())


def parse_cache_key(key: str) -> tuple[str, int]:
    """Parse '<source_file>::<page_num>' cache keys."""
    source_file, sep, page_num_text = key.rpartition("::")
    if not sep or not source_file or not page_num_text.isdigit():
        raise ValueError(f"Unexpected cache key format: {key!r}")
    return source_file, int(page_num_text)


def split_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split text into fixed-size word chunks with optional overlap."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    step = max_words - overlap_words
    if step <= 0:
        raise ValueError("--overlap-words must be smaller than --max-words")

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def detect_heading(line: str) -> str | None:
    """Detect recommendation and section headings from flattened extracted text."""
    cleaned = line.removeprefix("[Header]").strip()
    if not cleaned:
        return None
    if "|" in cleaned:
        return None
    if TABLE_CELL_RE.match(cleaned):
        return None
    if STRONG_HEADING_RE.match(cleaned):
        return cleaned[:120]
    if CLINICAL_SHORT_HEADING_RE.match(cleaned):
        return cleaned[:120]
    if cleaned.startswith(("•", "-", "|")):
        return None
    if cleaned.endswith((".", "?", "!", ";", ",", ":")):
        return None
    if not cleaned[0].isupper():
        return None
    if len(cleaned) > 100:
        return None
    if word_count(cleaned) > 12:
        return None
    if not re.search(r"[A-Za-z]", cleaned):
        return None
    return cleaned[:120]


def should_skip_section(heading: str) -> bool:
    return bool(SKIP_HEADING_RE.match(heading.strip()))


def is_table_of_contents(text: str) -> bool:
    text = normalize_text(text)
    return bool(TABLE_OF_CONTENTS_RE.search(text[:1000]))


def is_front_matter_only(heading: str, text: str) -> bool:
    if heading.strip().casefold() not in {
        "ace clinical guidance",
        "ace clinical guideline",
        "document start",
    }:
        return False

    normalized = normalize_text(text)
    return bool(FRONT_MATTER_RE.search(normalized[:800]))


def should_skip_text(text: str) -> bool:
    return bool(SKIP_TEXT_RE.search(text) or is_table_of_contents(text))


def is_strong_split_heading(heading: str) -> bool:
    return bool(STRONG_HEADING_RE.match(heading.strip()) or CLINICAL_SHORT_HEADING_RE.match(heading.strip()))


def text_fingerprint(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(data).__name__}")

    return data


def group_pages(cache: dict[str, Any]) -> tuple[dict[str, list[Page]], int]:
    pages_by_file: dict[str, list[Page]] = {}
    skipped_values = 0

    for key, value in cache.items():
        source_file, page_num = parse_cache_key(key)
        if not isinstance(value, str) or not normalize_text(value):
            skipped_values += 1
            continue
        pages_by_file.setdefault(source_file, []).append(Page(source_file, page_num, value))

    for pages in pages_by_file.values():
        pages.sort(key=lambda page: page.page_num)

    return pages_by_file, skipped_values


def split_pages_into_sections(pages: list[Page]) -> list[Section]:
    if not pages:
        return []

    sections: list[Section] = []
    current_heading = "Document start"
    current_start_page = pages[0].page_num
    current_end_page = pages[0].page_num
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = normalize_text(" ".join(current_lines))
        if text:
            sections.append(
                Section(
                    source_file=pages[0].source_file,
                    heading=current_heading,
                    start_page=current_start_page,
                    end_page=current_end_page,
                    text=text,
                )
            )
        current_lines = []

    for page in pages:
        for line in normalize_lines(page.text):
            heading = detect_heading(line)
            if heading and current_lines:
                flush()
                current_heading = heading
                current_start_page = page.page_num
                current_end_page = page.page_num
                current_lines = [line]
            else:
                if heading and not current_lines:
                    current_heading = heading
                    current_start_page = page.page_num
                current_lines.append(line)
                current_end_page = page.page_num

    flush()
    return sections


def merge_short_sections(sections: list[Section], merge_under_words: int, max_words: int) -> list[Section]:
    """Merge weak short sections into adjacent context from the same document."""
    if merge_under_words <= 0:
        return sections

    merged: list[Section] = []
    pending: Section | None = None

    for section in sections:
        if pending is not None:
            combined_words = word_count(pending.text) + word_count(section.text)
            if combined_words <= max_words:
                section = Section(
                    source_file=section.source_file,
                    heading=pending.heading,
                    start_page=pending.start_page,
                    end_page=max(pending.end_page, section.end_page),
                    text=f"{pending.text} {section.text}",
                )
                pending = None
            else:
                merged.append(pending)
                pending = None

        section_words = word_count(section.text)
        if section_words >= merge_under_words or is_strong_split_heading(section.heading):
            merged.append(section)
            continue

        if merged and word_count(merged[-1].text) + section_words <= max_words:
            previous = merged[-1]
            merged[-1] = Section(
                source_file=previous.source_file,
                heading=previous.heading,
                start_page=previous.start_page,
                end_page=max(previous.end_page, section.end_page),
                text=f"{previous.text} {section.text}",
            )
        else:
            pending = section

    if pending is not None:
        if merged and word_count(merged[-1].text) + word_count(pending.text) <= max_words:
            previous = merged[-1]
            merged[-1] = Section(
                source_file=previous.source_file,
                heading=previous.heading,
                start_page=previous.start_page,
                end_page=max(previous.end_page, pending.end_page),
                text=f"{previous.text} {pending.text}",
            )
        else:
            merged.append(pending)

    return merged


def split_section_to_records(
    section: Section,
    section_index: int,
    min_words: int,
    max_words: int,
    overlap_words: int,
) -> list[dict[str, str | int]]:
    if should_skip_section(section.heading) or is_front_matter_only(section.heading, section.text) or should_skip_text(section.text) or word_count(section.text) < min_words:
        return []

    chunks = split_words(section.text, max_words=max_words, overlap_words=overlap_words)
    records: list[dict[str, str | int]] = []
    section_slug = slugify(section.heading)

    for chunk_index, chunk_text in enumerate(chunks, start=1):
        if word_count(chunk_text) < min_words:
            continue
        records.append(
            {
                "chunk_id": (f"{section.source_file}::p{section.start_page}-{section.end_page}" f"::s{section_index:04d}::{section_slug}::c{chunk_index}"),
                "source_file": section.source_file,
                "start_page": section.start_page,
                "end_page": section.end_page,
                "heading": section.heading,
                "chunk_index": chunk_index,
                "word_count": word_count(chunk_text),
                "text": chunk_text,
            }
        )

    return records


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "section"


def build_section_records(
    cache: dict[str, Any],
    min_words: int,
    max_words: int,
    overlap_words: int,
    merge_under_words: int,
) -> tuple[list[dict[str, str | int]], int, int]:
    pages_by_file, skipped_pages = group_pages(cache)
    records: list[dict[str, str | int]] = []
    skipped_sections = 0
    seen_hashes: set[str] = set()

    for source_file in sorted(pages_by_file):
        sections = split_pages_into_sections(pages_by_file[source_file])
        sections = merge_short_sections(
            sections,
            merge_under_words=merge_under_words,
            max_words=max_words,
        )
        for section_index, section in enumerate(sections, start=1):
            section_records = split_section_to_records(
                section,
                section_index=section_index,
                min_words=min_words,
                max_words=max_words,
                overlap_words=overlap_words,
            )
            if section_records:
                for record in section_records:
                    fingerprint = text_fingerprint(str(record["text"]))
                    if fingerprint in seen_hashes:
                        skipped_sections += 1
                        continue
                    seen_hashes.add(fingerprint)
                    records.append(record)
            else:
                skipped_sections += 1

    return records, skipped_pages, skipped_sections


def build_page_records(
    cache: dict[str, Any],
    min_words: int,
    max_words: int | None,
    overlap_words: int,
) -> tuple[list[dict[str, str | int]], int, int]:
    records: list[dict[str, str | int]] = []
    pages_by_file, skipped_pages = group_pages(cache)
    skipped_chunks = 0
    seen_hashes: set[str] = set()

    for source_file in sorted(pages_by_file):
        for page in pages_by_file[source_file]:
            text = normalize_text(page.text)
            if word_count(text) < min_words:
                skipped_chunks += 1
                continue

            chunks = [text]
            if max_words is not None:
                chunks = split_words(text, max_words=max_words, overlap_words=overlap_words)

            for chunk_index, chunk_text in enumerate(chunks, start=1):
                fingerprint = text_fingerprint(chunk_text)
                if fingerprint in seen_hashes:
                    skipped_chunks += 1
                    continue
                seen_hashes.add(fingerprint)
                records.append(
                    {
                        "chunk_id": f"{source_file}::p{page.page_num}::c{chunk_index}",
                        "source_file": source_file,
                        "start_page": page.page_num,
                        "end_page": page.page_num,
                        "heading": "Page",
                        "chunk_index": chunk_index,
                        "word_count": word_count(chunk_text),
                        "text": chunk_text,
                    }
                )

    return records, skipped_pages, skipped_chunks


def write_csv(records: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chunk_id",
        "source_file",
        "start_page",
        "end_page",
        "heading",
        "chunk_index",
        "word_count",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert guideline extraction_cache.json into a CSV of RAG chunks.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Path to extraction cache JSON. Default: {DEFAULT_CACHE_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=f"Drop pages/chunks with fewer than this many words. Default: {DEFAULT_MIN_WORDS}",
    )
    parser.add_argument(
        "--mode",
        choices=["section", "page"],
        default="section",
        help="Chunking mode. Default: section",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
        help=("Split oversized chunks at this many words. " f"Default: {DEFAULT_MAX_WORDS}. Use 0 with --mode page to disable splitting."),
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=DEFAULT_OVERLAP_WORDS,
        help=("Word overlap between split chunks. " f"Default: {DEFAULT_OVERLAP_WORDS}"),
    )
    parser.add_argument(
        "--merge-under-words",
        type=int,
        default=DEFAULT_MERGE_UNDER_WORDS,
        help=("In section mode, merge non-critical sections below this many words " f"into adjacent context. Default: {DEFAULT_MERGE_UNDER_WORDS}. Use 0 to disable."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_words < 1:
        raise SystemExit("--min-words must be at least 1")
    if args.max_words < 0:
        raise SystemExit("--max-words cannot be negative")
    if args.mode == "section" and args.max_words < 1:
        raise SystemExit("--max-words must be at least 1 in section mode")
    if args.overlap_words < 0:
        raise SystemExit("--overlap-words cannot be negative")
    if args.merge_under_words < 0:
        raise SystemExit("--merge-under-words cannot be negative")
    if args.max_words == 0:
        args.overlap_words = 0

    cache = load_cache(args.cache)
    if args.mode == "section":
        records, skipped_pages, skipped_chunks = build_section_records(
            cache,
            min_words=args.min_words,
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            merge_under_words=args.merge_under_words,
        )
    else:
        max_words = args.max_words or None
        records, skipped_pages, skipped_chunks = build_page_records(
            cache,
            min_words=args.min_words,
            max_words=max_words,
            overlap_words=args.overlap_words,
        )
    write_csv(records, args.output)

    source_files = {record["source_file"] for record in records}
    print(f"Loaded {len(cache)} cached pages from {args.cache}")
    print(f"Skipped {skipped_pages} empty or failed extraction pages")
    print(f"Skipped {skipped_chunks} short or excluded chunks")
    print(f"Wrote {len(records)} {args.mode}-mode chunks " f"from {len(source_files)} source files to {args.output}")


if __name__ == "__main__":
    main()
