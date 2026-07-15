import base64
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ── configurable ──────────────────────────────────────────────────────────────
VISION_MODEL = "gpt-4.1"  # multimodal: processes page image + text
QA_MODEL = "gpt-4o-mini"  # text-only: generates question and answer
MIN_PAGE_WORDS = 30  # pages with fewer extracted words are skipped
MAX_CHUNK_WORDS = 600  # for .txt/.md fallback only
MIN_CHUNK_WORDS = 60  # for .txt/.md fallback only
RETRIES = 3
TARGET_PAIRS = 500  # total QA pairs to generate across all files
PDF_DPI = 150  # render resolution for PDF page images
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_SYSTEM_PROMPT = """\
You are a patient sitting in a doctor's waiting room. You have just read a medical leaflet \
and you want to ask your GP one question about something you found confusing, worrying, or \
relevant to your own situation.

Rules:
- Write exactly as a real patient would speak — informal, sometimes hesitant, emotionally \
  coloured. Use "I", "my", "we".
- NEVER use clinical or textbook language. A patient would say "my blood sugar" not \
  "glycaemic control"; "chest tightness" not "angina pectoris"; "water pill" not "diuretic".
- Always include enough personal context so the doctor understands your situation before \
  you ask. Mention relevant details such as: your symptoms and how long you have had them, \
  any medication you are on, a recent diagnosis or test result, a family member's situation, \
  or what triggered your concern. The context should feel natural, not like a formal history.
- The question itself must be grounded in the information on the page.
- Vary the style: some messages lead with symptoms, others with a recent event, others with \
  something confusing you read. Some can express worry, others curiosity, others frustration. \
  Do NOT always start with "I" or follow the same sentence pattern.
- 2–4 sentences: 1–2 sentences of personal context, then 1–2 sentences of question.

Good examples of the natural, varied tone we want:

• "I've been on metformin for about two weeks now and I keep feeling nauseous after every meal — \
it's making it really hard to eat properly. Is that a normal side effect that goes away, or \
should I be calling my doctor about it?"
• "I've had this tight feeling across my chest a few times this week, usually when I'm walking \
uphill, and it goes away when I stop. I'm 58 and my dad had a heart attack at 62 — should I \
be worried about this?"
• "My doctor mentioned my cholesterol is a bit high and gave me a leaflet about statins, but I \
read that they can cause muscle problems. How common is that really, and is it serious?"

Bad examples (too vague or too clinical — do NOT write like this):
• "What are the diagnostic criteria for hypertension according to current guidelines?"
• "Is it true that once you start insulin you're stuck on it forever?"
• "Can you explain the pharmacological mechanism of ACE inhibitors?"

Output ONLY the question as plain text. No preamble, no labels, no explanation.\
"""

# Patient personas injected per-call to drive variety
PATIENT_PERSONAS = [
    "a 52-year-old man who was just told his blood pressure is too high",
    "a 34-year-old mother worried about her child's recurring infections",
    "a 67-year-old retiree managing several medications at once",
    "a 29-year-old woman recently diagnosed with a chronic condition",
    "a 45-year-old who just got their first abnormal blood test result back",
    "a 58-year-old whose doctor mentioned they might need to start insulin",
    "a 40-year-old caregiver asking on behalf of an elderly parent",
    "a 23-year-old who is confused by advice they read online",
    "a 71-year-old man with joint pain that is getting worse",
    "a 38-year-old pregnant woman at her first antenatal visit",
    "a 55-year-old who had a recent scare and is now trying to understand their condition",
    "a 48-year-old woman who has been on the same medication for years and wants to know why",
]

QUESTION_USER_TEMPLATE = """\
You are {persona}.

## Page you just read (page {page_num})

{text}

Write your one question for the doctor now.\
"""

ANSWER_SYSTEM_PROMPT = """\
You are a helpful AI health assistant. You are NOT a doctor. \
Answer the patient's question using the background information provided — \
informative, balanced, and grounded in what that content actually says.

Rules:
- Answer the question directly. Most questions are asking for information or reassurance \
— just provide it clearly.
- Do NOT reference the background content, guidelines, or any document in your answer. \
Write as if you simply know this information — never say "according to the guidelines", \
"the source says", "based on the information provided", or any similar phrase.
- Do NOT default to "speak to your doctor / healthcare provider" as a closing line. That \
phrase is banned unless the background content explicitly states that this situation \
requires medical assessment or clinical intervention.
- Only recommend escalation (seeing a doctor, going to A&E, calling a helpline) when the \
background content describes red-flag symptoms, a situation requiring clinical diagnosis, \
or a condition that cannot safely be self-managed. Absent that, end the answer with the \
information itself — not a referral.
- Use hedged language where appropriate ("this could be", "it sounds like") but do not \
use hedging as an excuse to add an unnecessary referral at the end.
- Only use information present in the background content — do not add information that \
is not there.
- Keep it concise: 2–3 sentences.

Here are examples. Notice that two of the three do NOT end with a referral:

Question: "I've been taking allopurinol for gout for years and it seems to keep flares \
away. Is it normal to stay on it indefinitely, or should I expect to come off it at some \
point?"
Answer: "Long-term use of allopurinol is the standard approach for gout management — the \
goal is to keep uric acid levels low enough to prevent flares and dissolve existing \
crystals, which can take years. Stopping it usually causes uric acid to rise again and \
flares to return, so most people stay on it ongoing rather than stopping once things \
settle down."

Question: "I just read about bisphosphonates for osteoporosis and I'm worried about that \
jaw bone side effect I've heard about. How common is it really?"
Answer: "Medication-related osteonecrosis of the jaw is a real but very rare side effect — \
the risk in people taking bisphosphonates for osteoporosis is estimated at around 0.001% \
to 0.05%, which is extremely low. The risk is much higher in cancer patients receiving \
high-dose intravenous bisphosphonates, so the figures you may have seen online likely \
don't reflect what's typical for osteoporosis treatment."

Question: "I'm 55, my dad had a heart attack at 62, and I've been getting chest tightness \
when I walk uphill that goes away when I stop. Should I be worried?"
Answer: "What you're describing — chest tightness on exertion that relieves with rest — is \
a classic pattern that guidelines flag as needing prompt cardiac evaluation, especially \
given your age and family history. You should see a doctor soon rather than waiting to see \
if it gets worse."

Output ONLY the answer as plain text. No preamble, no explanation.\
"""

ANSWER_USER_TEMPLATE = """\
## Background

{text}

## Patient Question

{question}

Answer the patient's question now.\
"""


def _trim_to_sentence(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    last = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"))
    return truncated[: last + 1] if last != -1 else truncated


def _text_to_chunks(raw: str) -> list[str]:
    """Split plain text into word-bounded chunks (used for .txt/.md files)."""
    lines = re.split(r"\n+", raw)
    MIN_LINE_WORDS = 8
    filtered = [" ".join(ln.split()) for ln in lines if len(ln.split()) >= MIN_LINE_WORDS]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for line in filtered:
        line_words = len(line.split())
        if current_words + line_words > MAX_CHUNK_WORDS and current_words >= MIN_CHUNK_WORDS:
            chunks.append(_trim_to_sentence(" ".join(current_parts), MAX_CHUNK_WORDS))
            current_parts = [line]
            current_words = line_words
        else:
            current_parts.append(line)
            current_words += line_words

    if current_words >= MIN_CHUNK_WORDS:
        chunks.append(_trim_to_sentence(" ".join(current_parts), MAX_CHUNK_WORDS))

    return chunks


def extract_pages(path: Path) -> list[dict[str, Any]]:
    """Return a list of page dicts: {text, image_b64, page_num}.

    PDFs: one entry per page, rendered as PNG + extracted text.
    TXT/MD: chunked text only, image_b64 is None.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            import fitz  # pymupdf
        except ImportError:
            sys.exit("pymupdf is required. Run: uv add pymupdf")

        fitz.TOOLS.mupdf_display_errors(False)
        doc = fitz.open(str(path))
        pages: list[dict[str, Any]] = []
        mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if len(text.split()) < MIN_PAGE_WORDS:
                continue

            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            image_b64 = base64.b64encode(pix.tobytes("png")).decode()

            pages.append({"text": text, "image_b64": image_b64, "page_num": page_num})

        doc.close()
        return pages

    elif suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8")
        chunks = _text_to_chunks(raw)
        return [{"text": c, "image_b64": None, "page_num": i + 1} for i, c in enumerate(chunks)]

    else:
        sys.exit(f"Unsupported file type: {suffix!r}. Supported: .pdf, .txt, .md")


def _create_client() -> Any:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else OpenAI()


CLIENT = _create_client()


def _call_api(model: str, system: str, user_text: str, image_b64: str | None, retries: int) -> str | None:
    if image_b64:
        user_content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
        ]
    else:
        user_content = user_text

    for attempt in range(retries):
        try:
            response = CLIENT.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            status_code = getattr(e, "status_code", None)
            resp = getattr(e, "response", None)
            if status_code is None and resp is not None:
                status_code = getattr(resp, "status_code", None)

            if status_code == 429:
                tqdm.write("  Rate limited (429) — sleeping 60 s …")
                time.sleep(60)
                continue

            tqdm.write(f"  API error (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                return None
            time.sleep(2**attempt)

    return None


EXTRACT_SYSTEM_PROMPT = """\
You are a medical document parser. Your job is to extract all readable content from a page \
of a clinical guideline, exactly as it appears — including text inside tables, figures, \
flowcharts, sidebars, and headers. Do not summarise or interpret. Do not omit anything. \
Output clean, continuous prose that preserves all the medical information on the page.\
"""

EXTRACT_USER_TEMPLATE = """\
Extract all content from page {page_num} now.\
"""


def extract_page_content(page: dict[str, Any], retries: int = RETRIES) -> str | None:
    return _call_api(
        model=VISION_MODEL,
        system=EXTRACT_SYSTEM_PROMPT,
        user_text=EXTRACT_USER_TEMPLATE.format(page_num=page["page_num"]),
        image_b64=page["image_b64"],
        retries=retries,
    )


def generate_question(content: str, page_num: int, retries: int = RETRIES) -> str | None:
    persona = random.choice(PATIENT_PERSONAS)
    return _call_api(
        model=QA_MODEL,
        system=QUESTION_SYSTEM_PROMPT,
        user_text=QUESTION_USER_TEMPLATE.format(persona=persona, page_num=page_num, text=content),
        image_b64=None,
        retries=retries,
    )


def generate_answer(content: str, page_num: int, question: str, retries: int = RETRIES) -> str | None:
    return _call_api(
        model=QA_MODEL,
        system=ANSWER_SYSTEM_PROMPT,
        user_text=ANSWER_USER_TEMPLATE.format(page_num=page_num, text=content, question=question),
        image_b64=None,
        retries=retries,
    )


GUIDELINES_DIR = Path("dataset/guidelines/local")
OUTPUT_PATH = Path("dataset/guidelines/guidelines_qa_pairs.csv")


if __name__ == "__main__":
    # ── collect pages per file ────────────────────────────────────────────────
    supported = {".pdf", ".txt", ".md"}
    source_files = sorted(p for p in GUIDELINES_DIR.iterdir() if p.suffix.lower() in supported)
    if not source_files:
        sys.exit(f"No supported files (.pdf/.txt/.md) found in {GUIDELINES_DIR}")

    pages_per_file: dict[Path, list[dict[str, Any]]] = {}
    for f in source_files:
        print(f"Extracting: {f.name} …")
        pages = extract_pages(f)
        print(f"  → {len(pages)} pages")
        if pages:
            pages_per_file[f] = pages

    if not pages_per_file:
        sys.exit("No usable pages found across all documents.")

    num_files = len(pages_per_file)
    total_pages = sum(len(p) for p in pages_per_file.values())
    print(f"\n{total_pages} total pages from {num_files} files.")

    # ── pool all pages and sample exactly TARGET_PAIRS ────────────────────────
    all_pages: list[tuple[str, dict[str, Any]]] = [(f.name, p) for f, pages in pages_per_file.items() for p in pages]
    print(f"Selecting {TARGET_PAIRS} pages from pool of {len(all_pages)}.")

    if len(all_pages) >= TARGET_PAIRS:
        selected = random.sample(all_pages, TARGET_PAIRS)
    else:
        selected = random.choices(all_pages, k=TARGET_PAIRS)

    random.shuffle(selected)

    # ── output ────────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_PATH.exists()

    # ── generation loop ───────────────────────────────────────────────────────
    n_success = n_failed = 0

    for i, (source_file, page) in enumerate(tqdm(selected, desc="Generating QA pairs")):
        page_num = page["page_num"]

        content = extract_page_content(page)
        if content is None:
            tqdm.write(f"  Skipping page {page_num} of {source_file}: content extraction failed.")
            n_failed += 1
            continue

        question = generate_question(content, page_num)
        if question is None:
            tqdm.write(f"  Skipping page {page_num} of {source_file}: question generation failed.")
            n_failed += 1
            continue

        answer = generate_answer(content, page_num, question)
        if answer is None:
            tqdm.write(f"  Skipping page {page_num} of {source_file}: answer generation failed.")
            n_failed += 1
            continue

        chunk = " ".join(content.split())
        record = {"source_file": source_file, "page_num": page_num, "chunk": chunk, "question": question, "answer": answer}
        pd.DataFrame([record]).to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
        write_header = False
        n_success += 1

    print(f"\nDone. {n_success} pairs saved to {OUTPUT_PATH}. {n_failed} failed.")
