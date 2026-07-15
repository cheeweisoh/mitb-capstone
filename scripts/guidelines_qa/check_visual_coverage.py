"""
check_visual_coverage.py

Audits how well the text-only extraction pipeline (pypdf) captures
information that lives inside images and tables in the PDF guidelines.

For each PDF page that contains embedded images or dense table-like
structures, the script:
  1. Renders the page to a PNG (via pymupdf)
  2. Sends it to GPT-4o vision, asking it to describe the visual content
     and judge whether the accompanying pypdf text excerpt captured it
  3. Writes one row per audited page to dataset/guidelines/visual_coverage_report.csv

Usage:
    uv run python scripts/guidelines_qa/check_visual_coverage.py [--max-pages N] [--file FILENAME]

Options:
    --max-pages N    Stop after auditing N pages total (default: 50)
    --file FILENAME  Audit only the named file (partial match ok)
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import fitz  # pymupdf
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
GUIDELINES_DIR = Path("dataset/guidelines/local")
OUTPUT_PATH = Path("dataset/guidelines/visual_coverage_report.csv")
OPENAI_MODEL = "gpt-4o"
DPI = 150  # render resolution — higher = better quality but larger payload
MIN_IMAGE_AREA = 5000  # px²; ignore tiny decorative images (logos, bullets)
RETRIES = 3
# ─────────────────────────────────────────────────────────────────────────────

AUDIT_SYSTEM_PROMPT = """\
You are auditing a medical PDF guideline to check how well a plain-text
extraction captured structured visual content on a page.

You will receive:
1. The rendered page image.
2. The raw text that a PDF text-extractor (pypdf) pulled from that page.

Your job:
A. Identify what visual elements are present on the page that carry
   meaningful clinical information — tick boxes from:
   [ ] table  [ ] figure/chart  [ ] diagram  [ ] dosing schedule
   [ ] algorithm/flowchart  [ ] image/photo  [ ] other visual
   [ ] none (page is text-only)

B. For each visual element found, write a brief plain-English summary of
   the key information it contains (≤ 3 sentences per element).

C. Rate how well the text extraction captured that information on a 1–5 scale:
   1 = completely missed  3 = partial  5 = fully captured

D. List any specific facts, values, or relationships that appear in the
   visuals but are ABSENT from the extracted text.

Respond in this exact JSON format — no markdown fences, no extra keys:

{
  "visual_elements": ["table", "figure"],
  "visual_summary": "...",
  "coverage_score": 3,
  "missed_facts": "..."
}

If there are no meaningful visual elements, return:
{
  "visual_elements": [],
  "visual_summary": "Page is text-only.",
  "coverage_score": 5,
  "missed_facts": ""
}
"""


def page_to_base64_png(page: fitz.Page, dpi: int = DPI) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    buf = io.BytesIO()
    buf.write(pix.tobytes("png"))
    return base64.b64encode(buf.getvalue()).decode()


def has_meaningful_visuals(page: fitz.Page) -> bool:
    """Return True if the page has embedded images above the size threshold."""
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            pix = fitz.Pixmap(page.parent, xref)
            if pix.width * pix.height >= MIN_IMAGE_AREA:
                return True
        except Exception:
            continue

    # Heuristic: pages with very short pypdf text but lots of drawn rectangles
    # are likely table-heavy. We detect this by checking drawing commands.
    paths = page.get_drawings()
    rect_count = sum(1 for p in paths if p.get("type") == "f" or len(p.get("items", [])) > 3)
    return rect_count >= 10


def extract_page_text_pypdf(path: Path, page_number: int) -> str:
    """Extract text from a single page using pypdf (mirrors the main pipeline)."""
    reader = PdfReader(str(path))
    if page_number >= len(reader.pages):
        return ""
    return reader.pages[page_number].extract_text() or ""


def audit_page(client: OpenAI, b64_image: str, page_text: str, retries: int = RETRIES) -> dict:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": f"## Extracted text (pypdf)\n\n{page_text or '(empty)'}",
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            import json

            return json.loads(response.choices[0].message.content)
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
                return {
                    "visual_elements": [],
                    "visual_summary": "API error",
                    "coverage_score": None,
                    "missed_facts": "",
                }
            time.sleep(2**attempt)

    return {
        "visual_elements": [],
        "visual_summary": "API error",
        "coverage_score": None,
        "missed_facts": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=50, metavar="N")
    parser.add_argument("--file", type=str, default=None, metavar="FILENAME")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    all_pdfs = sorted(GUIDELINES_DIR.glob("*.pdf"))
    if not all_pdfs:
        sys.exit(f"No PDFs found in {GUIDELINES_DIR}")

    if args.file:
        pdfs = [p for p in all_pdfs if args.file.lower() in p.name.lower()]
        if not pdfs:
            sys.exit(f"No PDF matching {args.file!r} found in {GUIDELINES_DIR}")
    else:
        print("Available PDFs:\n")
        for i, p in enumerate(all_pdfs, 1):
            print(f"  {i:2d}. {p.name}")
        print()
        raw = input("Enter number(s) to audit (e.g. 1  or  1,3,5): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in raw.replace(",", " ").split()]
            pdfs = [all_pdfs[i] for i in indices]
        except ValueError, IndexError:
            sys.exit("Invalid selection.")
        print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_PATH.exists()

    pages_audited = 0
    pages_skipped = 0

    outer = tqdm(pdfs, desc="PDFs", unit="file")
    for pdf_path in outer:
        outer.set_postfix(file=pdf_path.name[:40])

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            tqdm.write(f"  Cannot open {pdf_path.name}: {e}")
            continue

        for page_idx in range(len(doc)):
            if pages_audited >= args.max_pages:
                break

            page = doc[page_idx]
            if not has_meaningful_visuals(page):
                pages_skipped += 1
                continue

            page_text = extract_page_text_pypdf(pdf_path, page_idx)
            b64_image = page_to_base64_png(page)

            tqdm.write(f"  Auditing {pdf_path.name} p.{page_idx + 1} …")
            result = audit_page(client, b64_image, page_text)

            record = {
                "source_file": pdf_path.name,
                "page": page_idx + 1,
                "visual_elements": ", ".join(result.get("visual_elements", [])),
                "visual_summary": result.get("visual_summary", ""),
                "coverage_score": result.get("coverage_score"),
                "missed_facts": result.get("missed_facts", ""),
                "pypdf_text_excerpt": (page_text or "")[:500],
            }

            pd.DataFrame([record]).to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
            write_header = False
            pages_audited += 1

        doc.close()
        if pages_audited >= args.max_pages:
            tqdm.write(f"Reached --max-pages {args.max_pages}, stopping.")
            break

    print("\nAudit complete.")
    print(f"  Pages with visuals audited : {pages_audited}")
    print(f"  Pages skipped (text-only)  : {pages_skipped}")
    print(f"  Report written to          : {OUTPUT_PATH}")

    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        visual_pages = df[df["visual_elements"].str.len() > 0]
        if not visual_pages.empty:
            avg = visual_pages["coverage_score"].dropna().mean()
            low = (visual_pages["coverage_score"] < 3).sum()
            print("\nSummary (visual pages only):")
            print(f"  Average coverage score     : {avg:.2f} / 5")
            print(f"  Pages with poor coverage   : {low} (score < 3)")
            print("  Files with missed content  :")
            for f, grp in visual_pages.groupby("source_file"):
                bad = (grp["coverage_score"] < 3).sum()
                if bad:
                    print(f"    {f}: {bad} page(s) with score < 3")


if __name__ == "__main__":
    main()
