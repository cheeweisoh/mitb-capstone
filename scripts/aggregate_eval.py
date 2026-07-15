"""Aggregate scores from all *_eval.csv files and print a summary table."""

import re
import sys
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

SCORE_COLS = [
    "guideline_covered",
    "clinical_correctness",
    "guideline_adherence",
    "safety",
    "escalation_alignment",
]

FILENAME_RE = re.compile(r"test_(?P<model>.+)_(?P<dataset>guidelines|reddit)_qa_eval\.csv$")


def load_eval_files(directory: Path) -> pd.DataFrame:
    records = []
    for path in sorted(directory.glob("*_eval.csv")):
        m = FILENAME_RE.match(path.name)
        if not m:
            print(f"Skipping unrecognised file: {path.name}", file=sys.stderr)
            continue

        df = pd.read_csv(path)
        missing = [c for c in SCORE_COLS if c not in df.columns]
        if missing:
            print(f"Skipping {path.name} — missing columns: {missing}", file=sys.stderr)
            continue

        covered = df[df["guideline_covered"] == 1]
        row = {
            "model": m.group("model"),
            "dataset": m.group("dataset"),
            "n": len(df),
        }
        for col in SCORE_COLS:
            subset = covered if col == "clinical_correctness" else df
            row[col] = subset[col].mean()

        records.append(row)

    if not records:
        raise SystemExit("No eval files found.")

    return pd.DataFrame(records)


def print_table(summary: pd.DataFrame) -> None:
    display = summary.copy()
    display["guideline_covered"] = display["guideline_covered"].map("{:.1%}".format)
    for col in SCORE_COLS[1:]:
        display[col] = display[col].map("{:.2f}".format)
    display = display.rename(columns={
        "model": "Model",
        "dataset": "Dataset",
        "n": "N",
        "guideline_covered": "Guideline Covered",
        "clinical_correctness": "Clinical Correctness",
        "guideline_adherence": "Guideline Adherence",
        "safety": "Safety",
        "escalation_alignment": "Escalation Alignment",
    })
    print(display.to_string(index=False))


def print_per_dimension(summary: pd.DataFrame) -> None:
    print("\n--- Per-dimension averages (all datasets combined) ---")
    by_model = summary.groupby("model")[SCORE_COLS].mean()
    by_model["guideline_covered"] = by_model["guideline_covered"].map("{:.1%}".format)
    for col in SCORE_COLS[1:]:
        by_model[col] = by_model[col].map("{:.2f}".format)
    print(by_model.to_string())

    print("\n--- Per-dimension averages by dataset type ---")
    by_dataset = summary.groupby("dataset")[SCORE_COLS].mean()
    by_dataset["guideline_covered"] = by_dataset["guideline_covered"].map("{:.1%}".format)
    for col in SCORE_COLS[1:]:
        by_dataset[col] = by_dataset[col].map("{:.2f}".format)
    print(by_dataset.to_string())


def main() -> None:
    summary = load_eval_files(DATASET_DIR)

    print("=== Per-file scores ===\n")
    print_table(summary.sort_values(["model", "dataset"]))
    print_per_dimension(summary)


if __name__ == "__main__":
    main()
