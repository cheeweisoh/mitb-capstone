import argparse
import json
from pathlib import Path

import pandas as pd
from engine import check_answer, load_rules

DEFAULT_RULES = Path(__file__).with_name("rules.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check generated medical answers against guideline rules.")
    parser.add_argument("input", type=Path, help="CSV containing question and generated_answer")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_guardrails.csv")
    frame = pd.read_csv(args.input)
    required = {"question", "generated_answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    rules = load_rules(args.rules)
    details = []
    violation_ids = []
    for row in frame.to_dict(orient="records"):
        results = check_answer(
            question=str(row["question"]),
            generated_answer=str(row["generated_answer"]),
            rules=rules,
        )
        violations = [result for result in results if result.status == "violation"]
        details.append(json.dumps([result.as_dict() for result in results]))
        violation_ids.append("|".join(result.rule_id for result in violations))

    frame["guardrail_pass"] = [not ids for ids in violation_ids]
    frame["guardrail_violation_count"] = [len(ids.split("|")) if ids else 0 for ids in violation_ids]
    frame["guardrail_violation_ids"] = violation_ids
    frame["guardrail_details"] = details
    frame.to_csv(output, index=False)

    violation_rows = int((~frame["guardrail_pass"]).sum())
    print(f"Checked {len(frame)} rows with {len(rules)} rules.")
    print(f"Rows with violations: {violation_rows}. Output: {output}")
    if args.fail_on_violation and violation_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
