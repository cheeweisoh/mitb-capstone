# Guideline Guardrails

This directory contains deterministic checks applied after answer generation and before an
answer is released or included in evaluation. It complements, rather than replaces, the
LLM-based evaluator in `scripts/evaluation/`.

The guarded model runners in `scripts/guardrails/run_test/` apply fixed rules before generation.
For each patient question, the model first classifies the question into up to three guideline
disease categories, or `none` if no category applies. Rule categories are stored on each rule in
`rules.json`. When categories are selected, all rule messages and source provenance for those
categories are inserted into the model's hidden prompt as clinical guideline constraints. If the
classifier returns no usable category, the runner falls back to the deterministic per-question
matcher. The original runners in `scripts/run_test/` remain unchanged for baseline comparison.
Guarded runner outputs include `guardrail_category`, `guardrail_fallback_triggered`,
`guardrail_triggered`, `guardrail_rule_count`, and `guardrail_rule_ids`. These columns describe
rules injected before generation; they do not indicate whether the generated answer complied with
those rules. The post-generation checker remains available to detect cases where a model ignored
applicable fixed checks.

Run the checks against any model result CSV:

```bash
uv run python scripts/guardrails/check_outputs.py dataset/test_chatgpt_guidelines_qa.csv
```

The output adds `guardrail_pass`, violation count/IDs, and JSON details with source provenance.
Use `--fail-on-violation` in an automated pipeline when any flagged row should block release.

## Rule authoring workflow

1. Identify a recommendation that is explicit, narrow, and detectable from the available text.
2. Add its document, page, recommendation, applicability patterns, and answer check to
   `rules.json`.
3. Test positive, negative, negated, and out-of-scope examples.
4. Have a clinician or suitably qualified reviewer compare the formal rule with the source.
5. Record clinical approval in version control before deploying the updated rule set.

LLMs can propose rules from `extraction_cache.json`, but proposed rules should be reviewed before
the updated file is deployed. Avoid formalising recommendations that depend on examination
findings, unstated patient context, shared decision-making, or exceptions that cannot be
represented by the applicability fields.

Supported applicability keys are `question_all`, `question_any`, `question_none`, and
`question_relaxed_any`. `question_all` and `question_any` define the primary strict match.
`question_relaxed_any` is an opt-in fallback for broader patient-language aliases; keep each
pattern specific enough to include both the clinical topic and the rule-specific scenario. Checks
can use `forbidden_any` for prohibited advice or `required_any` for required safety language.
Patterns are case-insensitive Python regular expressions.
