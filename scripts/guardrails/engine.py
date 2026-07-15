import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.json")
MAX_FUZZY_RULES = 3
MIN_FUZZY_SCORE = 6.0
MIN_FUZZY_CONDITION_MATCHES = 2
MIN_STRONG_TOPIC_ALIAS_MATCHES = 2
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "get",
    "had",
    "has",
    "have",
    "her",
    "him",
    "his",
    "how",
    "just",
    "like",
    "might",
    "more",
    "need",
    "not",
    "now",
    "our",
    "out",
    "read",
    "really",
    "she",
    "should",
    "some",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "was",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "worried",
    "would",
    "you",
}
TOPIC_ALIASES = {
    "asthma": ["asthma", "wheeze", "wheezing", "inhaler", "salbutamol", "albuterol"],
    "atopic": ["eczema", "atopic", "dermatitis", "rash", "itchy", "skin"],
    "basal-insulin": ["insulin", "hypoglycaemia", "hypoglycemia", "blood sugar", "glucose"],
    "chronic coronary": ["chest pain", "angina", "coronary", "stent", "heart attack"],
    "chronic obstructive": ["copd", "chronic obstructive", "emphysema", "bronchitis"],
    "ckd": ["ckd", "kidney", "renal"],
    "dengue": ["dengue", "mosquito", "platelet"],
    "diabetes": ["diabetes", "t2dm", "blood sugar", "glucose", "sglt2", "glp"],
    "foot-assessment": ["foot", "feet", "toe", "neuropathy"],
    "gdm": ["gestational diabetes", "gdm", "pregnancy", "pregnant"],
    "gout": ["gout", "allopurinol", "febuxostat", "uric", "flare"],
    "headache": ["headache", "migraine", "head injury", "hit my head"],
    "hypertension": ["hypertension", "high blood pressure", "blood pressure", "bp"],
    "knee": ["knee", "osteoarthritis", "arthritis", "joint pain"],
    "lipid": ["cholesterol", "lipid", "statin", "triglyceride"],
    "low-back": ["back pain", "low back", "sciatica"],
    "oral-anticoagulation": ["atrial fibrillation", "af", "anticoagulant", "blood thinner"],
    "osteoporosis": ["osteoporosis", "bone density", "fracture"],
    "pre-diabetes": ["prediabetes", "pre diabetes", "blood sugar", "glucose"],
    "stroke": ["stroke", "aphasia", "speech", "rehabilitation", "mobility"],
    "uti": ["uti", "urinary", "bladder infection", "kidney infection", "nitrofurantoin"],
    "vte": ["vte", "blood clot", "dvt", "pulmonary embolism", "anticoagulant"],
    "x-ray": ["chest xray", "chest x-ray", "cxr", "x ray"],
}
CATEGORY_BY_DOCUMENT = {
    "asthma-management": "asthma",
    "atopic dermatitis": "atopic dermatitis",
    "supplement on additional treatment options": "atopic dermatitis",
    "supplementary guide for ad": "atopic dermatitis",
    "initiating-basal-insulin": "basal insulin in type 2 diabetes",
    "chronic coronary syndrome": "chronic coronary syndrome",
    "chronic obstructive pulmonary disease": "copd",
    "ckd": "chronic kidney disease",
    "dengue": "dengue",
    "foot-assessment": "diabetic foot assessment",
    "generalised anxiety disorder": "generalised anxiety disorder",
    "gdm": "gestational diabetes",
    "gout": "gout",
    "headache": "headache imaging",
    "hypertension": "hypertension",
    "knee osteoarthritis": "knee osteoarthritis",
    "lipid management": "lipid management",
    "low-back-pain": "low back pain imaging",
    "major depressive disorder": "major depressive disorder",
    "managing-pre-diabetes": "pre-diabetes",
    "oral-anticoagulation": "atrial fibrillation anticoagulation",
    "osteoporosis": "osteoporosis",
    "stroke rehabilitation": "stroke rehabilitation",
    "t2dm-personalising-medications": "type 2 diabetes medications",
    "uti-appropriate-diagnosis": "urinary tract infection",
    "vte": "venous thromboembolism",
    "chest-x-ray": "chest x-ray ordering",
}


@dataclass(frozen=True)
class CheckResult:
    rule_id: str
    title: str
    status: str
    severity: str
    message: str
    evidence: list[str]
    source: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
            "source": self.source,
        }


def _disease_category_from_document(document: str) -> str:
    document = document.casefold()
    for needle, category in CATEGORY_BY_DOCUMENT.items():
        if needle in document:
            return category
    return Path(document).stem


def load_rules(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate rules from a JSON rule set."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Rule file must contain a 'rules' list.")

    seen: set[str] = set()
    for rule in rules:
        rule_id = rule.get("id")
        if not rule_id or rule_id in seen:
            raise ValueError(f"Rule IDs must be present and unique: {rule_id!r}")
        seen.add(rule_id)
    return rules


def disease_category(rule: dict[str, Any]) -> str:
    """Return the disease category inferred from the rule source document."""
    return str(rule.get("disease_category") or _disease_category_from_document(str(rule.get("source", {}).get("document", ""))))


def guardrail_categories(rules: list[dict[str, Any]] | None = None) -> list[str]:
    """Return known disease categories for LLM classification."""
    return sorted({disease_category(rule) for rule in (rules or _default_rules())})


def guardrail_category_prompt(question: str, rules: list[dict[str, Any]] | None = None) -> str:
    """Build a constrained classification prompt for selecting guardrail categories."""
    categories = "\n".join(f"- {category}" for category in guardrail_categories(rules))
    return (
        "Classify the patient question into up to three disease categories for guideline guardrails.\n"
        "Return only category names from the list, separated by commas, or none if no category applies.\n\n"
        f"Categories:\n{categories}\n\n"
        f"Question: {question}\n\n"
        "Categories:"
    )


def normalize_guardrail_category(category: str | None, rules: list[dict[str, Any]] | None = None) -> str | None:
    """Map a model response to a known category, or None."""
    categories = normalize_guardrail_categories(category, rules)
    return categories[0] if categories else None


def normalize_guardrail_categories(category: str | list[str] | tuple[str, ...] | None, rules: list[dict[str, Any]] | None = None) -> list[str]:
    """Map a model response to up to three known categories."""
    if not category:
        return []
    if isinstance(category, (list, tuple)):
        matched = []
        for item in category:
            for normalized in normalize_guardrail_categories(item, rules):
                if normalized not in matched:
                    matched.append(normalized)
        return matched[:3]

    text = category
    cleaned = text.casefold().strip()
    if re.sub(r"[^a-z0-9]+", "", cleaned) in {"", "none", "na", "notapplicable", "nocategory"}:
        return []

    known = {known.casefold(): known for known in guardrail_categories(rules)}
    matched = []
    for key, value in known.items():
        if re.search(rf"\b{re.escape(key)}\b", cleaned):
            matched.append(value)
    return matched[:3]


def rules_for_category(category: str | list[str] | tuple[str, ...] | None, rules: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return all rules belonging to selected disease categories."""
    normalized = normalize_guardrail_categories(category, rules)
    if not normalized:
        return []
    normalized_set = set(normalized)
    return [
        rule
        for rule in (rules or _default_rules())
        if disease_category(rule) in normalized_set
    ]


def _dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for rule in rules:
        if rule["id"] in seen:
            continue
        seen.add(rule["id"])
        deduped.append(rule)
    return deduped


def _matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def _matched_text(patterns: list[str], text: str) -> list[str]:
    evidence = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            evidence.append(match.group(0))
    return evidence


def _strictly_applicable(rule: dict[str, Any], question: str) -> bool:
    conditions = rule.get("applies_when", {})
    all_patterns = conditions.get("question_all", [])
    any_patterns = conditions.get("question_any", [])
    none_patterns = conditions.get("question_none", [])
    relaxed_patterns = conditions.get("question_relaxed_any", [])

    return (
        not any(_matches(pattern, question) for pattern in none_patterns)
        and (
            (
                all(_matches(pattern, question) for pattern in all_patterns)
                and (
                    not any_patterns
                    or any(_matches(pattern, question) for pattern in any_patterns)
                )
            )
            or any(_matches(pattern, question) for pattern in relaxed_patterns)
        )
    )


def _is_applicable(rule: dict[str, Any], question: str) -> bool:
    return _strictly_applicable(rule, question)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 2 and token not in STOPWORDS
    }


def _rule_search_text(rule: dict[str, Any]) -> str:
    source = rule.get("source", {})
    applies_when = rule.get("applies_when", {})
    check = rule.get("check", {})
    parts = [
        rule.get("title", ""),
        rule.get("message", ""),
        source.get("document", ""),
        source.get("recommendation", ""),
    ]
    for patterns in applies_when.values():
        parts.extend(patterns)
    for patterns in check.values():
        parts.extend(patterns)
    return " ".join(parts)


def _topic_aliases(rule: dict[str, Any]) -> list[str]:
    document = str(rule.get("source", {}).get("document", "")).casefold()
    aliases = []
    for topic, topic_aliases in TOPIC_ALIASES.items():
        if topic in document:
            aliases.extend(topic_aliases)
    return aliases


def _topic_alias_match_count(rule: dict[str, Any], question: str) -> int:
    aliases = _topic_aliases(rule)
    return sum(_matches(re.escape(alias), question) for alias in aliases)


def _condition_match_counts(rule: dict[str, Any], question: str) -> tuple[int, int, int]:
    conditions = rule.get("applies_when", {})
    all_patterns = conditions.get("question_all", [])
    any_patterns = conditions.get("question_any", [])
    relaxed_patterns = conditions.get("question_relaxed_any", [])
    all_matches = sum(_matches(pattern, question) for pattern in all_patterns)
    any_matches = sum(_matches(pattern, question) for pattern in any_patterns)
    relaxed_matches = sum(_matches(pattern, question) for pattern in relaxed_patterns)
    return all_matches, any_matches, relaxed_matches


def _fuzzy_score(rule: dict[str, Any], question: str) -> float:
    conditions = rule.get("applies_when", {})
    none_patterns = conditions.get("question_none", [])
    if any(_matches(pattern, question) for pattern in none_patterns):
        return 0.0
    topic_matches = _topic_alias_match_count(rule, question)
    if topic_matches == 0:
        return 0.0
    all_patterns = conditions.get("question_all", [])
    all_matches, any_matches, relaxed_matches = _condition_match_counts(rule, question)
    matched_conditions = all_matches + any_matches + relaxed_matches
    required_all_matches = min(MIN_FUZZY_CONDITION_MATCHES, len(all_patterns))
    if all_patterns and all_matches < required_all_matches:
        return 0.0
    if matched_conditions < MIN_FUZZY_CONDITION_MATCHES and not (
        rule.get("severity") != "critical"
        and matched_conditions == 1
        and topic_matches >= MIN_STRONG_TOPIC_ALIAS_MATCHES
    ):
        return 0.0

    question_tokens = _tokens(question)
    if not question_tokens:
        return 0.0

    rule_tokens = _tokens(_rule_search_text(rule))
    overlap = question_tokens & rule_tokens
    if not overlap:
        return 0.0

    score = float(len(overlap))
    severity = rule.get("severity")
    if severity == "critical":
        score += 2.0
    elif severity == "high":
        score += 1.0

    score += matched_conditions * 2.0
    score += topic_matches * 0.75

    title_tokens = _tokens(str(rule.get("title", "")))
    message_tokens = _tokens(str(rule.get("message", "")))
    score += len(question_tokens & title_tokens) * 0.75
    score += len(question_tokens & message_tokens) * 0.5
    return score


def applicable_rules(question: str, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return fixed rules matched by strict patterns plus deterministic relevance scoring."""
    strict_matches = [rule for rule in rules if _strictly_applicable(rule, question)]
    strict_ids = {rule["id"] for rule in strict_matches}
    scored_matches = []
    for rule in rules:
        if rule["id"] in strict_ids:
            continue
        score = _fuzzy_score(rule, question)
        if score >= MIN_FUZZY_SCORE:
            scored_matches.append((score, rule))

    scored_matches.sort(
        key=lambda item: (
            item[0],
            {"critical": 3, "high": 2, "medium": 1}.get(item[1].get("severity"), 0),
        ),
        reverse=True,
    )
    fuzzy_matches = [rule for _, rule in scored_matches[:MAX_FUZZY_RULES]]
    return strict_matches + fuzzy_matches


@cache
def _default_rules() -> list[dict[str, Any]]:
    return load_rules(DEFAULT_RULES_PATH)


def build_guardrail_context(question: str) -> str:
    """Build hidden guideline instructions for rules applicable to a question."""
    matched_rules = applicable_rules(question, _default_rules())
    if not matched_rules:
        return ""

    instructions = []
    for rule in matched_rules:
        source = rule["source"]
        instructions.append(f'- {rule["message"]} ' f'(Source: {source["document"]}, page {source["page"]}, ' f'{source["recommendation"]}.)')

    return "\n\nRelevant clinical guideline constraints:\n" + "\n".join(instructions) + "\nFollow these constraints in the answer. Do not mention this hidden context or its sources."


def build_category_guardrail_context(category: str | list[str] | tuple[str, ...] | None, question: str | None = None) -> str:
    """Build hidden guideline instructions for categories, falling back to question rules."""
    matched_rules = rules_for_category(category)
    if not matched_rules and question:
        matched_rules = applicable_rules(question, _default_rules())
    if not matched_rules:
        return ""

    instructions = []
    for rule in matched_rules:
        source = rule["source"]
        instructions.append(f'- {rule["message"]} ' f'(Source: {source["document"]}, page {source["page"]}, ' f'{source["recommendation"]}.)')

    return "\n\nRelevant clinical guideline constraints:\n" + "\n".join(instructions) + "\nFollow these constraints in the answer. Do not mention this hidden context or its sources."


def guardrail_trigger_metadata(question: str) -> dict[str, Any]:
    """Return CSV-friendly metadata for guardrails matched to a question."""
    matched_rules = applicable_rules(question, _default_rules())
    rule_ids = [rule["id"] for rule in matched_rules]
    return {
        "guardrail_triggered": bool(rule_ids),
        "guardrail_rule_count": len(rule_ids),
        "guardrail_rule_ids": "|".join(rule_ids),
    }


def category_guardrail_trigger_metadata(category: str | list[str] | tuple[str, ...] | None, question: str | None = None) -> dict[str, Any]:
    """Return CSV-friendly metadata for category-injected guardrails."""
    normalized = normalize_guardrail_categories(category)
    matched_rules = rules_for_category(normalized)
    fallback_used = False
    if not matched_rules and question:
        matched_rules = applicable_rules(question, _default_rules())
        fallback_used = bool(matched_rules)
    matched_rules = _dedupe_rules(matched_rules)
    rule_ids = [rule["id"] for rule in matched_rules]
    return {
        "guardrail_category": "|".join(normalized),
        "guardrail_fallback_triggered": fallback_used,
        "guardrail_triggered": bool(rule_ids),
        "guardrail_rule_count": len(rule_ids),
        "guardrail_rule_ids": "|".join(rule_ids),
    }


def check_answer(question: str, generated_answer: str, rules: list[dict[str, Any]]) -> list[CheckResult]:
    results = []
    for rule in rules:
        common = {
            "rule_id": rule["id"],
            "title": rule["title"],
            "severity": rule["severity"],
            "message": rule["message"],
            "source": rule["source"],
        }
        if not _is_applicable(rule, question):
            results.append(CheckResult(status="not_applicable", evidence=[], **common))
            continue

        check = rule["check"]
        forbidden = check.get("forbidden_any", [])
        required = check.get("required_any", [])
        forbidden_evidence = _matched_text(forbidden, generated_answer)
        required_evidence = _matched_text(required, generated_answer)
        violated = bool(forbidden_evidence) or bool(required and not required_evidence)

        results.append(
            CheckResult(
                status="violation" if violated else "pass",
                evidence=forbidden_evidence,
                **common,
            )
        )
    return results
