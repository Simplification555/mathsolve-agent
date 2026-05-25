"""Strict JSON parsing and Pydantic schemas for MathSolve-Agent outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator


DOMAIN_VALUES = {
    "linear_algebra",
    "calculus_real_analysis",
    "complex_analysis",
    "ordinary_differential_equations",
    "partial_differential_equations",
    "probability_statistics",
    "topology",
    "functional_analysis",
    "operations_research_optimization",
    "number_theory",
    "combinatorics",
    "discrete_mathematics",
    "geometry",
    "graph_theory",
    "numerical_analysis",
    "mathematical_modeling",
    "control_dynamical_systems",
    "other",
}

DOMAIN_ALIASES = {
    "higher_algebra": "linear_algebra",
    "advanced_algebra": "linear_algebra",
    "algebra": "linear_algebra",
    "linear algebra": "linear_algebra",
    "real_analysis": "calculus_real_analysis",
    "calculus": "calculus_real_analysis",
    "analysis": "calculus_real_analysis",
    "complex analysis": "complex_analysis",
    "ode": "ordinary_differential_equations",
    "ordinary_differential_equation": "ordinary_differential_equations",
    "ordinary differential equations": "ordinary_differential_equations",
    "pde": "partial_differential_equations",
    "partial_differential_equation": "partial_differential_equations",
    "partial differential equations": "partial_differential_equations",
    "probability": "probability_statistics",
    "statistics": "probability_statistics",
    "optimization": "operations_research_optimization",
    "operations_research": "operations_research_optimization",
    "or": "operations_research_optimization",
    "discrete math": "discrete_mathematics",
    "dynamical_systems": "control_dynamical_systems",
    "control": "control_dynamical_systems",
}

ANSWER_TYPES = {
    "formula",
    "numeric",
    "proof",
    "choice",
    "set",
    "interval",
    "matrix",
    "vector",
    "tuple",
    "text",
    "other",
}
DIFFICULTIES = {"easy", "medium", "hard"}
FINAL_DIFFICULTIES = {"easy", "medium", "hard", "very_hard"}
FINAL_CONFIDENCES = {"high", "medium", "low"}
FINAL_STATUSES = {
    "solved",
    "partially_solved",
    "insufficient_information",
    "unsolved",
}
FINAL_PROBLEM_TYPES = {
    "computation",
    "proof",
    "derivation",
    "classification",
    "construction",
    "counterexample",
    "optimization",
    "equation_solving",
    "theorem_application",
    "multi_step_reasoning",
}
ERROR_TYPES = {
    "none",
    "missing_condition",
    "wrong_theorem_condition",
    "calculation_error",
    "missing_case_split",
    "answer_not_simplified",
    "not_answering_question",
    "boundary_condition_error",
    "domain_error",
    "proof_gap",
    "format_error",
    "unknown",
}
CLAIM_STATUSES = {"passed", "failed", "uncertain"}
CLAIM_CHECK_TYPES = {"symbolic", "numeric", "logical", "definition", "format", "tool", "other"}

T = TypeVar("T", bound=BaseModel)


def _trim(value: str, limit: int) -> str:
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def normalize_domain(value: Any) -> str:
    raw = str(value or "other").strip()
    key = raw.lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key)
    if key in DOMAIN_VALUES:
        return key
    spaced = raw.lower().strip()
    if spaced in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[spaced]
    return DOMAIN_ALIASES.get(key, "other")


def normalize_answer_type(value: Any) -> str:
    raw = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "number": "numeric",
        "integer": "numeric",
        "float": "numeric",
        "latex": "formula",
        "expression": "formula",
        "boolean": "choice",
        "multiple_choice": "choice",
        "range": "interval",
        "array": "matrix",
        "list": "tuple",
        "ordered_pair": "tuple",
        "ordered_tuple": "tuple",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in ANSWER_TYPES else "other"


def normalize_difficulty(value: Any) -> str:
    raw = str(value or "medium").strip().lower()
    aliases = {"simple": "easy", "normal": "medium", "difficult": "hard"}
    raw = aliases.get(raw, raw)
    return raw if raw in DIFFICULTIES else "medium"


def normalize_final_difficulty(value: Any) -> str:
    raw = str(value or "medium").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "simple": "easy",
        "normal": "medium",
        "difficult": "hard",
        "very hard": "very_hard",
        "veryhard": "very_hard",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in FINAL_DIFFICULTIES else "medium"


def normalize_error_type(value: Any) -> str:
    raw = str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "missing_boundary_condition": "boundary_condition_error",
        "boundary_error": "boundary_condition_error",
        "wrong_domain": "domain_error",
        "format": "format_error",
        "not_answering": "not_answering_question",
        "not_answered": "not_answering_question",
        "wrong_calculation": "calculation_error",
        "compute_error": "calculation_error",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in ERROR_TYPES else "unknown"


def normalize_claim_status(value: Any) -> str:
    raw = str(value or "uncertain").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pass": "passed",
        "ok": "passed",
        "valid": "passed",
        "fail": "failed",
        "invalid": "failed",
        "unknown": "uncertain",
        "not_checked": "uncertain",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in CLAIM_STATUSES else "uncertain"


def normalize_claim_check_type(value: Any) -> str:
    raw = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "math": "logical",
        "logic": "logical",
        "definition_check": "definition",
        "executable": "tool",
        "program": "tool",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in CLAIM_CHECK_TYPES else "other"


def _stringify(value: Any, limit: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _trim(value, limit)
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            text = str(value)
        return _trim(text, limit)
    return _trim(value, limit)


def _list_text(value: Any, limit: int = 300, max_items: int = 6) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify(item, limit) for item in value if _stringify(item, limit)][:max_items]
    return [_stringify(value, limit)] if _stringify(value, limit) else []


class LayerCheck(BaseModel):
    passed: bool = True
    issues: List[str] = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def issues_as_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 240) for item in value if str(item).strip()]
        return [_trim(value, 240)] if str(value).strip() else []


class ClaimCheck(BaseModel):
    claim: str = ""
    status: str = "uncertain"
    check_type: str = "other"
    reason: str = ""

    @field_validator("claim")
    @classmethod
    def claim_short(cls, value: Any) -> str:
        return _stringify(value, 300)

    @field_validator("status")
    @classmethod
    def status_allowed(cls, value: Any) -> str:
        return normalize_claim_status(value)

    @field_validator("check_type")
    @classmethod
    def check_type_allowed(cls, value: Any) -> str:
        return normalize_claim_check_type(value)

    @field_validator("reason")
    @classmethod
    def reason_short(cls, value: Any) -> str:
        return _stringify(value, 300)


class VerificationResult(BaseModel):
    passed: bool = False
    confidence: float = 0.0
    issues: List[str] = Field(default_factory=list)
    format_check: LayerCheck = Field(default_factory=LayerCheck)
    question_target_check: LayerCheck = Field(default_factory=LayerCheck)
    condition_check: LayerCheck = Field(default_factory=LayerCheck)
    result_check: LayerCheck = Field(default_factory=LayerCheck)
    judgeability_check: LayerCheck = Field(default_factory=LayerCheck)
    claim_checks: List[ClaimCheck] = Field(default_factory=list)
    error_type: str = "none"
    repair_instruction: str = ""
    corrected_answer: str = Field(default="", exclude=True)

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, value: float) -> float:
        try:
            value = float(value)
        except Exception:
            value = 0.0
        return max(0.0, min(1.0, value))

    @field_validator("issues", mode="before")
    @classmethod
    def issues_as_list(cls, value: Any) -> List[str]:
        return _list_text(value, 300, 8)

    @field_validator("claim_checks", mode="before")
    @classmethod
    def claim_checks_list(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value[:8]
        if isinstance(value, dict):
            return [value]
        return [
            {
                "claim": _stringify(value, 300),
                "status": "uncertain",
                "check_type": "other",
                "reason": "",
            }
        ] if _stringify(value, 300) else []

    @field_validator("error_type")
    @classmethod
    def error_type_allowed(cls, value: Any) -> str:
        return normalize_error_type(value)

    @field_validator("repair_instruction")
    @classmethod
    def repair_instruction_short(cls, value: str) -> str:
        return _stringify(value, 500)

    @field_validator("corrected_answer")
    @classmethod
    def corrected_answer_short(cls, value: Any) -> str:
        return _stringify(value, 1200)


class ClassificationResult(BaseModel):
    domain: str = "other"
    subtype: str = ""
    goal: str = ""
    difficulty: str = "medium"
    answer_type: str = "other"
    required_methods: List[str] = Field(default_factory=list)
    solution_plan: List[str] = Field(default_factory=list)
    possible_pitfalls: List[str] = Field(default_factory=list)
    constraints_to_check: List[str] = Field(default_factory=list)
    risk_points: List[str] = Field(default_factory=list)
    needs_case_split: bool = False
    needs_tool_verification: bool = False
    expected_answer_shape: str = ""

    @field_validator("domain")
    @classmethod
    def domain_allowed(cls, value: Any) -> str:
        return normalize_domain(value)

    @field_validator("difficulty")
    @classmethod
    def difficulty_allowed(cls, value: Any) -> str:
        return normalize_difficulty(value)

    @field_validator("answer_type")
    @classmethod
    def answer_type_allowed(cls, value: Any) -> str:
        return normalize_answer_type(value)

    @field_validator(
        "required_methods",
        "solution_plan",
        "possible_pitfalls",
        "constraints_to_check",
        "risk_points",
        mode="before",
    )
    @classmethod
    def list_fields(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 300) for item in value if str(item).strip()]
        return [_trim(value, 300)] if str(value).strip() else []

    @field_validator("subtype", "goal", "expected_answer_shape")
    @classmethod
    def text_short(cls, value: str) -> str:
        return _trim(value, 120)


class CandidateSolution(BaseModel):
    candidate_id: str = "A"
    method: str = ""
    reasoning_summary: str = ""
    key_steps: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    target: str = ""
    derivation_steps: List[str] = Field(default_factory=list)
    checkable_claims: List[str] = Field(default_factory=list)
    final_answer: str = ""
    answer_type: str = "other"
    verification_code: str = ""

    @field_validator("answer_type")
    @classmethod
    def answer_type_allowed(cls, value: Any) -> str:
        return normalize_answer_type(value)

    @field_validator("key_steps", mode="before")
    @classmethod
    def key_steps_list(cls, value: Any) -> List[str]:
        return _list_text(value, 260, 5)

    @field_validator("assumptions", "derivation_steps", "checkable_claims", mode="before")
    @classmethod
    def checkable_lists(cls, value: Any) -> List[str]:
        return _list_text(value, 300, 6)

    @field_validator("final_answer", mode="before")
    @classmethod
    def final_answer_short(cls, value: Any) -> str:
        return _stringify(value, 1200)

    @field_validator("reasoning_summary", "target")
    @classmethod
    def reasoning_short(cls, value: Any) -> str:
        return _stringify(value, 800)

    @field_validator("verification_code")
    @classmethod
    def code_short(cls, value: Any) -> str:
        return _stringify(value, 4000)


class SelectionResult(BaseModel):
    selected_candidate_id: str = "A"
    answer: str = ""
    reasoning_summary: str = ""
    key_steps: List[str] = Field(default_factory=list)
    learning_hint: str = ""
    verification: VerificationResult = Field(default_factory=VerificationResult)

    @field_validator("key_steps", mode="before")
    @classmethod
    def key_steps_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 260) for item in value if str(item).strip()][:5]
        return [_trim(value, 260)] if str(value).strip() else []

    @field_validator("answer")
    @classmethod
    def answer_short(cls, value: Any) -> str:
        return _stringify(value, 1200)

    @field_validator("reasoning_summary", "learning_hint")
    @classmethod
    def text_short(cls, value: Any) -> str:
        return _stringify(value, 800)


class MathSolution(BaseModel):
    """Final judgeable JSON object for one math problem."""

    problem_id: str
    domain: str = "other"
    answer: str = "unable_to_determine"
    answer_type: str = "other"
    reasoning_summary: str = ""
    key_steps: List[str] = Field(default_factory=list)
    learning_hint: str = ""
    verification: VerificationResult = Field(default_factory=VerificationResult)

    @field_validator("problem_id")
    @classmethod
    def problem_id_not_empty(cls, value: Any) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("problem_id cannot be empty")
        return value

    @field_validator("domain")
    @classmethod
    def domain_allowed(cls, value: Any) -> str:
        return normalize_domain(value)

    @field_validator("answer_type")
    @classmethod
    def answer_type_allowed(cls, value: Any) -> str:
        return normalize_answer_type(value)

    @field_validator("answer", mode="before")
    @classmethod
    def answer_not_empty(cls, value: Any) -> str:
        value = _stringify(value, 1200)
        return value or "unable_to_determine"

    @field_validator("reasoning_summary", "learning_hint")
    @classmethod
    def summary_short(cls, value: Any) -> str:
        return _stringify(value, 800)

    @field_validator("key_steps", mode="before")
    @classmethod
    def key_steps_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 260) for item in value if str(item).strip()][:5]
        return [_trim(value, 260)] if str(value).strip() else []

    @property
    def final_answer(self) -> str:
        return self.answer

    @property
    def is_solved(self) -> bool:
        return bool(self.verification.passed and self.answer != "unable_to_determine")

    @property
    def logs(self) -> List[Any]:
        return []


class FinalVerification(BaseModel):
    checked: bool = False
    summary: str = ""
    risk_points: List[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def summary_short(cls, value: Any) -> str:
        return _stringify(value, 800)

    @field_validator("risk_points", mode="before")
    @classmethod
    def risk_points_list(cls, value: Any) -> List[str]:
        return _list_text(value, 260, 8)


class EducationalExplanation(BaseModel):
    key_insight: str = ""
    common_mistakes: List[str] = Field(default_factory=list)
    transfer_hint: str = ""

    @field_validator("key_insight", "transfer_hint")
    @classmethod
    def text_short(cls, value: Any) -> str:
        return _stringify(value, 500)

    @field_validator("common_mistakes", mode="before")
    @classmethod
    def mistakes_list(cls, value: Any) -> List[str]:
        return _list_text(value, 260, 6)


class CompetitionSolution(BaseModel):
    """Official final JSON schema used for math-agent competition output."""

    problem_id: str = ""
    domain: List[str] = Field(default_factory=list)
    problem_type: List[str] = Field(default_factory=list)
    difficulty_estimate: str = "medium"
    final_answer: str = ""
    answer_latex: str = ""
    solution_summary: str = ""
    key_reasoning: List[str] = Field(default_factory=list)
    verification: FinalVerification = Field(default_factory=FinalVerification)
    educational_explanation: EducationalExplanation = Field(
        default_factory=EducationalExplanation
    )
    confidence: str = "low"
    status: str = "unsolved"

    @field_validator("problem_id")
    @classmethod
    def problem_id_string(cls, value: Any) -> str:
        return _stringify(value, 120)

    @field_validator("domain", mode="before")
    @classmethod
    def domain_list(cls, value: Any) -> List[str]:
        raw_items = _list_text(value, 120, 6)
        domains: List[str] = []
        for item in raw_items:
            normalized = normalize_domain(item)
            if normalized == "other" and item.lower() not in {"other", ""}:
                domain = _trim(item, 120)
            else:
                domain = normalized
            if domain and domain not in domains:
                domains.append(domain)
        return domains

    @field_validator("problem_type", mode="before")
    @classmethod
    def problem_type_list(cls, value: Any) -> List[str]:
        raw_items = _list_text(value, 120, 6)
        aliases = {
            "calculate": "computation",
            "calculation": "computation",
            "numeric": "computation",
            "compute": "computation",
            "prove": "proof",
            "证明": "proof",
            "derive": "derivation",
            "solve": "equation_solving",
            "equation": "equation_solving",
            "equation solving": "equation_solving",
            "theorem": "theorem_application",
            "multi step": "multi_step_reasoning",
            "multi-step": "multi_step_reasoning",
        }
        problem_types: List[str] = []
        for item in raw_items:
            key = item.strip().lower().replace("-", "_").replace(" ", "_")
            key = aliases.get(item.strip().lower(), aliases.get(key, key))
            if key in FINAL_PROBLEM_TYPES and key not in problem_types:
                problem_types.append(key)
        return problem_types or ["multi_step_reasoning"]

    @field_validator("difficulty_estimate")
    @classmethod
    def difficulty_allowed(cls, value: Any) -> str:
        return normalize_final_difficulty(value)

    @field_validator("final_answer", "answer_latex", "solution_summary", mode="before")
    @classmethod
    def final_text_fields(cls, value: Any) -> str:
        return _stringify(value, 1200)

    @field_validator("key_reasoning", mode="before")
    @classmethod
    def reasoning_list(cls, value: Any) -> List[str]:
        return _list_text(value, 300, 8)

    @field_validator("confidence")
    @classmethod
    def confidence_allowed(cls, value: Any) -> str:
        raw = str(value or "low").strip().lower()
        return raw if raw in FINAL_CONFIDENCES else "low"

    @field_validator("status")
    @classmethod
    def status_allowed(cls, value: Any) -> str:
        raw = str(value or "unsolved").strip().lower().replace("-", "_").replace(" ", "_")
        return raw if raw in FINAL_STATUSES else "unsolved"


class LogEntry(BaseModel):
    """Backward-compatible compact log entry for callers that still import it."""

    step: int = 1
    thought: str = ""
    action: str = ""
    observation: str = ""


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from free-form text."""

    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    json_str = extract_json_from_text(raw_text)
    if json_str is None:
        raise ValueError("No JSON object found in model output")
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object")
    return data


def parse_model(raw_text: str, model: Type[T]) -> T:
    data = parse_json_object(raw_text)
    return model(**data)


def _coerce_solution_payload(data: Dict[str, Any], problem_id: str) -> Dict[str, Any]:
    data = dict(data)
    data["problem_id"] = str(data.get("problem_id") or problem_id)
    if "answer" not in data and "final_answer" in data:
        data["answer"] = data.get("final_answer")
    if "reasoning_summary" not in data and "reasoning_process" in data:
        data["reasoning_summary"] = data.get("reasoning_process")
    if "verification" not in data:
        data["verification"] = {
            "passed": bool(data.get("is_solved", False)),
            "confidence": 0.0,
            "issues": [],
            "corrected_answer": data.get("answer", ""),
        }
    data.setdefault("domain", "other")
    data.setdefault("answer", "unable_to_determine")
    data.setdefault("answer_type", "other")
    data.setdefault("reasoning_summary", "")
    data.setdefault("key_steps", [])
    data.setdefault("learning_hint", "")
    return data


def parse_and_validate(raw_text: str, problem_id: str, max_retries: int = 2) -> MathSolution:
    """Parse a model response or JSON string into the final MathSolution schema."""

    del max_retries
    data = _coerce_solution_payload(parse_json_object(raw_text), problem_id)
    try:
        solution = MathSolution(**data)
    except ValidationError as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc
    if solution.problem_id != str(problem_id):
        solution.problem_id = str(problem_id)
    return solution


def validate_solution_dict(data: Dict[str, Any], problem_id: str) -> MathSolution:
    payload = _coerce_solution_payload(data, problem_id)
    return MathSolution(**payload)


def _get_classification_value(classification: Any, key: str, default: Any = "") -> Any:
    if classification is None:
        return default
    if isinstance(classification, BaseModel):
        return getattr(classification, key, default)
    if isinstance(classification, dict):
        return classification.get(key, default)
    return default


def _infer_final_problem_types(
    solution: MathSolution,
    classification: Optional[Any] = None,
) -> List[str]:
    answer_type = str(
        solution.answer_type
        or _get_classification_value(classification, "answer_type", "")
        or ""
    )
    domain = str(solution.domain or _get_classification_value(classification, "domain", ""))
    subtype = str(_get_classification_value(classification, "subtype", ""))
    goal = str(_get_classification_value(classification, "goal", ""))
    text = " ".join([answer_type, domain, subtype, goal]).lower()

    problem_types: List[str] = []
    if any(token in text for token in ("proof", "prove", "证明", "show that")):
        problem_types.append("proof")
    if any(token in text for token in ("counterexample", "反例")):
        problem_types.append("counterexample")
    if any(token in text for token in ("construct", "构造")):
        problem_types.append("construction")
    if any(token in text for token in ("classify", "classification", "分类")):
        problem_types.append("classification")
    if "optimization" in text or "operations_research" in text or "最优" in text:
        problem_types.append("optimization")
    if any(token in text for token in ("equation", "root", "roots", "solve", "方程", "解集")):
        problem_types.append("equation_solving")
    if answer_type in {"formula", "numeric", "choice", "set", "interval", "matrix", "vector", "tuple"}:
        problem_types.append("computation")
    if any(token in text for token in ("derive", "derivation", "推导")):
        problem_types.append("derivation")

    if not problem_types:
        problem_types.append("multi_step_reasoning")

    unique: List[str] = []
    for item in problem_types:
        if item in FINAL_PROBLEM_TYPES and item not in unique:
            unique.append(item)
    return unique[:4] or ["multi_step_reasoning"]


def _final_confidence(solution: MathSolution) -> str:
    confidence = float(solution.verification.confidence or 0.0)
    if solution.verification.passed and confidence >= 0.80:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def _final_status(solution: MathSolution) -> str:
    answer = str(solution.answer or "").strip()
    issues_text = " ".join(solution.verification.issues).lower()
    if not answer or answer == "unable_to_determine":
        if any(token in issues_text for token in ("insufficient", "missing problem", "条件不足")):
            return "insufficient_information"
        return "unsolved"
    if solution.verification.passed:
        return "solved"
    if any(token in issues_text for token in ("insufficient", "条件不足", "缺少条件")):
        return "insufficient_information"
    return "partially_solved"


def _answer_latex(solution: MathSolution) -> str:
    try:
        from .normalizer import normalize_answer

        normalized = normalize_answer(solution.answer, solution.answer_type)
        return normalized.latex or solution.answer
    except Exception:
        return solution.answer


def _verification_summary(solution: MathSolution, status: str) -> str:
    if status == "solved":
        parts = ["已检查题目目标、条件使用、最终答案格式和可判分性"]
        if solution.verification.confidence:
            parts.append(f"内部置信度为 {solution.verification.confidence:.2f}")
        return "；".join(parts) + "。"
    if solution.verification.issues:
        return "未完全通过校验：" + "；".join(solution.verification.issues[:3])
    return "答案未达到 solved 标准，保留为非完全解决状态。"


def _coerce_competition_payload(data: Dict[str, Any], problem_id: str = "") -> Dict[str, Any]:
    payload = dict(data)
    payload["problem_id"] = str(payload.get("problem_id") or problem_id or "")
    if "final_answer" not in payload and "answer" in payload:
        payload["final_answer"] = payload.get("answer")
    payload.setdefault("final_answer", "")
    payload.setdefault("answer_latex", payload.get("final_answer", ""))
    payload.setdefault("domain", payload.get("domain", []))
    payload.setdefault("problem_type", payload.get("problem_type", []))
    payload.setdefault("difficulty_estimate", payload.get("difficulty", "medium"))
    if "solution_summary" not in payload:
        payload["solution_summary"] = payload.get("reasoning_summary", "")
    if "key_reasoning" not in payload:
        payload["key_reasoning"] = payload.get("key_steps", [])
    payload.setdefault(
        "verification",
        {
            "checked": False,
            "summary": "",
            "risk_points": [],
        },
    )
    payload.setdefault(
        "educational_explanation",
        {
            "key_insight": "",
            "common_mistakes": [],
            "transfer_hint": "",
        },
    )
    payload.setdefault("confidence", "low")
    payload.setdefault("status", "unsolved")
    return payload


def validate_competition_solution_dict(
    data: Dict[str, Any],
    problem_id: str = "",
) -> CompetitionSolution:
    payload = _coerce_competition_payload(data, problem_id)
    return CompetitionSolution(**payload)


def parse_competition_and_validate(raw_text: str, problem_id: str = "") -> CompetitionSolution:
    data = parse_json_object(raw_text)
    return validate_competition_solution_dict(data, problem_id)


def solution_to_competition_dict(
    solution: MathSolution,
    classification: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert the internal legacy solution to the strict official final schema."""

    status = _final_status(solution)
    confidence = _final_confidence(solution)
    difficulty = normalize_final_difficulty(
        _get_classification_value(classification, "difficulty", "medium")
    )
    classification_risks = _list_text(
        _get_classification_value(classification, "risk_points", []), 260, 6
    )
    possible_mistakes = _list_text(
        _get_classification_value(classification, "possible_pitfalls", []), 260, 6
    )
    risk_points = list(solution.verification.issues[:6])
    for item in classification_risks:
        if status != "solved" and item not in risk_points:
            risk_points.append(item)
    common_mistakes = []
    for item in possible_mistakes + list(solution.verification.issues):
        if item and item not in common_mistakes:
            common_mistakes.append(item)

    final_answer = solution.answer
    if final_answer == "unable_to_determine":
        final_answer = "无法可靠确定"
    key_reasoning = solution.key_steps or []
    key_insight = solution.learning_hint or (key_reasoning[0] if key_reasoning else "")
    transfer_hint = solution.learning_hint or "同类题应先锁定题目目标，再逐项检查定义域、条件和最终答案格式。"

    payload = {
        "problem_id": solution.problem_id,
        "domain": [solution.domain] if solution.domain else [],
        "problem_type": _infer_final_problem_types(solution, classification),
        "difficulty_estimate": difficulty,
        "final_answer": final_answer,
        "answer_latex": _answer_latex(solution),
        "solution_summary": solution.reasoning_summary,
        "key_reasoning": key_reasoning,
        "verification": {
            "checked": True if status == "solved" else bool(solution.verification.confidence),
            "summary": _verification_summary(solution, status),
            "risk_points": risk_points[:8],
        },
        "educational_explanation": {
            "key_insight": key_insight,
            "common_mistakes": common_mistakes[:6],
            "transfer_hint": transfer_hint,
        },
        "confidence": confidence,
        "status": status,
    }
    return CompetitionSolution(**payload).model_dump(mode="json")


def fallback_solution(
    problem_id: str,
    reason: str = "",
    domain: str = "other",
    answer_type: str = "other",
) -> MathSolution:
    issue = _trim(reason, 300) if reason else "Unable to determine a reliable answer"
    return MathSolution(
        problem_id=str(problem_id),
        domain=domain,
        answer="unable_to_determine",
        answer_type=answer_type,
        reasoning_summary="The system could not produce a reliable final answer.",
        key_steps=[],
        learning_hint="Check the problem conditions and rerun with a stricter method.",
        verification=VerificationResult(
            passed=False,
            confidence=0.0,
            issues=[issue],
            corrected_answer="unable_to_determine",
        ),
    )


def solution_to_json(solution: MathSolution, indent: Optional[int] = 2) -> str:
    """Serialize a final solution as strict JSON."""

    return json.dumps(solution.model_dump(mode="json"), ensure_ascii=False, indent=indent)


def solution_to_competition_json(
    solution: MathSolution,
    classification: Optional[Any] = None,
    indent: Optional[int] = 2,
) -> str:
    """Serialize a solution with the official competition JSON schema."""

    return json.dumps(
        solution_to_competition_dict(solution, classification=classification),
        ensure_ascii=False,
        indent=indent,
    )


def model_to_json(data: BaseModel, indent: Optional[int] = 2) -> str:
    return json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=indent)


def build_json_prompt(schema: Type[BaseModel] = MathSolution) -> str:
    fields = []
    for name, field in schema.model_fields.items():
        desc = field.description or field.annotation
        fields.append(f'- "{name}": {desc}')
    return "Return exactly one JSON object with these fields:\n" + "\n".join(fields)
