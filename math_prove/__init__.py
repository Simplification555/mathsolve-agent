"""MathSolve-Agent package."""

from .parser import (
    CandidateSolution,
    ClaimCheck,
    ClassificationResult,
    CompetitionSolution,
    EducationalExplanation,
    FinalVerification,
    LogEntry,
    MathSolution,
    SelectionResult,
    VerificationResult,
    fallback_solution,
    parse_competition_and_validate,
    parse_and_validate,
    solution_to_competition_dict,
    solution_to_competition_json,
    solution_to_json,
    validate_competition_solution_dict,
)
from .config import SolverConfig, load_config
from .normalizer import AnswerForms, EquivalenceResult, equivalent_answers, normalize_answer
from .validator import ValidationReport, validate_results


def __getattr__(name):
    if name == "MathSolverAgent":
        from .agent import MathSolverAgent

        return MathSolverAgent
    if name == "MathSandbox":
        from .sandbox import MathSandbox

        return MathSandbox
    raise AttributeError(name)


__all__ = [
    "MathSandbox",
    "MathSolverAgent",
    "MathSolution",
    "CompetitionSolution",
    "ClassificationResult",
    "CandidateSolution",
    "ClaimCheck",
    "VerificationResult",
    "FinalVerification",
    "EducationalExplanation",
    "SelectionResult",
    "LogEntry",
    "fallback_solution",
    "parse_and_validate",
    "parse_competition_and_validate",
    "validate_competition_solution_dict",
    "solution_to_json",
    "solution_to_competition_dict",
    "solution_to_competition_json",
    "SolverConfig",
    "load_config",
    "AnswerForms",
    "EquivalenceResult",
    "normalize_answer",
    "equivalent_answers",
    "ValidationReport",
    "validate_results",
]
