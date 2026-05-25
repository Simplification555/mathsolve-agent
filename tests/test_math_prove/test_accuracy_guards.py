import json

from math_prove.agent import MathSolverAgent
from math_prove.config import load_config, SolverConfig
from math_prove.normalizer import equivalent_answers
from math_prove.parser import (
    CandidateSolution,
    ClassificationResult,
    MathSolution,
    VerificationResult,
    parse_competition_and_validate,
    solution_to_competition_json,
)


def test_common_math_formats_are_equivalent():
    cases = [
        (r"\sqrt{2}", "2**0.5", "numeric"),
        (r"\sin{x}", "sin(x)", "formula"),
        (r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", "[[1,2],[3,4]]", "matrix"),
        ("B. 3/2", "B", "choice"),
        ("Option B", "B", "choice"),
        ("[2,1]", "{1,2}", "set"),
        ("1, 2", "(1,2)", "tuple"),
    ]

    for prediction, expected, answer_type in cases:
        result = equivalent_answers(prediction, expected, answer_type)
        assert result.equivalent, result


def test_tool_mismatch_can_fail_candidate_when_strict_equivalence_enabled():
    agent = object.__new__(MathSolverAgent)
    agent._config = SolverConfig(
        enable_llm_verify=False,
        enable_equivalence_check=True,
        equivalence_can_fail_candidate=True,
    )

    classification = ClassificationResult(
        domain="calculus_real_analysis",
        answer_type="numeric",
    )
    candidate = CandidateSolution(
        candidate_id="A",
        final_answer="1",
        answer_type="numeric",
    )
    tool_result = {
        "passed": True,
        "check_output": "2",
        "has_check_marker": True,
    }
    run_log = {}

    verification = agent._verify_candidate(
        problem="Compute 1+1.",
        classification=classification,
        candidate=candidate,
        tool_result=tool_result,
        run_log=run_log,
    )

    assert verification.passed is False
    assert verification.error_type == "calculation_error"
    assert run_log["local_equivalence_checks"][0]["equivalent"] is False


def test_extract_stage_does_not_upgrade_failed_candidate_verification():
    agent = object.__new__(MathSolverAgent)
    agent._config = SolverConfig(enable_extract_stage=False)
    agent._problem_timeout = 240.0

    classification = ClassificationResult(
        domain="calculus_real_analysis",
        answer_type="numeric",
    )
    candidate = CandidateSolution(
        candidate_id="A",
        final_answer="4",
        answer_type="numeric",
        reasoning_summary="Computed directly.",
        key_steps=["Add the terms."],
    )
    verification = VerificationResult(
        passed=False,
        confidence=0.9,
        issues=["tool verification failed"],
        error_type="calculation_error",
        corrected_answer="5",
    )

    solution = agent._extract_answer(
        problem_id="p1",
        problem="Compute 2+3.",
        classification=classification,
        candidate=candidate,
        verification=verification,
        run_log={},
        start_time=0.0,
    )

    assert solution.answer == "4"
    assert solution.verification.passed is False


def test_official_stable_keeps_accuracy_guards_enabled():
    config = load_config(ablation="official_stable")

    assert config.official_mode is True
    assert config.enable_sandbox is True
    assert config.enable_equivalence_check is True
    assert config.equivalence_can_fail_candidate is True
    assert config.verifier_can_overwrite_answer is True


def test_competition_schema_export_is_strict_and_judgeable():
    solution = MathSolution(
        problem_id="p2",
        domain="calculus_real_analysis",
        answer="2",
        answer_type="numeric",
        reasoning_summary="直接计算得到目标值。",
        key_steps=["确认题目要求计算数值。", "代入并化简得到 2。"],
        learning_hint="同类题要先确认最终要求的量。",
        verification=VerificationResult(
            passed=True,
            confidence=0.91,
            issues=[],
        ),
    )
    classification = {
        "domain": "calculus_real_analysis",
        "difficulty": "easy",
        "answer_type": "numeric",
        "possible_pitfalls": ["不要把中间式当最终答案。"],
    }

    raw = solution_to_competition_json(solution, classification, indent=None)
    data = json.loads(raw)
    parsed = parse_competition_and_validate(raw, "p2")

    assert data["final_answer"] == "2"
    assert data["answer_latex"] == "2"
    assert data["domain"] == ["calculus_real_analysis"]
    assert "computation" in data["problem_type"]
    assert data["verification"]["checked"] is True
    assert data["confidence"] == "high"
    assert data["status"] == "solved"
    assert parsed.problem_id == "p2"
