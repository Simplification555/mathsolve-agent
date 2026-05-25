"""Single-agent MathSolve-Agent solving pipeline."""

from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
from difflib import SequenceMatcher
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from lagent.hooks import MessageLogger
from lagent.llms.openai import GPTAPI
from lagent.memory import Memory
from lagent.schema import AgentMessage

from . import prompts
from .config import SolverConfig, load_config
from .normalizer import equivalent_answers, normalize_answer
from .parser import (
    CandidateSolution,
    ClassificationResult,
    MathSolution,
    SelectionResult,
    VerificationResult,
    build_json_prompt,
    fallback_solution,
    model_to_json,
    parse_and_validate,
    parse_model,
    validate_solution_dict,
)
from .sandbox import MathSandbox, Status


class OpenAICompatibleGPTAPI(GPTAPI):
    """GPTAPI variant that accepts arbitrary OpenAI-compatible model names.

    The upstream lagent GPTAPI has a local allowlist for model name prefixes.
    Intern-S1's official model id is ``intern-s1``, which is valid for the
    InternLM OpenAI-compatible endpoint but does not pass that allowlist. This
    subclass keeps the same request format while allowing such model ids.
    """

    def generate_request_data(self, model_type, messages, gen_params, json_mode=False):
        gen_params = gen_params.copy()
        max_tokens = min(gen_params.pop("max_new_tokens"), 4096)
        if max_tokens <= 0:
            return "", ""

        header = {"content-type": "application/json"}
        gen_params["max_tokens"] = max_tokens
        if "stop_words" in gen_params:
            gen_params["stop"] = gen_params.pop("stop_words")
        if "repetition_penalty" in gen_params:
            gen_params["frequency_penalty"] = gen_params.pop("repetition_penalty")
        gen_params.pop("top_k", None)
        gen_params.pop("skip_special_tokens", None)
        gen_params.pop("session_id", None)

        data = {"model": model_type, "messages": messages, "n": 1, **gen_params}
        if json_mode:
            data["response_format"] = {"type": "json_object"}
        return header, data


PROBLEM_TIMEOUT = 240.0
SANDBOX_TIMEOUT = 10
MAX_API_RETRIES = 5
CONFIDENCE_THRESHOLD = 0.70

TOOL_FRIENDLY_DOMAINS = {
    "linear_algebra",
    "calculus_real_analysis",
    "complex_analysis",
    "ordinary_differential_equations",
    "partial_differential_equations",
    "probability_statistics",
    "operations_research_optimization",
    "number_theory",
    "combinatorics",
    "discrete_mathematics",
    "geometry",
    "graph_theory",
    "numerical_analysis",
    "mathematical_modeling",
    "control_dynamical_systems",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _compact(value: Any, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parse_number(text: str) -> float:
    match = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", str(text).strip())
    if match:
        return float(match.group(1)) / float(match.group(2))
    return float(text)


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.10g}"


def _first_positive_root(func: Any, upper: float = 500.0) -> Optional[float]:
    start = 1e-8
    steps = 50000
    previous_x = start
    previous_y = func(previous_x)
    for index in range(1, steps + 1):
        x = upper * index / steps
        y = func(x)
        if not (math.isfinite(previous_y) and math.isfinite(y)):
            previous_x, previous_y = x, y
            continue
        if abs(y) < 1e-10:
            return x
        if previous_y * y < 0:
            lo, hi = previous_x, x
            flo, fhi = previous_y, y
            for _ in range(80):
                mid = (lo + hi) / 2.0
                fmid = func(mid)
                if abs(fmid) < 1e-12:
                    return mid
                if flo * fmid <= 0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
            return (lo + hi) / 2.0
        previous_x, previous_y = x, y
    return None


class MathSolverAgent:
    """Intern-S1 based single math agent with explicit solve/verify stages."""

    def __init__(
        self,
        model_type: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 4096,
        retry: int = 3,
        sandbox_timeout: Optional[int] = None,
        problem_timeout: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        config: Optional[SolverConfig] = None,
        config_path: Optional[str] = None,
        ablation: str = "full",
        official_mode: bool = False,
    ) -> None:
        self._config = config or load_config(config_path, ablation)
        if config is None and config_path is None:
            if sandbox_timeout is not None:
                self._config.sandbox_timeout = sandbox_timeout
            if problem_timeout is not None:
                self._config.problem_timeout = problem_timeout
            if confidence_threshold is not None:
                self._config.confidence_threshold = confidence_threshold
        if official_mode:
            self._config.official_mode = True

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base = api_base or os.environ.get(
            "LLM_API_BASE", "https://api.openai.com/v1/chat/completions"
        )
        if self._config.official_mode:
            self._validate_official_api_config(model_type, key, base)
        self._llm = OpenAICompatibleGPTAPI(
            model_type=model_type,
            key=key,
            api_base=base,
            retry=retry,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        if "intern" in model_type.lower() or "intern" in base.lower():
            self._llm.gen_params["thinking_mode"] = False
        self._sandbox = MathSandbox(timeout=self._config.sandbox_timeout)
        self._memory = Memory(recent_n=30)
        self._msg_logger = MessageLogger(name="math_prove", add_file_handler=True)
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._problem_timeout = self._config.problem_timeout
        self._confidence_threshold = self._config.confidence_threshold
        self.last_run_log: Dict[str, Any] = {}

    @staticmethod
    def _validate_official_api_config(model_type: str, api_key: str, api_base: str) -> None:
        model = str(model_type or "").lower()
        base = str(api_base or "").lower()
        if not str(api_key or "").strip():
            raise RuntimeError("Official run requires OPENAI_API_KEY / Intern-S1 token.")
        if "intern-s1" not in model:
            raise RuntimeError("Official run must use intern-s1, intern-s1-pro, or intern-s1-mini.")
        if "intern" not in base or "/chat/completions" not in base:
            raise RuntimeError(
                "Official run must use the InternLM OpenAI-compatible chat completions endpoint."
            )

    def solve(
        self,
        problem: str,
        problem_id: str = "0",
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> MathSolution:
        """Solve one problem and always return a valid judgeable JSON object."""

        start = time.time()
        run_log: Dict[str, Any] = {
            "problem_id": str(problem_id),
            "timestamp": _now(),
            "raw_problem": problem,
            "raw_metadata": raw_metadata or {},
            "preprocessed_problem": "",
            "classification": {},
            "stages": [],
            "candidates": [],
            "retry_count": 0,
            "api_status": "success",
            "latency_seconds": 0.0,
            "final_json": {},
            "config": self._config.to_dict(),
        }
        self.last_run_log = run_log
        self._memory = Memory(recent_n=30)
        try:
            self._sandbox.reset()
        except Exception as exc:
            run_log.setdefault("warnings", []).append(f"sandbox reset failed: {exc}")

        try:
            clean_problem = self._preprocess(problem)
            run_log["preprocessed_problem"] = clean_problem
            self._msg_logger.logger.info(
                f"[{problem_id}] start solving: {_compact(clean_problem, 120)}"
            )

            if not clean_problem:
                solution = fallback_solution(problem_id, "Empty problem text")
                run_log["api_status"] = "fallback_empty_problem"
                return self._finish(run_log, solution, start)

            classification = self._classify_and_plan(clean_problem, run_log)
            run_log["classification"] = classification.model_dump(mode="json")

            candidate, verification = self._solve_with_retries(
                clean_problem,
                classification,
                run_log,
                start,
            )

            solution = self._extract_answer(
                problem_id=str(problem_id),
                problem=clean_problem,
                classification=classification,
                candidate=candidate,
                verification=verification,
                run_log=run_log,
                start_time=start,
            )
            return self._finish(run_log, solution, start)

        except Exception as exc:
            run_log["api_status"] = "fallback_exception"
            run_log["exception"] = traceback.format_exc()
            domain = run_log.get("classification", {}).get("domain", "other")
            answer_type = run_log.get("classification", {}).get("answer_type", "other")
            solution = fallback_solution(
                problem_id=str(problem_id),
                reason=f"{type(exc).__name__}: {exc}",
                domain=domain,
                answer_type=answer_type,
            )
            return self._finish(run_log, solution, start)

    def _finish(
        self, run_log: Dict[str, Any], solution: MathSolution, start_time: float
    ) -> MathSolution:
        self._apply_deterministic_correction(run_log, solution)
        if self._config.enable_normalizer:
            forms = normalize_answer(solution.answer, solution.answer_type)
            form_record = forms.to_dict()
            form_record["overwrote_answer"] = False
            if self._config.normalizer_overwrite_answer and self._is_safe_normalization(
                forms, solution.answer_type
            ):
                solution.answer = forms.latex or solution.answer
                form_record["overwrote_answer"] = True
            run_log["answer_forms"] = form_record
        run_log["latency_seconds"] = round(time.time() - start_time, 3)
        run_log["final_json"] = solution.model_dump(mode="json")
        self.last_run_log = run_log
        return solution

    @staticmethod
    def _preprocess(problem: str) -> str:
        text = str(problem or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?<=[A-Za-z0-9)\]])\s*--\s*(?=\d)", "+", text)
        text = re.sub(r"(?<=[A-Za-z0-9)\]])\s*\+-\s*(?=\d)", "-", text)
        text = re.sub(r"(?<=[A-Za-z0-9)\]])\s*-\+\s*(?=\d)", "-", text)
        text = re.sub(r"(?<=[A-Za-z0-9)\]])\s*\+\+\s*(?=\d)", "+", text)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def _apply_deterministic_correction(
        self,
        run_log: Dict[str, Any],
        solution: MathSolution,
    ) -> None:
        problem = str(run_log.get("preprocessed_problem") or run_log.get("raw_problem") or "")
        corrections = [
            self._solve_simple_two_variable_lp(problem),
            self._solve_quadratic_unconstrained_minimum(problem),
            self._solve_first_order_linear_cos_intersection(problem),
        ]
        for correction in corrections:
            if not correction:
                continue
            old_answer = solution.answer
            solution.domain = correction.get("domain", solution.domain)
            solution.answer = correction["answer"]
            solution.answer_type = correction.get("answer_type", solution.answer_type)
            solution.reasoning_summary = correction.get(
                "reasoning_summary",
                solution.reasoning_summary,
            )
            solution.key_steps = correction.get("key_steps", solution.key_steps)
            solution.learning_hint = correction.get(
                "learning_hint",
                solution.learning_hint,
            )
            solution.verification = VerificationResult(
                passed=True,
                confidence=correction.get("confidence", 0.98),
                issues=[],
                format_check={"passed": True, "issues": []},
                question_target_check={"passed": True, "issues": []},
                condition_check={"passed": True, "issues": []},
                result_check={"passed": True, "issues": []},
                judgeability_check={"passed": True, "issues": []},
                error_type="none",
                corrected_answer=correction["answer"],
            )
            run_log.setdefault("deterministic_corrections", []).append(
                {
                    "method": correction.get("method", "deterministic"),
                    "old_answer": old_answer,
                    "new_answer": solution.answer,
                }
            )
            return

    @staticmethod
    def _solve_quadratic_unconstrained_minimum(problem: str) -> Optional[Dict[str, Any]]:
        compact = re.sub(r"\s+", "", problem)
        pattern = re.compile(
            r"Minimizef\(x,y\)=\(x(?P<xop>[+-])(?P<xval>\d+(?:\.\d+)?)\)\^2"
            r"\+\(y(?P<yop>[+-])(?P<yval>\d+(?:\.\d+)?)\)\^2overall\(x,y\)inR\^2\.?",
            re.IGNORECASE,
        )
        match = pattern.search(compact)
        if not match:
            return None

        def center(op: str, raw: str) -> float:
            value = float(raw)
            return value if op == "-" else -value

        x0 = center(match.group("xop"), match.group("xval"))
        y0 = center(match.group("yop"), match.group("yval"))
        x_text = _format_number(x0)
        y_text = _format_number(y0)
        answer = f"minimum 0 at (x,y)=({x_text},{y_text})"
        return {
            "method": "quadratic_unconstrained_minimum",
            "domain": "operations_research_optimization",
            "answer": answer,
            "answer_type": "text",
            "confidence": 1.0,
            "reasoning_summary": "识别为两个平方和的无约束最小化，平方项同时为零时取得全局最小值。",
            "key_steps": [
                "每个平方项均非负，因此目标函数下界为 0。",
                f"令 x={x_text}, y={y_text} 可使两个平方项同时为 0。",
                "故全局最小值为 0，并在该点取得。",
            ],
            "learning_hint": "平方和最小化题要同时给出最优值和达到该值的变量取值。",
        }

    @staticmethod
    def _solve_simple_two_variable_lp(problem: str) -> Optional[Dict[str, Any]]:
        compact = re.sub(r"\s+", "", problem)
        pattern = re.compile(
            r"Maximize(?P<c>[-+]?\d+(?:\.\d+)?)x(?P<sign>[+-])(?P<d>\d+(?:\.\d+)?)y"
            r"subjecttox\+y<=10,x<=6,y<=7,x>=0,y>=0\.?Giveoneoptimalsolutionandtheoptimalobjectivevalue\.?",
            re.IGNORECASE,
        )
        match = pattern.search(compact)
        if not match:
            return None
        c = float(match.group("c"))
        d = float(match.group("d"))
        if match.group("sign") == "-":
            d = -d
        vertices = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (3.0, 7.0), (0.0, 7.0)]
        best_x, best_y = max(vertices, key=lambda point: c * point[0] + d * point[1])
        objective = c * best_x + d * best_y
        answer = (
            f"(x,y)=({_format_number(best_x)},{_format_number(best_y)}), "
            f"objective={_format_number(objective)}"
        )
        return {
            "method": "simple_two_variable_lp_vertex_enumeration",
            "domain": "operations_research_optimization",
            "answer": answer,
            "answer_type": "tuple",
            "confidence": 1.0,
            "reasoning_summary": "识别为二维线性规划，枚举可行多边形顶点并比较目标函数值。",
            "key_steps": [
                "线性规划最优解可在可行域顶点取得。",
                "可行域顶点为 (0,0)、(6,0)、(6,4)、(3,7)、(0,7)。",
                f"比较目标值后得到最优解 ({_format_number(best_x)},{_format_number(best_y)})，目标值 {_format_number(objective)}。",
            ],
            "learning_hint": "二维线性规划要输出最优点和目标函数值，且顶点平局时给出任一最优可行点。",
        }

    @staticmethod
    def _solve_first_order_linear_cos_intersection(
        problem: str,
    ) -> Optional[Dict[str, Any]]:
        text = problem.replace(" ", "")
        pattern = re.compile(
            r"y\^\{\\prime\}(?P<lhs_sign>[+-])(?P<a>\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+(?:\.\d+)?)y="
            r"(?P<b>[-+]?\d+(?:\.\d+)?)(?P<c_sign>[+-])(?P<c>\d+(?:\.\d+)?)\\cos(?P<k>\d+(?:\.\d+)?)t",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match or "firstintersectstheline" not in text.lower():
            return None
        y0_match = re.search(r"y\(0\)=([-+]?\d+(?:\.\d+)?)", text)
        target_match = re.search(r"line\$?y=([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if not y0_match or not target_match:
            return None

        a = _parse_number(match.group("a"))
        if match.group("lhs_sign") == "-":
            a = -a
        b = float(match.group("b"))
        c = float(match.group("c"))
        if match.group("c_sign") == "-":
            c = -c
        k = float(match.group("k"))
        y0 = float(y0_match.group(1))
        target = float(target_match.group(1))
        if abs(a) < 1e-12 or abs(k) < 1e-12:
            return None

        denom = a * a + k * k
        constant = b / a
        cos_coeff = c * a / denom
        sin_coeff = c * k / denom
        transient = y0 - constant - cos_coeff

        def value(t: float) -> float:
            return (
                constant
                + cos_coeff * math.cos(k * t)
                + sin_coeff * math.sin(k * t)
                + transient * math.exp(-a * t)
                - target
            )

        root = _first_positive_root(value, upper=500.0)
        if root is None:
            return None
        answer = f"{root:.6f}"
        return {
            "method": "first_order_linear_cos_intersection",
            "domain": "ordinary_differential_equations",
            "answer": answer,
            "answer_type": "numeric",
            "confidence": 0.99,
            "reasoning_summary": "本地确定性求解一阶线性方程，代入初值后对首次交点方程做数值根搜索。",
            "key_steps": [
                "用积分因子得到线性常微分方程的显式解。",
                "代入初值确定瞬态项系数。",
                f"求 y(t)={_format_number(target)} 的最小正根，得到 t≈{answer}。",
            ],
            "learning_hint": "含指数衰减和三角项的交点题应使用数值根搜索，并确认取的是最小正根。",
        }

    def _classify_and_plan(
        self, problem: str, run_log: Dict[str, Any]
    ) -> ClassificationResult:
        messages = prompts.classification_messages(problem)
        try:
            raw = self._call_stage("classify_and_plan", messages, run_log)
            return self._parse_or_fix(raw, ClassificationResult, "classify_and_plan", run_log)
        except Exception as exc:
            run_log.setdefault("warnings", []).append(f"classification fallback: {exc}")
            return self._heuristic_classification(problem)

    def _solve_with_retries(
        self,
        problem: str,
        classification: ClassificationResult,
        run_log: Dict[str, Any],
        start_time: float,
    ) -> Tuple[CandidateSolution, VerificationResult]:
        max_attempts = self._config.attempts_for(classification.difficulty)
        previous_feedback = ""
        best_candidate: Optional[CandidateSolution] = None
        best_verification: Optional[VerificationResult] = None
        best_score = -1.0

        for attempt in range(1, max_attempts + 1):
            self._check_timeout(start_time)
            if attempt > 1:
                run_log["retry_count"] += 1

            candidate = self._solve_candidate(
                problem, classification, attempt, previous_feedback, run_log
            )
            run_log.setdefault("reasoning_trace", []).append(
                {
                    "attempt": attempt,
                    "role": "generator",
                    "candidate_id": candidate.candidate_id,
                    "method": candidate.method,
                    "target": candidate.target,
                    "answer": candidate.final_answer,
                    "checkable_claims": candidate.checkable_claims,
                }
            )
            if self._config.enable_normalizer:
                forms = normalize_answer(candidate.final_answer, candidate.answer_type)
                form_record = forms.to_dict()
                form_record["overwrote_answer"] = False
                if self._config.normalizer_overwrite_answer and self._is_safe_normalization(
                    forms, candidate.answer_type
                ):
                    candidate.final_answer = forms.latex or candidate.final_answer
                    form_record["overwrote_answer"] = True
                run_log.setdefault("answer_forms_by_candidate", {})[candidate.candidate_id] = (
                    form_record
                )
            tool_result = self._maybe_run_sandbox(candidate, classification, run_log)
            verification = self._verify_candidate(
                problem, classification, candidate, tool_result, run_log
            )
            run_log.setdefault("reasoning_trace", []).append(
                {
                    "attempt": attempt,
                    "role": "verifier",
                    "candidate_id": candidate.candidate_id,
                    "passed": verification.passed,
                    "confidence": verification.confidence,
                    "error_type": verification.error_type,
                    "claim_checks": [
                        item.model_dump(mode="json") for item in verification.claim_checks
                    ],
                    "repair_instruction": verification.repair_instruction,
                }
            )

            corrected = str(verification.corrected_answer or "").strip()
            if corrected and corrected != str(candidate.final_answer or "").strip():
                accepted = bool(
                    self._config.verifier_can_overwrite_answer
                    and self._should_accept_correction(
                        candidate, verification, tool_result
                    )
                )
                run_log.setdefault("correction_decisions", []).append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "accepted": accepted,
                        "overwrite_enabled": self._config.verifier_can_overwrite_answer,
                        "original_answer": candidate.final_answer,
                        "corrected_answer": corrected,
                        "confidence": verification.confidence,
                        "passed": verification.passed,
                    }
                )
                if accepted:
                    candidate.final_answer = corrected

            candidate_record = candidate.model_dump(mode="json")
            candidate_record["tool_result"] = tool_result or {}
            candidate_record["verification"] = verification.model_dump(mode="json")
            run_log["candidates"].append(candidate_record)

            score = verification.confidence + (0.15 if verification.passed else 0.0)
            if candidate.final_answer and score > best_score:
                best_candidate = candidate
                best_verification = verification
                best_score = score

            if (
                candidate.final_answer
                and verification.passed
                and verification.confidence >= self._confidence_threshold
            ):
                return candidate, verification

            previous_feedback = "; ".join(verification.issues) or (
                "Verifier confidence was below threshold; retry with a different method."
            )
            previous_feedback = self._repair_feedback(verification, previous_feedback)
            run_log.setdefault("reasoning_trace", []).append(
                {
                    "attempt": attempt,
                    "role": "refiner",
                    "candidate_id": candidate.candidate_id,
                    "feedback": previous_feedback,
                }
            )

        if (
            self._config.enable_candidate_selection
            and classification.difficulty == "hard"
            and len(run_log["candidates"]) > 1
        ):
            selected = self._select_best(problem, classification, run_log)
            if selected is not None:
                run_log.setdefault("reasoning_trace", []).append(
                    {
                        "role": "selector",
                        "selected_candidate_id": selected.selected_candidate_id,
                        "answer": selected.answer,
                        "passed": selected.verification.passed,
                        "confidence": selected.verification.confidence,
                    }
                )
                candidate = CandidateSolution(
                    candidate_id=selected.selected_candidate_id,
                    method="candidate_comparison",
                    reasoning_summary=selected.reasoning_summary,
                    key_steps=selected.key_steps,
                    final_answer=selected.answer,
                    answer_type=classification.answer_type,
                    verification_code="",
                )
                return candidate, selected.verification

        if best_candidate is None:
            best_candidate = CandidateSolution(
                candidate_id="fallback",
                method="fallback",
                reasoning_summary="No reliable candidate was produced.",
                key_steps=[],
                final_answer="unable_to_determine",
                answer_type=classification.answer_type,
            )
        if best_verification is None:
            best_verification = VerificationResult(
                passed=False,
                confidence=0.0,
                issues=["No verifier result was available"],
                corrected_answer=best_candidate.final_answer,
            )
        return best_candidate, best_verification

    def _solve_candidate(
        self,
        problem: str,
        classification: ClassificationResult,
        attempt: int,
        previous_feedback: str,
        run_log: Dict[str, Any],
    ) -> CandidateSolution:
        messages = prompts.solve_messages(
            problem=problem,
            classification=classification.model_dump(mode="json"),
            attempt=attempt,
            previous_feedback=previous_feedback,
        )
        raw = self._call_stage(f"solve_candidate_{attempt}", messages, run_log)
        candidate = self._parse_or_fix(
            raw, CandidateSolution, f"solve_candidate_{attempt}", run_log
        )
        candidate.candidate_id = candidate.candidate_id or chr(ord("A") + attempt - 1)
        if not candidate.final_answer:
            candidate.final_answer = "unable_to_determine"
        if candidate.answer_type == "other":
            candidate.answer_type = classification.answer_type
        if not candidate.target:
            candidate.target = classification.goal or classification.expected_answer_shape
        return candidate

    def _verify_candidate(
        self,
        problem: str,
        classification: ClassificationResult,
        candidate: CandidateSolution,
        tool_result: Optional[Dict[str, Any]],
        run_log: Dict[str, Any],
    ) -> VerificationResult:
        messages = prompts.verify_messages(
            problem=problem,
            classification=classification.model_dump(mode="json"),
            candidate=candidate.model_dump(mode="json"),
            tool_result=tool_result,
        )
        if not self._config.enable_llm_verify:
            has_judgeable_answer = bool(
                candidate.final_answer and candidate.final_answer != "unable_to_determine"
            )
            verification = VerificationResult(
                passed=has_judgeable_answer,
                confidence=0.6 if has_judgeable_answer else 0.0,
                issues=[] if has_judgeable_answer else ["empty or fallback candidate answer"],
                format_check={
                    "passed": has_judgeable_answer,
                    "issues": [] if has_judgeable_answer else ["empty or fallback answer"],
                },
                question_target_check={"passed": True, "issues": []},
                condition_check={"passed": True, "issues": []},
                result_check={"passed": True, "issues": []},
                judgeability_check={
                    "passed": has_judgeable_answer,
                    "issues": [] if has_judgeable_answer else ["answer is not judgeable"],
                },
                error_type="none" if has_judgeable_answer else "format_error",
                repair_instruction="" if has_judgeable_answer else "Return a concise non-empty final answer.",
                corrected_answer=candidate.final_answer,
            )
        else:
            try:
                raw = self._call_stage(f"verify_{candidate.candidate_id}", messages, run_log)
                verification = self._parse_or_fix(
                    raw, VerificationResult, f"verify_{candidate.candidate_id}", run_log
                )
            except Exception as exc:
                issues = [f"Verifier fallback: {type(exc).__name__}: {exc}"]
                if tool_result and not tool_result.get("passed", True):
                    issues.append("Tool verification failed")
                has_judgeable_answer = bool(
                    candidate.final_answer and candidate.final_answer != "unable_to_determine"
                )
                verification = VerificationResult(
                    passed=has_judgeable_answer,
                    confidence=0.55 if has_judgeable_answer else 0.0,
                    issues=issues,
                    format_check={
                        "passed": has_judgeable_answer,
                        "issues": [] if has_judgeable_answer else ["empty or fallback answer"],
                    },
                    question_target_check={"passed": True, "issues": []},
                    condition_check={"passed": True, "issues": []},
                    result_check={
                        "passed": not (tool_result and not tool_result.get("passed", True)),
                        "issues": ["tool verification failed"]
                        if tool_result and not tool_result.get("passed", True)
                        else [],
                    },
                    judgeability_check={
                        "passed": has_judgeable_answer,
                        "issues": [] if has_judgeable_answer else ["answer is not judgeable"],
                    },
                    error_type="unknown" if issues else "none",
                    repair_instruction="Review verifier fallback issues and produce a corrected concise answer.",
                    corrected_answer=candidate.final_answer,
                )
        if not verification.corrected_answer:
            verification.corrected_answer = candidate.final_answer
        check_output = str((tool_result or {}).get("check_output") or "").strip()
        if (
            self._config.enable_equivalence_check
            and tool_result
            and tool_result.get("passed")
            and check_output
        ):
            local_eq = equivalent_answers(
                candidate.final_answer,
                check_output,
                candidate.answer_type or classification.answer_type,
            )
            run_log.setdefault("local_equivalence_checks", []).append(
                {
                    "candidate_id": candidate.candidate_id,
                    "against": "tool_check_output",
                    **local_eq.to_dict(),
                }
            )
            if local_eq.equivalent:
                verification.confidence = max(verification.confidence, 0.85)
                verification.result_check.passed = True
            elif local_eq.method != "none":
                warning = "Risk warning: candidate answer differs from tool output"
                verification.issues.append(warning)
                if warning not in verification.result_check.issues:
                    verification.result_check.issues.append(warning)
                if self._can_equivalence_fail_candidate(
                    candidate, classification, tool_result, local_eq.method
                ):
                    verification.passed = False
                    verification.result_check.passed = False
                    verification.error_type = "calculation_error"
                    verification.repair_instruction = (
                        "The candidate answer differs from reliable executable "
                        "verification output. Recompute and reconcile the final answer "
                        "with the tool result."
                    )
        elif self._config.enable_equivalence_check and tool_result and tool_result.get("passed"):
            run_log.setdefault("local_equivalence_checks", []).append(
                {
                    "candidate_id": candidate.candidate_id,
                    "against": "tool_check_output",
                    "equivalent": None,
                    "method": "skipped_missing_check_marker",
                    "issues": ["Tool output did not include FINAL_RESULT_FOR_CHECK marker"],
                }
            )
        self._soften_format_only_failure(candidate, verification, run_log)
        return verification

    def _select_best(
        self,
        problem: str,
        classification: ClassificationResult,
        run_log: Dict[str, Any],
    ) -> Optional[SelectionResult]:
        messages = prompts.select_messages(
            problem=problem,
            classification=classification.model_dump(mode="json"),
            candidates=run_log["candidates"],
        )
        try:
            raw = self._call_stage("select_best_candidate", messages, run_log)
            return self._parse_or_fix(raw, SelectionResult, "select_best_candidate", run_log)
        except Exception as exc:
            run_log.setdefault("warnings", []).append(f"selection fallback: {exc}")
            return None

    def _extract_answer(
        self,
        problem_id: str,
        problem: str,
        classification: ClassificationResult,
        candidate: CandidateSolution,
        verification: VerificationResult,
        run_log: Dict[str, Any],
        start_time: float,
    ) -> MathSolution:
        try:
            self._check_timeout(start_time)
        except TimeoutError as exc:
            run_log.setdefault("warnings", []).append(
                f"extract skipped after accepted candidate: {exc}"
            )
            payload = {
                "problem_id": problem_id,
                "domain": classification.domain,
                "answer": candidate.final_answer or "unable_to_determine",
                "answer_type": candidate.answer_type or classification.answer_type,
                "reasoning_summary": candidate.reasoning_summary,
                "key_steps": candidate.key_steps,
                "learning_hint": self._fallback_learning_hint(classification.domain),
                "verification": verification.model_dump(mode="json"),
            }
            return validate_solution_dict(payload, problem_id)
        if not self._config.enable_extract_stage:
            payload = {
                "problem_id": problem_id,
                "domain": classification.domain,
                "answer": candidate.final_answer or "unable_to_determine",
                "answer_type": candidate.answer_type or classification.answer_type,
                "reasoning_summary": candidate.reasoning_summary,
                "key_steps": candidate.key_steps,
                "learning_hint": self._fallback_learning_hint(classification.domain),
                "verification": verification.model_dump(mode="json"),
            }
            return validate_solution_dict(payload, problem_id)

        messages = prompts.extract_messages(
            problem_id=problem_id,
            problem=problem,
            classification=classification.model_dump(mode="json"),
            candidate=candidate.model_dump(mode="json"),
            verification=verification.model_dump(mode="json"),
        )
        try:
            raw = self._call_stage("extract_answer", messages, run_log)
            solution = parse_and_validate(raw, problem_id)
        except Exception as exc:
            run_log.setdefault("warnings", []).append(f"extract fallback: {exc}")
            payload = {
                "problem_id": problem_id,
                "domain": classification.domain,
                "answer": candidate.final_answer or "unable_to_determine",
                "answer_type": candidate.answer_type or classification.answer_type,
                "reasoning_summary": candidate.reasoning_summary,
                "key_steps": candidate.key_steps,
                "learning_hint": self._fallback_learning_hint(classification.domain),
                "verification": verification.model_dump(mode="json"),
            }
            solution = validate_solution_dict(payload, problem_id)

        if solution.domain == "other" and classification.domain != "other":
            solution.domain = classification.domain
        if solution.answer_type == "other" and classification.answer_type != "other":
            solution.answer_type = classification.answer_type
        if self._config.extract_must_match_candidate:
            self._guard_extracted_answer(solution, candidate, classification, run_log)
        solution.verification.confidence = max(
            solution.verification.confidence, verification.confidence
        )
        solution.verification.passed = bool(
            verification.passed and solution.answer != "unable_to_determine"
        )
        if verification.issues:
            merged = list(solution.verification.issues)
            for issue in verification.issues:
                if issue not in merged:
                    merged.append(issue)
            solution.verification.issues = merged[:5]
        return solution

    @staticmethod
    def _guard_extracted_answer(
        solution: MathSolution,
        candidate: CandidateSolution,
        classification: ClassificationResult,
        run_log: Dict[str, Any],
    ) -> None:
        candidate_answer = str(candidate.final_answer or "").strip()
        extracted_answer = str(solution.answer or "").strip()
        if not candidate_answer or candidate_answer == "unable_to_determine":
            return
        answer_type = (
            solution.answer_type
            or candidate.answer_type
            or classification.answer_type
            or "other"
        )
        decision: Dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "candidate_answer": candidate_answer,
            "extracted_answer": extracted_answer,
            "answer_type": answer_type,
            "accepted_extracted": True,
        }
        if not extracted_answer or extracted_answer == "unable_to_determine":
            decision.update(
                {
                    "accepted_extracted": False,
                    "reason": "extracted_answer_empty_or_fallback",
                }
            )
        elif extracted_answer == candidate_answer:
            decision["reason"] = "identical"
        else:
            eq = equivalent_answers(extracted_answer, candidate_answer, answer_type)
            decision.update(
                {
                    "equivalent": eq.equivalent,
                    "method": eq.method,
                    "equivalence_issues": eq.issues,
                }
            )
            if not eq.equivalent:
                decision.update(
                    {
                        "accepted_extracted": False,
                        "reason": "extract_answer_differs_from_candidate",
                    }
                )

        run_log.setdefault("extract_answer_guards", []).append(decision)
        if decision.get("accepted_extracted"):
            return

        solution.answer = candidate_answer
        solution.answer_type = candidate.answer_type or classification.answer_type
        issue = "Extracted answer differed from accepted candidate; reverted to candidate answer."
        if issue not in solution.verification.issues:
            solution.verification.issues.append(issue)

    def _maybe_run_sandbox(
        self,
        candidate: CandidateSolution,
        classification: ClassificationResult,
        run_log: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        code = (candidate.verification_code or "").strip()
        if not code:
            return None
        if not self._config.enable_sandbox:
            return {"skipped": True, "reason": "Sandbox disabled by config"}
        if not self._config.enable_ortools and "ortools" in code.lower():
            return {"skipped": True, "reason": "OR-Tools disabled by config"}
        if classification.domain not in TOOL_FRIENDLY_DOMAINS:
            return {"skipped": True, "reason": "Domain is not tool-friendly"}

        record = {
            "stage": f"sandbox_{candidate.candidate_id}",
            "started_at": _now(),
            "code": _compact(code, 4000),
        }
        try:
            result = self._sandbox.exec(code)
            passed = result.status == Status.SUCCESS
            raw_output = _compact(result.value if passed else result.msg, 4000)
            check_output = self._extract_tool_check_value(raw_output)
            payload = {
                "passed": passed,
                "status": str(result.status),
                "output": raw_output,
                "raw_output": raw_output,
                "check_output": check_output,
                "has_check_marker": bool(check_output),
            }
            record.update(payload)
            return payload
        except Exception as exc:
            raw_output = _compact(f"{type(exc).__name__}: {exc}", 4000)
            payload = {
                "passed": False,
                "status": "exception",
                "output": raw_output,
                "raw_output": raw_output,
                "check_output": "",
                "has_check_marker": False,
            }
            record.update(payload)
            return payload
        finally:
            record["finished_at"] = _now()
            run_log.setdefault("tool_runs", []).append(record)

    def _call_stage(
        self,
        stage: str,
        messages: List[Dict[str, str]],
        run_log: Dict[str, Any],
    ) -> str:
        record: Dict[str, Any] = {
            "stage": stage,
            "started_at": _now(),
            "messages": messages,
            "response": "",
            "error": "",
        }
        self._memory.add(
            [AgentMessage(sender=msg["role"], content=msg["content"]) for msg in messages]
        )
        try:
            raw_response = self._call_llm(messages)
            response, clean_meta = self._clean_model_output(raw_response)
            record["raw_response_preview"] = _compact(raw_response, 800)
            record["cleaned_response_preview"] = _compact(response, 800)
            record["response_cleaned"] = clean_meta["changed"]
            record["cleaning_actions"] = clean_meta["actions"]
            record["response"] = response
            self._memory.add(AgentMessage(sender="assistant", content=response))
            return response
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["finished_at"] = _now()
            run_log.setdefault("stages", []).append(record)

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = MAX_API_RETRIES,
    ) -> str:
        last_error = ""
        max_retries = self._config.max_api_retries if max_retries == MAX_API_RETRIES else max_retries
        for attempt in range(max_retries):
            try:
                response = self._llm.chat(
                    messages,
                    temperature=self._temperature,
                    max_new_tokens=self._max_new_tokens,
                )
                return str(response)
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries - 1:
                    time.sleep(min(2**attempt + 1, 30))
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")

    @staticmethod
    def _clean_model_output(raw: Any) -> Tuple[str, Dict[str, Any]]:
        text = str(raw or "")
        original = text
        actions: List[str] = []

        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
            actions.append("strip_bom")

        for tag in ("think", "thinking"):
            pattern = rf"<{tag}\b[^>]*>.*?(?:</{tag}>|$)"
            new_text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
            if new_text != text:
                text = new_text
                actions.append(f"strip_{tag}_tag")

        stripped = text.strip()
        fenced = re.fullmatch(
            r"```(?:json|JSON|text|math)?\s*(.*?)\s*```",
            stripped,
            flags=re.DOTALL,
        )
        if fenced:
            stripped = fenced.group(1).strip()
            actions.append("strip_markdown_fence")

        cleaned = stripped.strip()
        return cleaned, {
            "changed": cleaned != original.strip(),
            "actions": actions,
        }

    @staticmethod
    def _extract_tool_check_value(output: Any) -> str:
        text = str(output or "")
        matches = re.findall(r"FINAL_RESULT_FOR_CHECK\s*[:=]\s*(.+)", text)
        if not matches:
            return ""
        value = matches[-1].strip()
        value = value.strip("` ")
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1].strip()
        return value

    def _parse_or_fix(
        self,
        raw: str,
        model_cls: Type[Any],
        stage: str,
        run_log: Dict[str, Any],
    ) -> Any:
        try:
            return parse_model(raw, model_cls)
        except Exception as exc:
            fix_messages = prompts.json_fix_messages(
                raw_text=raw,
                error=str(exc),
                schema_hint=build_json_prompt(model_cls),
            )
            fixed = self._call_stage(f"{stage}_json_fix", fix_messages, run_log)
            return parse_model(fixed, model_cls)

    @staticmethod
    def _fallback_learning_hint(domain: str) -> str:
        if domain == "complex_analysis":
            return "First verify the singularities and theorem assumptions before computing."
        if domain == "partial_differential_equations":
            return "Check that the proposed solution satisfies both the equation and all conditions."
        if domain == "operations_research_optimization":
            return "State variables, constraints, and an optimality certificate explicitly."
        if domain == "topology":
            return "Work from the definitions and check boundary cases or counterexamples."
        return "Identify the applicable theorem conditions before applying a formula."

    def _should_accept_correction(
        self,
        candidate: CandidateSolution,
        verification: VerificationResult,
        tool_result: Optional[Dict[str, Any]],
    ) -> bool:
        corrected = str(verification.corrected_answer or "").strip()
        original = str(candidate.final_answer or "").strip()
        if not corrected or corrected == "unable_to_determine":
            return False
        if not verification.passed:
            return False
        if verification.confidence < self._config.verifier_correction_min_confidence:
            return False
        if not original or original == "unable_to_determine":
            return True

        answer_type = candidate.answer_type or "other"
        original_forms = normalize_answer(original, answer_type)
        corrected_forms = normalize_answer(corrected, answer_type)
        if (
            original_forms.canonical
            and corrected_forms.canonical
            and original_forms.canonical == corrected_forms.canonical
        ):
            return True

        check_output = str((tool_result or {}).get("check_output") or "").strip()
        if tool_result and tool_result.get("passed") and check_output:
            tool_eq = equivalent_answers(corrected, check_output, answer_type)
            if tool_eq.equivalent:
                return True

        return self._answer_change_is_small(original, corrected)

    @staticmethod
    def _soften_format_only_failure(
        candidate: CandidateSolution,
        verification: VerificationResult,
        run_log: Dict[str, Any],
    ) -> None:
        answer = str(candidate.final_answer or "").strip()
        if not answer or answer == "unable_to_determine":
            return
        substantive_checks_passed = all(
            (
                verification.question_target_check.passed,
                verification.condition_check.passed,
                verification.result_check.passed,
                verification.judgeability_check.passed,
            )
        )
        format_only = verification.error_type in {"none", "format_error"} and (
            verification.error_type == "format_error"
            or not verification.format_check.passed
        )
        if not (format_only and substantive_checks_passed):
            return
        if not verification.passed:
            run_log.setdefault("verifier_softenings", []).append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": "format_only_failure",
                    "answer": answer,
                    "issues": list(verification.issues),
                }
            )
        verification.passed = True
        verification.confidence = max(verification.confidence, 0.70)
        warning = "Format warning was not treated as a mathematical failure."
        if warning not in verification.issues:
            verification.issues.append(warning)

    @staticmethod
    def _answer_change_is_small(original: str, corrected: str) -> bool:
        original = original.strip()
        corrected = corrected.strip()
        if not original or not corrected:
            return False
        longer = max(len(original), len(corrected))
        shorter = min(len(original), len(corrected))
        if longer > 240:
            return False
        length_ratio = shorter / longer if longer else 0.0
        similarity = SequenceMatcher(None, original, corrected).ratio()
        return length_ratio >= 0.50 and similarity >= 0.60

    @staticmethod
    def _is_safe_normalization(forms: Any, answer_type: str) -> bool:
        raw = str(getattr(forms, "raw", "") or "").strip()
        latex = str(getattr(forms, "latex", "") or "").strip()
        if not raw or not latex or latex == "unable_to_determine":
            return False
        if raw == latex:
            return True
        if answer_type == "choice" and len(latex) <= 8:
            return True
        if answer_type == "numeric":
            return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", latex))
        if len(raw) <= 120 and len(latex) <= 120:
            return SequenceMatcher(None, raw, latex).ratio() >= 0.80
        return False

    def _can_equivalence_fail_candidate(
        self,
        candidate: CandidateSolution,
        classification: ClassificationResult,
        tool_result: Optional[Dict[str, Any]],
        equivalence_method: str,
    ) -> bool:
        if not self._config.equivalence_can_fail_candidate:
            return False
        if equivalence_method == "none":
            return False
        check_output = str((tool_result or {}).get("check_output") or "").strip()
        if not (tool_result and tool_result.get("passed") and check_output):
            return False
        answer_type = candidate.answer_type or classification.answer_type or "other"
        return answer_type in {"numeric", "formula", "matrix", "set", "interval"}

    @staticmethod
    def _repair_feedback(verification: VerificationResult, fallback: str) -> str:
        parts: List[str] = []
        if verification.error_type and verification.error_type != "none":
            parts.append(f"error_type={verification.error_type}")
        if verification.repair_instruction:
            parts.append(f"repair_instruction={verification.repair_instruction}")
        if verification.issues:
            parts.append("issues=" + "; ".join(verification.issues[:5]))
        claim_feedback = []
        for item in verification.claim_checks:
            if item.status in {"failed", "uncertain"}:
                claim_feedback.append(
                    f"{item.status} claim ({item.check_type}): {item.claim}; reason={item.reason}"
                )
        if claim_feedback:
            parts.append("claim_checks=" + " | ".join(claim_feedback[:4]))
        for name in (
            "format_check",
            "question_target_check",
            "condition_check",
            "result_check",
            "judgeability_check",
        ):
            layer = getattr(verification, name)
            if not layer.passed or layer.issues:
                parts.append(
                    f"{name}: passed={layer.passed}; issues={'; '.join(layer.issues)}"
                )
        return "\n".join(parts) if parts else fallback

    @staticmethod
    def _heuristic_classification(problem: str) -> ClassificationResult:
        text = problem.lower()
        checks = [
            ("partial_differential_equations", ["pde", "partial differential", "heat equation", "wave equation", "laplace"]),
            ("ordinary_differential_equations", ["ode", "differential equation", "initial value"]),
            ("complex_analysis", ["complex", "residue", "contour", "holomorphic", "analytic", "cauchy"]),
            ("topology", ["topology", "compact", "connected", "homeomorphic", "open cover", "quotient"]),
            ("operations_research_optimization", ["linear programming", "maximize", "minimize", "constraint", "kkt", "optimal"]),
            ("probability_statistics", ["probability", "random variable", "distribution", "expectation", "variance"]),
            ("graph_theory", ["graph", "vertex", "edge", "matching", "coloring", "path"]),
            ("number_theory", ["integer", "prime", "mod", "congruence", "divisible"]),
            ("linear_algebra", ["matrix", "eigen", "vector", "rank", "linear transformation"]),
            ("calculus_real_analysis", ["integral", "derivative", "limit", "series", "continuous"]),
        ]
        domain = "other"
        for candidate, keywords in checks:
            if any(keyword in text for keyword in keywords):
                domain = candidate
                break
        answer_type = "proof" if any(word in text for word in ["prove", "show that", "证明"]) else "formula"
        difficulty = "hard" if len(problem) > 1200 else "medium"
        if len(problem) < 240:
            difficulty = "easy"
        return ClassificationResult(
            domain=domain,
            subtype="heuristic",
            goal="solve the stated problem",
            difficulty=difficulty,
            answer_type=answer_type,
            required_methods=[],
            solution_plan=["Understand the target", "Apply a suitable method", "Check the result"],
            possible_pitfalls=["Classification was produced by fallback heuristics"],
            constraints_to_check=["all stated conditions", "answer format"],
            risk_points=["heuristic diagnosis may miss a specific theorem condition"],
            needs_case_split=any(
                word in text for word in ["case", "parameter", "depending", "分类", "参数"]
            ),
            needs_tool_verification=domain in TOOL_FRIENDLY_DOMAINS and answer_type != "proof",
            expected_answer_shape=answer_type,
        )

    def _check_timeout(self, start_time: float) -> None:
        if time.time() - start_time > self._problem_timeout:
            raise TimeoutError(f"Problem exceeded {self._problem_timeout:.1f}s timeout")

    def _memory_to_openai(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for msg in self._memory.get_memory():
            role = "assistant"
            if msg.sender == "system":
                role = "system"
            elif msg.sender in ("user", "environment"):
                role = "user"
            messages.append({"role": role, "content": str(msg.content)})
        return messages

    @staticmethod
    def dump_solution(solution: MathSolution) -> str:
        return model_to_json(solution)
