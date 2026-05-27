"""Evaluate bare LLM result files with the MathSolve-Agent validation flow.

This module is for experiments where solving is done by a direct Chat Completions
call, but evaluation should still use the agent project's schema, local
equivalence checks, and optional LLM judge.

Example:
  python -m math_prove.evaluate_direct_outputs \
    --dataset path/to/dataset.json \
    --run path/to/direct_results.jsonl \
    --out-dir outputs/direct_eval \
    --prefix harder_balanced18 \
    --llm-judge \
    --judge-api-key %INTERNLM_API_TOKEN% \
    --judge-api-base https://chat.intern-ai.org.cn/api/v1/chat/completions \
    --judge-model intern-s1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from .validator import LLMJudgeConfig, validate_results, write_validation_report


DEFAULT_ENDPOINT = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
TOKEN_ENV_NAMES = (
    "INTERNLM_API_TOKEN",
    "INTERNLM_API_KEY",
    "INTERN_API_TOKEN",
    "INTERN_API_KEY",
    "DEEPSEEK_API_KEY",
)


DOMAIN_MAP = {
    "calculus_analysis": "calculus_real_analysis",
    "real_analysis_measure": "calculus_real_analysis",
    "linear_algebra_matrix": "linear_algebra",
    "abstract_algebra": "linear_algebra",
    "ode": "ordinary_differential_equations",
    "pde": "partial_differential_equations",
    "complex_analysis": "complex_analysis",
    "topology": "topology",
    "functional_analysis": "functional_analysis",
    "probability": "probability_statistics",
    "mathematical_statistics": "probability_statistics",
    "combinatorics": "combinatorics",
    "graph_discrete_math": "graph_theory",
    "number_theory": "number_theory",
    "geometry_analytic_geometry": "geometry",
    "operations_research": "operations_research_optimization",
    "convex_nonlinear_optimization": "operations_research_optimization",
    "numerical_analysis": "numerical_analysis",
}


def load_json_any(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list: {path}")
        return data
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token.strip()
    for name in TOKEN_ENV_NAMES:
        token = os.environ.get(name)
        if token:
            return token.strip()
    return ""


def math_domain(item: dict[str, Any]) -> str:
    raw = str(item.get("math_domain") or item.get("domain") or "")
    return DOMAIN_MAP.get(raw, raw or "other")


def simple_eval_number(value: Any) -> float | None:
    text = str(value or "").strip().replace("$", "")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
    text = text.replace("\\sqrt", "sqrt")
    text = text.replace("\\pi", "pi")
    text = text.replace("^", "**")
    alpha_probe = re.sub(r"\b(sqrt|pi|e|E)\b", "", text)
    if re.search(r"[A-Za-z\\]", alpha_probe):
        return None
    text = re.sub(r"[^0-9+\-*/(). sqrtpieE]", "", text)
    if not text.strip():
        return None
    try:
        return float(
            eval(  # noqa: S307 - expression is reduced to numeric tokens above.
                text,
                {"__builtins__": {}},
                {"sqrt": math.sqrt, "pi": math.pi, "e": math.e},
            )
        )
    except Exception:
        return None


def looks_numeric(value: Any) -> bool:
    raw = str(value or "").strip()
    return bool(raw) and ("%" in raw or simple_eval_number(raw) is not None)


def answer_type_from_item(item: dict[str, Any]) -> str:
    form = str(item.get("answer_form") or "").lower()
    answer = item.get("answer", item.get("gold_answer", ""))
    if form == "proof_text":
        return "proof"
    if form in {"integer", "real_number"}:
        return "numeric"
    if form == "rational":
        return "numeric" if looks_numeric(answer) else "formula"
    if form in {"choice", "multiple_choice"}:
        return "choice"
    if form == "matrix":
        return "matrix"
    if form == "set":
        return "set"
    if form in {"tuple", "vector"}:
        return form
    if form == "symbolic_expression":
        compact_answer = str(answer).strip().lower()
        if compact_answer in {"true", "false", "(a)", "(b)", "(c)", "(d)", "a", "b", "c", "d"}:
            return "choice"
        return "numeric" if looks_numeric(answer) else "formula"
    return "other"


def extract_braced(text: str, start: int) -> str:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return ""


def extract_boxed(text: str) -> str:
    hits = []
    for command in ("\\boxed", "\\fbox"):
        pos = 0
        while True:
            idx = text.find(command, pos)
            if idx < 0:
                break
            brace = text.find("{", idx)
            if brace >= 0:
                value = extract_braced(text, brace)
                if value:
                    hits.append(value.strip())
            pos = idx + len(command)
    return hits[-1] if hits else ""


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def extract_answer(text: str) -> str:
    parsed = parse_json_object(text)
    if parsed.get("answer"):
        return str(parsed["answer"]).strip()

    # Handle truncated JSON after an answer field has already appeared.
    match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)', str(text), flags=re.S)
    if match:
        try:
            return json.loads(f'"{match.group(1)}"').strip()
        except json.JSONDecodeError:
            return match.group(1).strip()

    boxed = extract_boxed(str(text))
    return boxed.strip()


def parse_raw_solution(raw_text: str) -> tuple[str, str]:
    parsed = parse_json_object(raw_text)
    answer = str(parsed.get("answer") or "").strip()
    solution = str(parsed.get("solution") or "").strip()
    if not answer:
        answer = extract_answer(raw_text)
    return answer, solution


def compact(value: Any, limit: int = 800) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_expected(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in dataset:
        pid = str(item.get("id") or item.get("problem_id") or "").strip()
        if not pid:
            continue
        rows.append(
            {
                "problem_id": pid,
                "problem_text": item.get("problem") or item.get("problem_text") or item.get("question") or "",
                "domain": math_domain(item),
                "answer_type": answer_type_from_item(item),
                "expected_answer": item.get("answer", item.get("expected_answer", "")),
                "source_math_domain": item.get("math_domain", ""),
                "source_answer_form": item.get("answer_form", ""),
            }
        )
    return rows


def build_results(dataset: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = {str(item.get("id") or item.get("problem_id")): item for item in dataset}
    results = []
    for run in run_rows:
        pid = str(run.get("id") or run.get("problem_id") or "").strip()
        item = items.get(pid, run)
        raw_text = str(run.get("raw_text") or "")
        answer, solution = parse_raw_solution(raw_text)
        if not answer:
            answer = str(run.get("model_answer") or "").strip()
        if not solution:
            solution = raw_text

        api_ok = run.get("status", "ok") == "ok"
        answer_ok = bool(answer.strip())
        passed = bool(api_ok and answer_ok)
        issues = []
        if not api_ok:
            issues.append(f"direct API run status={run.get('status')}")
        if not answer_ok:
            issues.append("empty extracted answer")

        results.append(
            {
                "problem_id": pid,
                "domain": math_domain(item),
                "answer": answer or "unable_to_determine",
                "answer_type": answer_type_from_item(item),
                "reasoning_summary": compact(solution, 800),
                "key_steps": [compact(solution, 260)] if solution else [],
                "learning_hint": "",
                "verification": {
                    "passed": passed,
                    "confidence": 0.6 if passed else 0.0,
                    "issues": issues,
                    "format_check": {"passed": answer_ok, "issues": [] if answer_ok else ["empty answer"]},
                    "question_target_check": {"passed": passed, "issues": [] if passed else issues},
                    "condition_check": {"passed": True, "issues": []},
                    "result_check": {"passed": passed, "issues": [] if passed else issues},
                    "judgeability_check": {
                        "passed": answer_ok,
                        "issues": [] if answer_ok else ["answer is not judgeable"],
                    },
                    "claim_checks": [],
                    "error_type": "none" if passed else "format_error",
                    "repair_instruction": "",
                },
            }
        )
    return results


def parse_numeric_percent(value: Any) -> tuple[float | None, bool]:
    raw = str(value or "").strip()
    return simple_eval_number(raw.replace("%", "")), "%" in raw


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=5e-3, abs_tol=1e-2)


def numeric_guard(expected: Any, predicted: Any, answer_type: str) -> dict[str, Any] | None:
    if answer_type != "numeric":
        return None
    expected_num, expected_percent = parse_numeric_percent(expected)
    predicted_num, predicted_percent = parse_numeric_percent(predicted)
    if expected_num is None or predicted_num is None:
        return None

    candidates = [(predicted_num, "direct numeric comparison")]
    if expected_percent and not predicted_percent:
        candidates.append((predicted_num * 100.0, "prediction decimal converted to percent"))
    if predicted_percent and not expected_percent:
        candidates.append((predicted_num / 100.0, "prediction percent converted to decimal"))

    for value, reason in candidates:
        if close_enough(value, expected_num):
            return {
                "correct": True,
                "confidence": 0.98,
                "reason": f"Numeric guard accepted: {reason}. expected={expected_num:g}, predicted={predicted_num:g}",
                "method": "agent_numeric_guard",
            }
    return {
        "correct": False,
        "confidence": 0.95,
        "reason": f"Numeric guard rejected: expected={expected_num:g}, predicted={predicted_num:g}",
        "method": "agent_numeric_guard",
    }


def parse_judge_json(text: str) -> tuple[bool | None, float, str]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    candidate = match.group(0) if match else cleaned
    try:
        parsed = json.loads(candidate)
        if "correct" in parsed:
            correct = bool(parsed.get("correct"))
        elif "verdict" in parsed:
            correct = str(parsed.get("verdict", "")).lower() == "correct"
        else:
            correct = None
        confidence = float(parsed.get("confidence", 0.0))
        return correct, max(0.0, min(1.0, confidence)), str(parsed.get("reason", ""))[:600]
    except Exception:
        lowered = cleaned.lower()
        if "incorrect" in lowered or "not correct" in lowered or "wrong" in lowered:
            return False, 0.45, cleaned[:600]
        if "correct" in lowered or "equivalent" in lowered:
            return True, 0.45, cleaned[:600]
        return None, 0.0, cleaned[:600]


def robust_llm_judge_one(
    *,
    token: str,
    endpoint: str,
    model: str,
    timeout: int,
    item: dict[str, Any],
    run: dict[str, Any],
    result_row: dict[str, Any],
    expected_row: dict[str, Any],
    retries: int,
) -> dict[str, Any]:
    prompt = (
        "You are a strict but format-tolerant mathematical answer judge.\n"
        "Decide whether the model's visible output is mathematically correct for the problem.\n"
        "Treat equivalent forms as correct. For proof problems, judge whether the visible output "
        "contains a valid proof, not merely whether the final conclusion text matches.\n"
        "Return only JSON with keys: correct, confidence, reason.\n\n"
        + json.dumps(
            {
                "problem": item.get("problem") or item.get("problem_text") or item.get("question") or "",
                "answer_type": expected_row.get("answer_type"),
                "reference_answer": expected_row.get("expected_answer"),
                "extracted_answer": result_row.get("answer"),
                "visible_model_output": compact(run.get("raw_text") or result_row.get("reasoning_summary") or "", 3500),
            },
            ensure_ascii=False,
        )
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict mathematical grader. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 768,
        "stream": False,
        "thinking_mode": False,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            start = time.perf_counter()
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=timeout,
            )
            data = response.json()
            content = str(data["choices"][0]["message"].get("content") or "")
            correct, confidence, reason = parse_judge_json(content)
            return {
                "correct": correct,
                "confidence": confidence,
                "reason": reason,
                "method": f"robust_llm_judge:{model}",
                "latency_sec": round(time.perf_counter() - start, 3),
                "raw_judge_text": compact(content, 1200),
            }
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(5.0)
    return {
        "correct": None,
        "confidence": 0.0,
        "reason": f"judge_api_failed: {last_error!r}",
        "method": "robust_llm_judge_error",
    }


def write_robust_judge_report(
    *,
    dataset: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    strict_payload: dict[str, Any],
    output_path: Path,
    token: str,
    endpoint: str,
    model: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    items_by_id = {str(item.get("id") or item.get("problem_id")): item for item in dataset}
    runs_by_id = {str(row.get("id") or row.get("problem_id")): row for row in run_rows}
    results_by_id = {str(row.get("problem_id")): row for row in result_rows}
    expected_by_id = {str(row.get("problem_id")): row for row in expected_rows}
    strict_items = {str(row.get("problem_id")): row for row in strict_payload.get("items", [])}

    judged_rows = []
    correct = incorrect = unknown = llm_calls = 0
    for expected in expected_rows:
        pid = str(expected.get("problem_id"))
        strict_item = strict_items.get(pid, {})
        local_correct = strict_item.get("answer_equivalent") is True
        judge = {
            "correct": True,
            "confidence": 1.0,
            "reason": "Accepted by local equivalence check.",
            "method": "local_equivalence_shortcut",
        }
        if not local_correct:
            judge = numeric_guard(
                expected.get("expected_answer"),
                results_by_id.get(pid, {}).get("answer"),
                str(expected.get("answer_type") or ""),
            )
            if judge is None:
                llm_calls += 1
                judge = robust_llm_judge_one(
                    token=token,
                    endpoint=endpoint,
                    model=model,
                    timeout=timeout,
                    retries=retries,
                    item=items_by_id.get(pid, {}),
                    run=runs_by_id.get(pid, {}),
                    result_row=results_by_id.get(pid, {}),
                    expected_row=expected_by_id.get(pid, {}),
                )

        final_correct = judge.get("correct")
        if final_correct is True:
            correct += 1
        elif final_correct is False:
            incorrect += 1
        else:
            unknown += 1

        judged_rows.append(
            {
                "problem_id": pid,
                "math_domain": expected.get("source_math_domain"),
                "answer_type": expected.get("answer_type"),
                "expected_answer": expected.get("expected_answer"),
                "predicted_answer": results_by_id.get(pid, {}).get("answer"),
                "strict_local_correct": local_correct,
                "strict_method": strict_item.get("equivalence_method", ""),
                "judge_correct": final_correct,
                "judge_confidence": judge.get("confidence"),
                "judge_method": judge.get("method"),
                "judge_reason": judge.get("reason", ""),
            }
        )

    report = {
        "total": len(expected_rows),
        "strict_local_correct": sum(
            1 for row in strict_payload.get("items", []) if row.get("answer_equivalent") is True
        ),
        "llm_calls": llm_calls,
        "judge_correct": correct,
        "judge_incorrect": incorrect,
        "judge_unknown": unknown,
        "judge_accuracy_counting_unknown_wrong": correct / len(expected_rows) if expected_rows else None,
        "judge_accuracy_known_only": correct / (correct + incorrect) if correct + incorrect else None,
        "items": judged_rows,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "problem_id",
            "math_domain",
            "answer_type",
            "expected_answer",
            "predicted_answer",
            "strict_local_correct",
            "strict_method",
            "judge_correct",
            "judge_confidence",
            "judge_method",
            "judge_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(judged_rows)
    return report


def summarize_agent_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "total": report.get("total"),
        "schema_valid": report.get("schema_valid"),
        "schema_valid_rate": report.get("schema_valid_rate"),
        "preflight_issue_count": report.get("preflight_issue_count"),
        "answer_checked": report.get("answer_checked"),
        "answer_correct": report.get("answer_correct"),
        "answer_accuracy": report.get("answer_accuracy"),
        "llm_judge_checked": report.get("llm_judge_checked"),
        "llm_judge_correct": report.get("llm_judge_correct"),
        "llm_judge_accuracy": report.get("llm_judge_accuracy"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Original dataset JSON/JSONL with gold answers.")
    parser.add_argument("--run", type=Path, required=True, help="Bare LLM result JSONL/JSON file.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="direct")
    parser.add_argument("--llm-judge", action="store_true", help="Also run LLM judge reports.")
    parser.add_argument("--llm-judge-all", action="store_true")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-api-base", default=DEFAULT_ENDPOINT)
    parser.add_argument("--judge-model", default="intern-s1")
    parser.add_argument("--judge-timeout", type=int, default=120)
    parser.add_argument("--judge-retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_json_any(args.dataset)
    runs = load_json_any(args.run)
    expected = build_expected(dataset)
    results = build_results(dataset, runs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_path = args.out_dir / f"{args.prefix}.agent_expected.jsonl"
    results_path = args.out_dir / f"{args.prefix}.direct_as_agent_results.jsonl"
    strict_report_path = args.out_dir / f"{args.prefix}.agent_strict_report.json"
    llm_report_path = args.out_dir / f"{args.prefix}.agent_llm_judge_report.json"
    robust_report_path = args.out_dir / f"{args.prefix}.agent_style_robust_judge_report.json"
    summary_path = args.out_dir / f"{args.prefix}.agent_eval_summary.json"

    write_jsonl(expected_path, expected)
    write_jsonl(results_path, results)

    strict_report = validate_results(str(results_path), str(expected_path), strict_expected_ids=True)
    write_validation_report(strict_report, str(strict_report_path))
    strict_payload = strict_report.to_dict()

    llm_payload = None
    robust_payload = None
    if args.llm_judge:
        token = get_token(args.judge_api_key)
        if not token:
            raise RuntimeError(
                "Missing judge API token. Pass --judge-api-key or set one of: "
                + ", ".join(TOKEN_ENV_NAMES)
            )
        judge_config = LLMJudgeConfig(
            enabled=True,
            api_key=token,
            api_base=args.judge_api_base,
            model=args.judge_model,
            timeout=args.judge_timeout,
            judge_all=args.llm_judge_all,
        )
        llm_report = validate_results(
            str(results_path),
            str(expected_path),
            strict_expected_ids=True,
            llm_judge=judge_config,
        )
        write_validation_report(llm_report, str(llm_report_path))
        llm_payload = llm_report.to_dict()

        robust_payload = write_robust_judge_report(
            dataset=dataset,
            run_rows=runs,
            result_rows=results,
            expected_rows=expected,
            strict_payload=strict_payload,
            output_path=robust_report_path,
            token=token,
            endpoint=args.judge_api_base,
            model=args.judge_model,
            timeout=args.judge_timeout,
            retries=args.judge_retries,
        )

    summary = {
        "dataset": str(args.dataset),
        "direct_run": str(args.run),
        "expected_adapter": str(expected_path),
        "result_adapter": str(results_path),
        "strict_report": str(strict_report_path),
        "llm_judge_report": str(llm_report_path) if llm_payload else None,
        "agent_style_robust_judge_report": str(robust_report_path) if robust_payload else None,
        "strict": summarize_agent_report(strict_payload),
        "llm_judge": summarize_agent_report(llm_payload),
        "agent_style_robust_judge": {
            key: robust_payload.get(key)
            for key in [
                "total",
                "strict_local_correct",
                "llm_calls",
                "judge_correct",
                "judge_incorrect",
                "judge_unknown",
                "judge_accuracy_counting_unknown_wrong",
                "judge_accuracy_known_only",
            ]
        }
        if robust_payload
        else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
