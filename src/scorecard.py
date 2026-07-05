"""Headless robust-formulation scorecard for the wafer optimizer."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_OUTPUT = Path("reports") / "scorecard.jsonl"
FORMULATIONS = ("nominal", "box", "ellipsoidal")
OPTIMAL_STATUS = "optimal"
DELEGATE_ENV = "SCORECARD_NO_PYTHON_DELEGATE"
# Mirrors the default lost_sales_penalty_frac in src/adaptive.py solve_two_stage.
LOST_SALES_PENALTY_FRAC = 1.50

PLANNED_FIELDS = (
    "robust_objective",
    "total_cost",
    "revenue",
    "nominal_profit",
    "total_good_dies",
    "demand_ratio",
)
REALIZED_FIELDS = (
    "expected_profit",
    "worst_case_profit",
    "cvar_95",
    "var_95",
    "demand_met_pct",
)


class ScorecardError(Exception):
    """Typed scorecard failure suitable for headless CLI output."""


def _dependency_available() -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import cvxpy  # noqa: F401
            import numpy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _candidate_pythons() -> list[Path]:
    candidates: list[Path] = []
    win_python = Path("C:/Python314/python.exe")
    if win_python.exists():
        candidates.append(win_python)

    return candidates


def _delegate_if_needed(argv: list[str]) -> int | None:
    if os.environ.get(DELEGATE_ENV) == "1" or _dependency_available():
        return None

    env = os.environ.copy()
    env[DELEGATE_ENV] = "1"
    for candidate in _candidate_pythons():
        if candidate == Path(sys.executable):
            continue

        command = [str(candidate), "-m", "src.scorecard", *argv]
        result = subprocess.run(command, env=env)
        return result.returncode

    raise ScorecardError("missing dependencies: numpy and cvxpy are unavailable")


def _parse_args(argv: list[str]) -> Path:
    output = DEFAULT_OUTPUT
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--output", "-o"):
            if i + 1 >= len(argv):
                raise ScorecardError("--output requires a path")
            output = Path(argv[i + 1])
            i += 2
            continue
        raise ScorecardError(f"unknown argument: {arg}")
    return output


def _load_engine() -> tuple[Any, Any, Any, Any]:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from src.model import create_default_instance
            from src.optimizer import solve
            from src.simulator import evaluate_solution, generate_scenarios
    except ModuleNotFoundError as exc:
        raise ScorecardError(f"missing dependency or module: {exc.name}") from exc
    except Exception as exc:
        raise ScorecardError(f"engine import failed: {exc}") from exc

    return create_default_instance, solve, generate_scenarios, evaluate_solution


def _number(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ScorecardError(f"{field} is not numeric: {value!r}") from exc


def build_rows() -> list[dict[str, float | str]]:
    import numpy as np

    create_default_instance, solve, generate_scenarios, evaluate_solution = _load_engine()

    instance = create_default_instance(demand=10_000_000)
    scenarios = generate_scenarios(instance, n_simulations=2000, seed=42)
    price = _number(
        instance.costs.selling_price_per_die,
        "selling_price_per_die",
    )
    rows: list[dict[str, float | str]] = []

    for formulation in FORMULATIONS:
        result = solve(instance, formulation=formulation)
        if result.status != OPTIMAL_STATUS:
            raise ScorecardError(
                f"{formulation} solve returned non-optimal status: {result.status}"
            )

        sim = evaluate_solution(result, instance, scenarios)
        row: dict[str, float | str] = {"formulation": formulation}
        for field in PLANNED_FIELDS:
            row[field] = _number(getattr(result, field), field)
        for field in REALIZED_FIELDS:
            row[field] = _number(getattr(sim, field), field)

        shortfall = np.maximum(0.0, float(sim.demand) - sim.total_dies)
        penalized_profit = sim.profits - LOST_SALES_PENALTY_FRAC * price * shortfall
        row["profit_std"] = float(sim.profits.std())
        row["penalized_worst_case_profit"] = float(penalized_profit.min())
        rows.append(row)

    nominal_worst = _number(rows[0]["worst_case_profit"], "nominal worst_case_profit")
    nominal_penalized_worst = _number(
        rows[0]["penalized_worst_case_profit"],
        "nominal penalized_worst_case_profit",
    )
    for row in rows:
        row["worst_case_delta"] = (
            _number(row["worst_case_profit"], "worst_case_profit") - nominal_worst
        )
        row["penalized_worst_case_delta"] = (
            _number(row["penalized_worst_case_profit"], "penalized_worst_case_profit")
            - nominal_penalized_worst
        )

    return rows


def write_scorecard(output_path: Path) -> list[dict[str, float | str]]:
    rows = build_rows()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    output_path.write_text(payload, encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        delegated = _delegate_if_needed(argv)
        if delegated is not None:
            return delegated

        output_path = _parse_args(argv)
        rows = write_scorecard(output_path)
        best = max(
            rows,
            key=lambda row: _number(
                row["penalized_worst_case_profit"],
                "penalized_worst_case_profit",
            ),
        )
        print(
            f"wrote {len(rows)} rows to {output_path}; "
            f"best_penalized_worst_case_profit={best['formulation']}"
        )
        return 0
    except ScorecardError as exc:
        print(f"ERROR[scorecard]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR[scorecard]: unexpected failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
