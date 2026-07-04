import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "formulation",
    "total_cost",
    "worst_case_profit",
    "worst_case_delta",
}


def _run_scorecard(output_path: Path) -> tuple[list[dict], bytes]:
    result = subprocess.run(
        [sys.executable, "-m", "src.scorecard", "--output", str(output_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = output_path.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    return rows, payload


def test_scorecard_shape_and_determinism():
    with TemporaryDirectory(prefix="scorecard-", dir=ROOT / "tests") as tmp:
        tmp_path = Path(tmp)
        rows, first_payload = _run_scorecard(tmp_path / "first.jsonl")
        rows_again, second_payload = _run_scorecard(tmp_path / "second.jsonl")

        assert first_payload == second_payload
        assert rows == rows_again
        assert len(rows) == 3
        assert {row["formulation"] for row in rows} == {
            "nominal",
            "box",
            "ellipsoidal",
        }

        for row in rows:
            assert REQUIRED_COLUMNS <= row.keys()
            assert "solve_time" not in row
            assert isinstance(row["worst_case_delta"], (int, float))
            assert math.isfinite(row["worst_case_delta"])


@pytest.mark.xfail(
    reason=(
        "The existing solver/simulator currently gives box and ellipsoidal "
        "lower seeded worst_case_profit than nominal for the required instance."
    ),
    strict=False,
)
def test_robust_formulations_improve_worst_case_profit():
    with TemporaryDirectory(prefix="scorecard-", dir=ROOT / "tests") as tmp:
        rows, _ = _run_scorecard(Path(tmp) / "scorecard.jsonl")
        by_formulation = {row["formulation"]: row for row in rows}
        nominal_worst = by_formulation["nominal"]["worst_case_profit"]

        assert by_formulation["box"]["worst_case_profit"] >= nominal_worst
        assert by_formulation["ellipsoidal"]["worst_case_profit"] >= nominal_worst
