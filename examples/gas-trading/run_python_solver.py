#!/usr/bin/env python3
"""Show how PRISMA values waiting for a gas demand signal before hedging.

This bypasses the .NET API, database, authentication, HTTP, and Docker. It builds
the PRISMA decision/uncertainty/utility DTOs directly and calls the Python solver.

Install the FastAPI dependencies once with uv:

    cd PrismaFastApi
    uv python install 3.11
    uv venv --python 3.11
    uv pip install -e .

Then run:

    uv run python ../examples/gas-trading/run_python_solver.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import sys


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
FASTAPI_DIR = REPO_ROOT / "PrismaFastApi"
DEFAULT_OUTPUT_PATH = THIS_DIR / "gas-trading-analysis-result.json"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(FASTAPI_DIR))

from gas_trading_model import build_model  # noqa: E402
from src.constants import Type  # noqa: E402
from src.dtos.model_solution_dtos import DecisionSolution  # noqa: E402
from src.services.solver_service import SolverService  # noqa: E402


def approx_equal(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(left - right) <= tolerance


def only_decision(solution: Any) -> DecisionSolution:
    assert len(solution.decision_solutions) == 1
    return solution.decision_solutions[0]


def unconditional_action(decision: DecisionSolution) -> str:
    assert len(decision.optimal_decisions) == 1
    return decision.optimal_decisions[0].state.name


def policy_by_parent_state(decision: DecisionSolution) -> list[dict[str, Any]]:
    rows = []
    for optimal in decision.optimal_decisions:
        states = [state.state.name for state in optimal.parent_states]
        rows.append(
            {
                "observed_states": states,
                "action": optimal.state.name,
            }
        )
    return sorted(rows, key=lambda row: row["observed_states"])


def direct_expected_value_for_action(action: str) -> float:
    issues, _ = build_model()
    probabilities = {}
    for issue in issues:
        if issue.type != Type.UNCERTAINTY.value:
            continue
        assert issue.uncertainty is not None
        for row in issue.uncertainty.discrete_probabilities:
            probabilities[row.outcome_id] = row.probability

    utility_issue = next(issue for issue in issues if issue.type == Type.UTILITY.value)
    decision_issue = next(issue for issue in issues if issue.type == Type.DECISION.value)
    assert utility_issue.utility is not None
    assert decision_issue.decision is not None
    option_by_id = {option.id: option.name for option in decision_issue.decision.options}

    expected_value = 0.0
    for row in utility_issue.utility.discrete_utilities:
        if option_by_id[row.parent_option_ids[0]] != action:
            continue
        probability = 1.0
        for outcome_id in row.parent_outcome_ids:
            probability *= probabilities[outcome_id]
        expected_value += probability * (row.utility_value or 0)
    return expected_value


async def solve(observe: list[str] | None = None):
    issues, edges = build_model(observe=observe or [])
    return await SolverService().find_optimal_decision_pyagrum_from_dtos(issues, edges)


async def run(output_path: Path) -> dict[str, Any]:
    # Choose the best single hedge action now, before knowing the weather-demand outcome.
    decide_now = only_decision(await solve())
    # Choose a hedge policy after observing weather demand.
    wait_for_weather = only_decision(await solve(observe=["weather-demand"]))
    value_of_information = wait_for_weather.mean - decide_now.mean

    result = {
        "decide_now": {
            "expected_value_eur": decide_now.mean,
            "action": unconditional_action(decide_now),
        },
        "wait_for_weather_report": {
            "expected_value_eur": wait_for_weather.mean,
            "policy": policy_by_parent_state(wait_for_weather),
        },
        "value_of_weather_report_eur": value_of_information,
        "solver_solutions": {
            "decide_now": json.loads(decide_now.model_dump_json()),
            "wait_for_weather_report": json.loads(wait_for_weather.model_dump_json()),
        },
    }

    assert approx_equal(result["decide_now"]["expected_value_eur"], 900)
    assert approx_equal(
        result["decide_now"]["expected_value_eur"],
        direct_expected_value_for_action(result["decide_now"]["action"]),
    )
    assert approx_equal(result["wait_for_weather_report"]["expected_value_eur"], 1875)
    assert approx_equal(result["value_of_weather_report_eur"], 975)
    mild_row = next(
        row
        for row in result["wait_for_weather_report"]["policy"]
        if row["observed_states"] == ["Mild"]
    )
    assert mild_row["action"] == "Sell 500 MWh"

    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write analysis result JSON.",
    )
    args = parser.parse_args()

    result = asyncio.run(run(args.output))
    print("Gas trading hedge decision")
    print()
    print("Decide now:")
    print(
        f"  {result['decide_now']['action']} "
        f"(EV {result['decide_now']['expected_value_eur']:.0f} EUR)"
    )
    print()
    print("Wait for weather report:")
    for row in result["wait_for_weather_report"]["policy"]:
        observed = ", ".join(row["observed_states"])
        print(f"  If weather demand is {observed}: {row['action']}")
    print(
        f"  EV {result['wait_for_weather_report']['expected_value_eur']:.0f} EUR"
    )
    print()
    print(
        "Value of the weather report: "
        f"{result['value_of_weather_report_eur']:.0f} EUR per delivery period"
    )
    print()
    print(f"Wrote full result to {args.output}")


if __name__ == "__main__":
    main()
