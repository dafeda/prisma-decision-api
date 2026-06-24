#!/usr/bin/env python3
"""Define a reproducible gas-trading decision model for value-of-information examples.

An influence diagram is a directed decision model with decision nodes, uncertainty
nodes, utility nodes, and edges showing what affects what. This example models a
gas trader choosing a hedge action: sell 500 MWh, hold, or buy 500 MWh. Weather
demand, storage level, and supply shock are uncertain market drivers with outcome
probabilities and price impacts. The hedge action and all uncertainty outcomes
feed into a risk-adjusted trading value utility table.

The base diagram is:

    Weather demand  ----\
    Storage level   ----- > Risk-adjusted trading value
    Supply shock    ----/
    Hedge action    ---/

Terminology:

    Issue: PRISMA's top-level object for a modeled decision-analysis item. Each
    issue has a type, such as Decision, Uncertainty, or Utility, and is backed by
    a diagram node.

    DTO: Data Transfer Object. These Pydantic models carry validated data between
    routes, services, solvers, and examples without being the database or solver
    implementation itself.

    IssueOutgoingDto: the full issue DTO used by the solver. Depending on its
    type, it contains a DecisionOutgoingDto, UncertaintyOutgoingDto, or
    UtilityOutgoingDto.

Value-of-information analysis estimates how much expected utility improves when
a decision can be made after learning uncertain information. Passing uncertainty
keys to build_model(observe=...) adds informational edges from those uncertainty
nodes into the hedge decision, representing observations available before the
trader chooses an action.

How this helps a trader:

    The model compares "hedge now" against "wait for information, then hedge".
    Without observations, the solver recommends the best single hedge action
    across all modeled weather, storage, and supply-shock scenarios. With an
    observation such as weather demand, the solver recommends a conditional hedge
    policy: what to do if demand is mild, normal, or cold. The difference in
    expected utility is the value of that information, which can be compared with
    the cost, delay, slippage, or execution risk of waiting for the signal.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Sequence

import sys


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
FASTAPI_DIR = REPO_ROOT / "PrismaFastApi"
sys.path.insert(0, str(FASTAPI_DIR))

from src.constants import Boundary, DecisionHierarchy, Type  # noqa: E402
from src.dtos.decision_dtos import DecisionOutgoingDto  # noqa: E402
from src.dtos.discrete_probability_dtos import (  # noqa: E402
    DiscreteProbabilityOutgoingDto,
)
from src.dtos.discrete_utility_dtos import DiscreteUtilityOutgoingDto  # noqa: E402
from src.dtos.edge_dtos import EdgeOutgoingDto  # noqa: E402
from src.dtos.issue_dtos import IssueOutgoingDto  # noqa: E402
from src.dtos.node_dtos import NodeOutgoingDto, NodeViaIssueOutgoingDto  # noqa: E402
from src.dtos.node_style_dtos import NodeStyleOutgoingDto  # noqa: E402
from src.dtos.option_dtos import OptionOutgoingDto  # noqa: E402
from src.dtos.outcome_dtos import OutcomeOutgoingDto  # noqa: E402
from src.dtos.shared_issue_node_dtos import IssueViaNodeOutgoingDto  # noqa: E402
from src.dtos.uncertainty_dtos import UncertaintyOutgoingDto  # noqa: E402
from src.dtos.utility_dtos import UtilityOutgoingDto  # noqa: E402


PROJECT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
CREATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class OptionSpec:
    name: str
    volume_mwh: float


@dataclass(frozen=True)
class StateSpec:
    name: str
    probability: float
    price_impact: float


OPTION_SPECS = [
    OptionSpec("Sell 500 MWh", -500),
    OptionSpec("Hold", 0),
    OptionSpec("Buy 500 MWh", 500),
]

UNCERTAINTY_SPECS = {
    "weather-demand": (
        "Weather demand",
        "Weather-driven gas demand for the delivery period.",
        2,
        350,
        80,
        [
            StateSpec("Mild", 0.25, -4),
            StateSpec("Normal", 0.50, 0),
            StateSpec("Cold", 0.25, 6),
        ],
    ),
    "storage-level": (
        "Storage level",
        "Regional gas storage level before delivery.",
        3,
        350,
        260,
        [
            StateSpec("High", 0.30, -3),
            StateSpec("Normal", 0.50, 0),
            StateSpec("Low", 0.20, 4),
        ],
    ),
    "supply-shock": (
        "Supply shock",
        "Unexpected supply disruption in the relevant market.",
        4,
        350,
        440,
        [
            StateSpec("None", 0.70, 0),
            StateSpec("Moderate", 0.20, 5),
            StateSpec("Severe", 0.10, 12),
        ],
    ),
}


def id_for(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"prisma-gas-trading-voi/{name}")


def calculate_utility(volume_mwh: float, price_impacts: Sequence[float]) -> float:
    current_position_mwh = 1000
    reference_price_eur_mwh = 40
    risk_penalty_eur_mwh = 2

    net_position_mwh = current_position_mwh + volume_mwh
    scenario_price = reference_price_eur_mwh + sum(price_impacts)
    pnl = net_position_mwh * (scenario_price - reference_price_eur_mwh)
    risk_penalty = abs(net_position_mwh) * risk_penalty_eur_mwh
    return pnl - risk_penalty


def make_node(issue_id: uuid.UUID, name: str, x: int, y: int) -> NodeViaIssueOutgoingDto:
    node_id = id_for(f"node/{issue_id}")
    return NodeViaIssueOutgoingDto(
        id=node_id,
        project_id=PROJECT_ID,
        issue_id=issue_id,
        name=name,
        node_style=NodeStyleOutgoingDto(
            id=id_for(f"node-style/{issue_id}"),
            node_id=node_id,
            x_position=x,
            y_position=y,
        ),
    )


def issue_via_node(issue: IssueOutgoingDto) -> IssueViaNodeOutgoingDto:
    return IssueViaNodeOutgoingDto(
        id=issue.id,
        project_id=issue.project_id,
        name=issue.name,
        description=issue.description,
        order=issue.order,
        type=issue.type,
        boundary=issue.boundary,
        decision=issue.decision,
        uncertainty=issue.uncertainty,
        utility=issue.utility,
    )


def node_out(issue: IssueOutgoingDto) -> NodeOutgoingDto:
    return NodeOutgoingDto(
        id=issue.node.id,
        project_id=issue.node.project_id,
        issue_id=issue.node.issue_id,
        name=issue.node.name,
        issue=issue_via_node(issue),
        node_style=issue.node.node_style,
    )


def make_decision_issue() -> tuple[IssueOutgoingDto, dict[uuid.UUID, OptionSpec]]:
    issue_id = id_for("issue/hedge-action")
    decision_id = id_for("decision/hedge-action")
    option_specs_by_id: dict[uuid.UUID, OptionSpec] = {}
    options = []

    for spec in OPTION_SPECS:
        option_id = id_for(f"option/{spec.name}")
        option_specs_by_id[option_id] = spec
        options.append(
            OptionOutgoingDto(
                id=option_id,
                name=spec.name,
                decision_id=decision_id,
            )
        )

    issue = IssueOutgoingDto(
        id=issue_id,
        project_id=PROJECT_ID,
        name="Hedge action",
        description="Choose the gas trading hedge action for the delivery period.",
        order=1,
        type=Type.DECISION.value,
        boundary=Boundary.IN.value,
        node=make_node(issue_id, "Hedge action", 100, 260),
        decision=DecisionOutgoingDto(
            id=decision_id,
            issue_id=issue_id,
            options=options,
            type=DecisionHierarchy.FOCUS.value,
        ),
        uncertainty=None,
        utility=None,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )
    return issue, option_specs_by_id


def make_probability(uncertainty_id: uuid.UUID, outcome_id: uuid.UUID, probability: float):
    return DiscreteProbabilityOutgoingDto(
        id=id_for(f"probability/{outcome_id}"),
        uncertainty_id=uncertainty_id,
        outcome_id=outcome_id,
        probability=probability,
        parent_outcome_ids=[],
        parent_option_ids=[],
    )


def make_uncertainty_issue(
    key: str,
    name: str,
    description: str,
    order: int,
    x: int,
    y: int,
    state_specs: Sequence[StateSpec],
) -> tuple[IssueOutgoingDto, dict[uuid.UUID, StateSpec]]:
    issue_id = id_for(f"issue/{key}")
    uncertainty_id = id_for(f"uncertainty/{key}")
    state_specs_by_id: dict[uuid.UUID, StateSpec] = {}
    outcomes = []

    for spec in state_specs:
        outcome_id = id_for(f"outcome/{key}/{spec.name}")
        state_specs_by_id[outcome_id] = spec
        outcomes.append(
            OutcomeOutgoingDto(
                id=outcome_id,
                name=spec.name,
                uncertainty_id=uncertainty_id,
            )
        )

    issue = IssueOutgoingDto(
        id=issue_id,
        project_id=PROJECT_ID,
        name=name,
        description=description,
        order=order,
        type=Type.UNCERTAINTY.value,
        boundary=Boundary.IN.value,
        node=make_node(issue_id, name, x, y),
        decision=None,
        uncertainty=UncertaintyOutgoingDto(
            id=uncertainty_id,
            issue_id=issue_id,
            is_key=True,
            outcomes=outcomes,
            discrete_probabilities=[
                make_probability(
                    uncertainty_id,
                    outcome.id,
                    state_specs_by_id[outcome.id].probability,
                )
                for outcome in outcomes
            ],
        ),
        utility=None,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )
    return issue, state_specs_by_id


def make_utility_issue(
    decision_issue: IssueOutgoingDto,
    option_specs_by_id: dict[uuid.UUID, OptionSpec],
    uncertainty_issues: Sequence[IssueOutgoingDto],
    state_specs_by_id: dict[uuid.UUID, StateSpec],
) -> IssueOutgoingDto:
    assert decision_issue.decision is not None
    issue_id = id_for("issue/risk-adjusted-trading-value")
    utility_id = id_for("utility/risk-adjusted-trading-value")
    outcome_groups = [
        issue.uncertainty.outcomes for issue in uncertainty_issues if issue.uncertainty
    ]
    utility_rows = []

    for option in decision_issue.decision.options:
        for outcomes in product(*outcome_groups):
            price_impacts = [
                state_specs_by_id[outcome.id].price_impact for outcome in outcomes
            ]
            utility_rows.append(
                DiscreteUtilityOutgoingDto(
                    id=id_for(
                        "utility/"
                        + str(utility_id)
                        + "/"
                        + str(option.id)
                        + "/"
                        + "/".join(str(outcome.id) for outcome in outcomes)
                    ),
                    utility_id=utility_id,
                    utility_value=calculate_utility(
                        option_specs_by_id[option.id].volume_mwh,
                        price_impacts,
                    ),
                    parent_outcome_ids=[outcome.id for outcome in outcomes],
                    parent_option_ids=[option.id],
                )
            )

    return IssueOutgoingDto(
        id=issue_id,
        project_id=PROJECT_ID,
        name="Risk-adjusted trading value",
        description=(
            "Scenario PnL less position risk penalty. Assumptions: current position "
            "1000 MWh, reference price 40 EUR/MWh, risk penalty 2 EUR/MWh."
        ),
        order=5,
        type=Type.UTILITY.value,
        boundary=Boundary.IN.value,
        node=make_node(issue_id, "Risk-adjusted trading value", 700, 260),
        decision=None,
        uncertainty=None,
        utility=UtilityOutgoingDto(
            id=utility_id,
            issue_id=issue_id,
            discrete_utilities=utility_rows,
        ),
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def make_edge(tail_issue: IssueOutgoingDto, head_issue: IssueOutgoingDto) -> EdgeOutgoingDto:
    return EdgeOutgoingDto(
        id=id_for(f"edge/{tail_issue.node.id}/{head_issue.node.id}"),
        tail_id=tail_issue.node.id,
        head_id=head_issue.node.id,
        project_id=PROJECT_ID,
        tail_node=node_out(tail_issue),
        head_node=node_out(head_issue),
        tail_issue_id=tail_issue.id,
        head_issue_id=head_issue.id,
    )


def build_model(
    observe: Sequence[str] = (),
) -> tuple[list[IssueOutgoingDto], list[EdgeOutgoingDto]]:
    decision_issue, option_specs_by_id = make_decision_issue()
    uncertainty_issues = []
    all_state_specs_by_id: dict[uuid.UUID, StateSpec] = {}

    for key, spec in UNCERTAINTY_SPECS.items():
        issue, state_specs_by_id = make_uncertainty_issue(key, *spec)
        uncertainty_issues.append(issue)
        all_state_specs_by_id.update(state_specs_by_id)

    utility_issue = make_utility_issue(
        decision_issue,
        option_specs_by_id,
        uncertainty_issues,
        all_state_specs_by_id,
    )
    issues = [decision_issue, *uncertainty_issues, utility_issue]
    edges = [
        make_edge(issue, utility_issue)
        for issue in [decision_issue, *uncertainty_issues]
    ]

    uncertainty_by_key = dict(zip(UNCERTAINTY_SPECS, uncertainty_issues, strict=True))
    edges.extend(make_edge(uncertainty_by_key[key], decision_issue) for key in observe)
    return issues, edges
