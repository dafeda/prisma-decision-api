# Gas Trading PRISMA Example

This example shows how PRISMA can help a gas trader decide whether it is worth waiting for better information before hedging a delivery-period position.

The question is:

```text
Should I hedge now, or wait for tomorrow's weather-demand report and then choose the hedge?
```

It uses the Python solver directly. It does not require the .NET API, a database, authentication, HTTP, or Docker.

## Model

The influence diagram contains:

- Decision: `Hedge action`
- Options: `Sell 500 MWh`, `Hold`, `Buy 500 MWh`
- Uncertainties: `Weather demand`, `Storage level`, `Supply shock`
- Utility: `Risk-adjusted trading value`

The utility table is populated for every combination of hedge action and market outcomes:

```text
net_position_mwh = current_position_mwh + action_volume_mwh
scenario_price = reference_price + sum(price_impacts)
pnl = net_position_mwh * (scenario_price - reference_price)
risk_penalty = abs(net_position_mwh) * risk_penalty_per_mwh
utility = pnl - risk_penalty
```

Assumptions:

- `current_position_mwh = 1000`
- `reference_price = 40 EUR/MWh`
- `risk_penalty_per_mwh = 2 EUR/MWh`

The runner solves two diagrams:

- **Decide now**: the hedge decision has no observed parent states.
- **Wait for report**: `Weather demand -> Hedge action`, so PRISMA returns the optimal hedge policy conditional on the observed weather-demand state.

## Run

Install the FastAPI/Python solver dependencies once with `uv`:

```bash
cd PrismaFastApi
uv python install 3.11
uv venv --python 3.11
uv pip install -e .
```

Run the gas trading example:

```bash
uv run python ../examples/gas-trading/run_python_solver.py
```

Expected output:

```text
Gas trading hedge decision

Decide now:
  Buy 500 MWh (EV 900 EUR)

Wait for weather report:
  If weather demand is Cold: Buy 500 MWh
  If weather demand is Mild: Sell 500 MWh
  If weather demand is Normal: Buy 500 MWh
  EV 1875 EUR

Value of the weather report: 975 EUR per delivery period
```

The important result is not just the higher expected value. The solver returns a conditional policy: waiting for the report flips the mild-weather action from buying more gas to selling 500 MWh. That is the value of an influence-diagram decision model over a static expected-value spreadsheet.

The script writes the full solver output to:

```text
examples/gas-trading/gas-trading-analysis-result.json
```
