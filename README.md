# Semiconductor end-to-end manufacturing optimization

A wafer plan can look profitable at 30 million dies until yield moves under it. This app buys wafers four ways and then makes the plans survive demand, yield, and cost weather.

## What it does

This Streamlit app models wafer sourcing for a fabless semiconductor company. It compares deterministic, static robust, two-stage adaptive, and ellipsoidal approaches under uncertainty in yield, cost, and partially-good die revenue.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## The setup

The buyer can source through two models:

- `KGD` - known good die; higher unit cost, lower buyer yield risk.
- `WB` - wafer buy; lower wafer cost, more testing, handling, and yield exposure.

The question is how many wafers to buy from each supplier and sourcing model when the yield and cost assumptions can move.

## Four ways to solve it

| Model | What it does |
|---|---|
| Nominal | Deterministic MILP with no uncertainty protection. |
| Static robust | Bertsimas-Sim budgeted uncertainty over a 5D set. |
| Two-stage adaptive | Stage 1 commits wafers and hardware; stage 2 reacts after yield is observed. |
| Ellipsoidal | Approximate comparison point, labeled as such in the app. |

Shared uncertainty set:

```text
Z = { z in R^5 : |z_i| <= 1, sum_i |z_i| <= Gamma }
```

## Default-scenario results

| Scenario | Result |
|---|---|
| `10M` demand, nominal | Supplier B `KGD` only: `(KGD-A, KGD-B, WB-A, WB-B) = (0, 14185, 0, 0)` |
| `30M` demand, nominal | `30.00M` good dies, about `$156.3M` nominal profit |
| `30M`, static robust, `Gamma = 1.0` | `32.12M` good dies, `$94.1M` nominal, `$51.9M` worst-case guarantee |
| `30M`, Monte Carlo, 5,000 scenarios | Demand met in `49.0%` nominal, `91.2%` static robust, `88.1%` adaptive |
| `50M` demand, nominal | Uses all four channels: `(9224, 30000, 7500, 20000)` |

The adaptive payoff includes salvage and lost-sales penalty. Treat its profit numbers as outputs of that formulation with a different accounting boundary from the static models.

## Validation

```bash
python validate.py
```

The validator checks solver sanity, `Gamma` monotonicity, `Gamma = 0` consistency with nominal, optimizer/simulator yield consistency, shortage revenue behavior, adaptive sensitivity to set widths, fractional-`Gamma` handling, and app-path smoke tests.

## Live demo

Deploy with Streamlit Cloud using:

```text
streamlit_app.py
```

Local run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Floorplan

```text
app.py
validate.py
requirements.txt
ProjectReport.pdf
ROText.pdf
src/
  model.py
  optimizer.py
  adaptive.py
  simulator.py
  visualization.py
```

## Connects to

- `chip-supply-chain-map` for the upstream dependency graph that motivates wafer-sourcing stress tests.
- `supplier-risk-rag-agent` for cited public filing risk text that can inform scenarios.
- `Robust-Facility-Location` for the shared uncertainty-modeling pattern.

## Limits

- The adaptive model observes yield only.
- The ellipsoidal formulation is approximate.
- The model is single-period.
- Contract features, expedite options, and reputation effects are outside the current scope.

## License

MIT.
