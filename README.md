# Semiconductor E2E Manufacturing Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![CVXPY](https://img.shields.io/badge/CVXPY-optimization-2C7FB8)
![Validation](https://img.shields.io/badge/Validation-passing-brightgreen)
![Focus](https://img.shields.io/badge/Focus-robust%20optimization-6A1B9A)

Interactive Streamlit app for semiconductor wafer supply-chain optimization under uncertainty. It turns a fabless sourcing problem into a concrete end-to-end workflow: formulate the model, solve nominal and robust variants, then simulate outcomes across thousands of scenarios.

The project is inspired by the included operations research references and `ProjectReport.pdf`, but it is not a line-by-line reproduction of the report appendix. The goal is to make robust and adaptive optimization tangible, not just theoretical.

## App preview

![App preview](docs/app-preview.svg)

_Schematic dashboard preview showing the sidebar controls, optimization views, uncertainty analysis, and Monte Carlo evaluation flow._

## What this demonstrates

- How wafer allocation changes with demand, yield risk, and cost uncertainty
- The tradeoff between nominal profit and worst-case protection
- The difference between static robust planning and two-stage adaptive recourse
- How Monte Carlo simulation changes the discussion from a single optimal plan to a distribution of outcomes

## Problem setup

A fabless semiconductor company sources wafers from two suppliers using two sourcing models:

- `KGD` (Known Good Die): simpler operationally, higher effective unit cost, lower yield risk on the buyer
- `WB` (Wafer Buy): lower wafer price, but the buyer absorbs more testing/handling complexity and yield uncertainty

The decision is how many wafers to buy from each supplier and sourcing model while facing uncertainty in:

- Yield
- WB overhead cost
- Downstream cost
- Extra revenue from partially good dies

## Models included

### 1. Nominal

Deterministic mixed-integer model with no uncertainty protection. This is the baseline plan.

### 2. Static robust

Bertsimas-Sim budgeted robust optimization over a shared 5D uncertainty set:

`Z = { z in R^5 : |z_i| <= 1, sum_i |z_i| <= Gamma }`

The static model chooses all wafer and hardware decisions before uncertainty is revealed. It protects both:

- demand feasibility
- a worst-case profit lower bound

under the same uncertainty family.

### 3. Two-stage adaptive

Stage 1 commits wafers and hardware. Stage 2 reacts after observing yield only. Cost and revenue uncertainty remain adversarial and unobserved.

Recourse decisions include:

- fulfillment
- salvage
- lost sales

This model is solved by vertex enumeration of the same 5D Bertsimas-Sim set, with nonanticipativity enforced by observed yield state.

### 4. Ellipsoidal (approximate)

An approximate ellipsoidal robust formulation is included for comparison. It is explicitly labeled approximate in the app.

## App walkthrough

The Streamlit dashboard has five tabs:

1. `Supply Chain Overview`
2. `Optimization Results`
3. `Uncertainty Analysis`
4. `Monte Carlo Simulation`
5. `Sensitivity & Insights`

From the sidebar you can adjust:

- supplier capacities and cost structure
- yield means and standard deviations
- robust set widths
- `Gamma` / `Rho`
- Monte Carlo sample count and seed

## Monte Carlo simulation

The simulator evaluates solved allocations under random scenarios.

- Yield randomness comes from supplier-level `yield_mean` and `yield_std`
- Cost and revenue randomness use the configurable set-width inputs
- Static solutions use shortage-aware revenue: you cannot sell more dies than you produce
- Adaptive solutions use a recourse payoff with fulfillment, salvage, and lost-sales penalty

Reported outputs include:

- expected profit
- worst-case / best-case observed profit
- VaR / CVaR style risk metrics
- demand-met percentage
- surplus / deficit behavior

## Validation

Run:

```bash
python validate.py
```

The validation suite checks:

- nominal, robust, ellipsoidal, and adaptive solver sanity
- Gamma monotonicity
- `Gamma = 0` consistency with the nominal model
- optimizer / simulator yield consistency
- shortage revenue behavior
- adaptive sensitivity to set widths
- fractional-Gamma handling
- app-path configuration smoke tests

## Tech stack

- Python
- Streamlit
- CVXPY
- HiGHS
- CLARABEL
- NumPy
- Pandas
- Plotly
- SciPy

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python -m streamlit run app.py
```

## Example results

Using the bundled default parameters:

| Scenario | Example result |
|---|---|
| `10M` demand, nominal | Uses Supplier B `KGD` only: `(KGD-A, KGD-B, WB-A, WB-B) = (0, 14185, 0, 0)` |
| `30M` demand, nominal | `30.00M` good dies and about `$156.3M` nominal profit |
| `30M` demand, static robust, `Gamma = 1.0` | `32.12M` good dies, about `$94.1M` nominal profit, about `$51.9M` worst-case guarantee |
| `30M` demand, Monte Carlo, 5,000 scenarios | Demand met in about `49.0%` of nominal scenarios, `91.2%` for static robust, and `88.1%` for adaptive |
| `50M` demand, nominal | Uses all four channels: `(9224, 30000, 7500, 20000)` |

The adaptive model uses a different payoff definition from the static model because it includes salvage and lost-sales penalty. Treat adaptive profit figures as decision-support outputs for that formulation, not direct apples-to-apples replacements for the static objective.

## Repository structure

```text
.
|-- app.py
|-- validate.py
|-- requirements.txt
|-- ProjectReport.pdf
|-- ROText.pdf
`-- src/
    |-- model.py
    |-- optimizer.py
    |-- adaptive.py
    |-- simulator.py
    `-- visualization.py
```

## Default scenario

The bundled default instance uses:

- two suppliers: premium and economy
- KGD and WB capacity limits for each supplier
- nominal good-dies-per-wafer assumptions
- yield mean / standard deviation for simulation
- fixed WB testing, transport, engineering, hardware, selling-price, and downstream cost parameters

These defaults are independently parameterized from the project report and are meant to support interactive experimentation.

## How to interpret results

- `Nominal Profit` is the value of a plan under nominal coefficients
- `Robust Objective` is the worst-case guaranteed objective for the robust formulation
- For adaptive results, the payoff definition differs from the static model because salvage and lost-sales penalty are modeled explicitly

That means static and adaptive robust-objective values are informative, but not strictly apples-to-apples.

## Known limitations

- The adaptive model observes yield only, not realized cost / revenue coefficients
- The ellipsoidal formulation is approximate
- The project is single-period, not multi-period
- The model does not capture every real-world contract feature, expedite option, or reputation effect
- The app is inspired by the provided report, not intended as an exact reproduction of published appendix tables
