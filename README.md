<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

# N° 05 · semiconductor e2e

> *wafers, monte carlo, two-stage adaptive.*

an interactive streamlit app for fabless semiconductor wafer sourcing under uncertainty. the same robust-optimization toolkit from N° 04, pointed at a fab. formulate the model, solve it four different ways, then watch what happens across thousands of demand, yield, and cost scenarios.

`python` · `streamlit` · `cvxpy` · `HiGHS` · `CLARABEL` · `MIT` · 2024 · **status: solved**

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## the setup

a fabless company sources wafers from two suppliers, with two sourcing models:

- **`KGD`** *(known good die)* — operationally simpler, higher unit cost, lower yield risk on the buyer
- **`WB`** *(wafer buy)* — cheaper wafers, but the buyer absorbs more testing, handling, and yield uncertainty

the question: **how many wafers from each supplier × each model**, given uncertainty in yield, WB overhead cost, downstream cost, and extra revenue from partially-good dies.

## four ways to solve it

| model | story |
|---|---|
| **nominal**          | deterministic MILP. no uncertainty protection. the cheapest plan, and the most fragile. |
| **static robust**    | bertsimas-sim budgeted robust over a 5D uncertainty set. all decisions before uncertainty is revealed. protects feasibility *and* a worst-case profit floor. |
| **two-stage adaptive** | stage 1 commits wafers and hardware. stage 2 reacts after observing yield only. recourse: fulfillment, salvage, lost sales. solved by vertex enumeration over the same 5D set. |
| **ellipsoidal** *(approximate)* | comparison point. labeled approximate everywhere it appears. |

the shared uncertainty set:

```
Z = { z ∈ R⁵ : |zᵢ| ≤ 1, Σᵢ |zᵢ| ≤ Γ }
```

## the dashboard

five tabs:

1. supply chain overview
2. optimization results
3. uncertainty analysis
4. monte carlo simulation
5. sensitivity & insights

sidebar controls: supplier capacities, cost structure, yield mean / std, robust set widths, `Γ` / `ρ`, monte carlo sample count + seed.

## what the monte carlo says

the simulator fixes the optimized allocation, then evaluates it under random scenarios.

- yield randomness uses supplier-level `yield_mean` / `yield_std`
- cost and revenue randomness use configurable set widths
- static solutions get **shortage-aware revenue** — you can't sell dies you didn't produce
- adaptive solutions get a recourse payoff with fulfillment, salvage, and lost-sales penalty

reported: expected profit, worst-case / best-case observed profit, VaR / CVaR, demand-met %, surplus / deficit.

## what came back, on the default scenario

| scenario | result |
|---|---|
| `10M` demand, nominal | uses supplier B `KGD` only: `(KGD-A, KGD-B, WB-A, WB-B) = (0, 14185, 0, 0)` |
| `30M` demand, nominal | `30.00M` good dies · about `$156.3M` nominal profit |
| `30M`, static robust, `Γ = 1.0` | `32.12M` good dies · `$94.1M` nominal · `$51.9M` worst-case guarantee |
| `30M`, monte carlo · 5,000 scenarios | demand met in **49.0%** (nominal), **91.2%** (static robust), **88.1%** (adaptive) |
| `50M` demand, nominal | uses all four channels: `(9224, 30000, 7500, 20000)` |

a note: the adaptive payoff includes salvage and lost-sales penalty, so its profit numbers are decision-support outputs for that formulation — not direct apples-to-apples replacements for static.

## validation

```bash
python validate.py
```

checks: solver sanity (nominal / robust / ellipsoidal / adaptive), `Γ` monotonicity, `Γ = 0` consistency with nominal, optimizer / simulator yield consistency, shortage revenue behavior, adaptive sensitivity to set widths, fractional-`Γ` handling, app-path config smoke tests.

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## the floorplan

```
.
├── app.py
├── validate.py
├── requirements.txt
├── ProjectReport.pdf
├── ROText.pdf
└── src/
    ├── model.py
    ├── optimizer.py
    ├── adaptive.py
    ├── simulator.py
    └── visualization.py
```

## known limitations

- the adaptive model observes yield only — not realized cost / revenue
- the ellipsoidal formulation is approximate (and labeled as such)
- single-period, not multi-period
- doesn't capture every contract feature, expedite option, or reputation effect
- inspired by the included `ProjectReport.pdf`; not a line-by-line reproduction

## colophon

operations research references and `ProjectReport.pdf` informed the formulation. the goal was to make robust and adaptive optimization tangible — not to reproduce the report appendix.

`MIT` license. *built downstairs.* — [the basement, room 7](https://github.com/AthenaTheOwl)
