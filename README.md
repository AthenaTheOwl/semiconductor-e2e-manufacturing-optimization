# Semiconductor Wafer Supply Chain: Robust & Adaptive Optimization

An interactive application that formulates, solves, and simulates robust optimization models for a fabless semiconductor wafer supply chain under uncertainty. Built to make the math behind robust optimization tangible -- showing how different scenarios play out and why protecting against uncertainty is worth the cost.

## The Problem

**Fabless** semiconductor companies (Qualcomm, MediaTek, etc.) design chips but own no fabrication plants. They source wafers from external fabs and must navigate a complex supply chain: fabrication, wafer testing, assembly/packaging, final test, and delivery to OEMs.

Two sourcing models are available from each fab:

| | KGD (Known Good Die) | WB (Wafer Buy) |
|---|---|---|
| **Pricing** | Fixed cost per wafer (yield risk on fab) | Lower cost per wafer (yield risk on buyer) |
| **Testing** | Fab handles all testing logistics | Buyer handles testing, needs probe card hardware |
| **Partial dies** | No access to partially good dies | Can harvest partial dies for extra revenue |
| **When preferred** | Lower demand (simpler, less risk) | Higher demand (cheaper per die at scale) |

The company must decide how many wafers to buy from each supplier under each model, while facing uncertainty in:
- **Yield** -- how many good dies each wafer produces
- **Testing costs** -- WB overhead varies with wafer quality
- **Downstream costs** -- packaging, final test, QA costs
- **Partial-die revenue** -- uncertain market for imperfect chips

The question: how do you allocate wafers across suppliers and models to maximize profit while protecting against these uncertainties?

## What's Built

### Optimization Engine

Three optimization formulations, all using the same 5-dimensional Bertsimas-Sim budget uncertainty set `Z = { z in R^5 : |z_i| <= 1, sum |z_i| <= Gamma }`:

**Nominal** -- deterministic MIP. No uncertainty protection. Solves `max Revenue - Cost` subject to demand and capacity constraints. Baseline for comparison.

**Static Robust (Bertsimas-Sim)** -- all decisions made before any uncertainty is revealed. Uses the B-S LP reformulation with auxiliary variables over the full 5D set. Both the demand feasibility constraint and the profit guarantee are protected under one shared budget. Epigraph form: `max t` subject to `nominal_profit - protection >= t`. Solved as MIP with HiGHS.

**Two-Stage Adaptive** -- stage-1 commits wafers and hardware; after observing yield (z1, z2), stage-2 chooses fulfillment, salvage, and lost sales. Cost/revenue parameters (z3-z5) remain adversarial and unobserved. Uses vertex enumeration over the full 5D budget set with proper fractional-Gamma support. Nonanticipativity enforced: same yield observation implies same recourse action. Per-vertex profit constraints use the full (z3, z4, z5) cost realization. Solved as MIP with HiGHS.

Additionally, an **Ellipsoidal (approximate)** formulation is included for comparison. It uses an SOCP demand constraint (`rho * ||deviation_vec||_2`) solved via CLARABEL with LP relaxation + rounding. Labeled as approximate throughout.

### Monte Carlo Simulator

Evaluates any allocation against thousands of random scenarios:
- Yield randomness driven by supplier-level `yield_std / yield_mean` (physical process variation, separate from robust set widths)
- Cost/revenue randomness driven by the set-width parameters
- **Shortage-aware revenue**: `revenue = price * min(produced, demand)` -- you can't sell what you didn't make
- **Adaptive payoff**: includes salvage revenue (30% of selling price) and lost-sales penalty (150%)
- Reports demand-met %, expected/worst-case/VaR/CVaR profit, surplus distribution

### Interactive Dashboard (Streamlit)

Five tabs:

1. **Supply Chain Overview** -- interactive Plotly flow diagram, KGD vs WB explanation, supplier parameter tables, LaTeX formulation
2. **Optimization Results** -- side-by-side comparison of all four formulations (Nominal, Static Robust, Ellipsoidal, Two-Stage Adaptive) with cost breakdowns and demand sweep charts
3. **Uncertainty Analysis** -- 2D visualization of the B-S budget polytope vs ellipsoid, Gamma sweep showing cost-of-robustness curve
4. **Monte Carlo Simulation** -- profit distribution histograms, demand fulfillment ratios, detailed statistics table comparing all formulations
5. **Sensitivity & Insights** -- tornado chart of parameter sensitivity, cost-of-robustness across demand levels, documented limitations

All supplier parameters, cost structures, uncertainty set widths, and simulation settings are tunable from the sidebar.

### Validation Suite

85 automated checks (`python validate.py`) covering:
- Solver optimality and capacity constraint satisfaction across 7 demand levels
- Gamma=0 reproduces nominal solution
- Gamma monotonicity (robust objective non-increasing)
- Yield model consistency between optimizer and simulator
- Shortage revenue capping
- Adaptive cost sensitivity (downstream and yield set widths change the objective)
- Fractional Gamma vertex correctness
- Policy consistency at the nominal scenario
- App-path smoke tests (UncertaintyConfig construction doesn't crash)

## Tech Stack

| Component | Technology |
|---|---|
| Optimization | [cvxpy](https://www.cvxpy.org/) with HiGHS (MIP) and CLARABEL (SOCP) solvers |
| Dashboard | [Streamlit](https://streamlit.io/) |
| Visualization | [Plotly](https://plotly.com/python/) |
| Numerics | NumPy, SciPy, Pandas |
| Language | Python 3.10+ |

## Project Structure

```
semiconductor-wafer-robust-optimization/
├── app.py                 # Streamlit dashboard (5 tabs, sidebar controls)
├── validate.py            # 85-check validation suite
├── requirements.txt
├── src/
│   ├── model.py           # Data classes: suppliers, costs, uncertainty, results
│   ├── optimizer.py        # Nominal + static robust + ellipsoidal solvers
│   ├── adaptive.py         # Two-stage adaptive solver (vertex enumeration)
│   ├── simulator.py        # Monte Carlo scenario generation + evaluation
│   └── visualization.py    # Plotly chart functions
```

## Setup

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

To run the validation suite:

```bash
python validate.py
```

## Default Parameters

Derived from the ProjectReport's formulation and results tables (independently parameterized, not a direct reproduction):

| Parameter | Supplier A (Premium) | Supplier B (Economy) |
|---|---|---|
| KGD capacity | 17,500 wafers/qtr | 30,000 wafers/qtr |
| WB capacity | 7,500 wafers/qtr | 20,000 wafers/qtr |
| KGD cost | $22,500/wafer | $15,750/wafer |
| WB cost | $20,000/wafer | $15,000/wafer |
| Good dies/wafer | 882 | 705 |
| Yield mean / std | 0.92 / 0.03 | 0.85 / 0.06 |
| Extra revenue (WB) | $787.50/wafer | $656.25/wafer |

Fixed costs: test $2,000/wafer, transport $80/wafer, engineering $5/wafer, hardware piece $300,000, selling price $35/die, downstream $7/die.

## Key Insights

**KGD dominates at low demand.** Below ~20M dies, Supplier B's KGD model alone satisfies demand at the lowest per-die cost ($22.34). The Wafer Buy model only enters when KGD capacity is exhausted.

**Robustness costs 5-15% but eliminates shortfall risk.** The static robust solution orders more wafers (5-11% surplus) at higher total cost. In exchange, it guarantees demand is met even under worst-case yield within the uncertainty set.

**The adaptive model improves worst-case profit significantly.** By observing yield before deciding fulfillment vs salvage, the two-stage model achieves a substantially higher worst-case guaranteed profit than the static model with the same uncertainty family.

**Yield uncertainty is the dominant risk factor.** Sensitivity analysis shows yield set width has the largest impact on both the allocation decision and the robust objective, consistent with industry experience where fab yield excursions are the primary supply chain risk.

**Supplier diversification increases with uncertainty.** At higher Gamma, the optimizer spreads allocation across both suppliers and both sourcing models, providing natural hedging against correlated disruptions.

## Known Limitations

- **Static vs adaptive payoffs differ.** The static model caps revenue at `min(produced, demand) * price`. The adaptive model adds salvage revenue (30%) and lost-sales penalty (150%). Their robust objectives are not directly comparable.
- **Ellipsoidal is approximate.** LP relaxation + rounding, not exact mixed-integer SOCP.
- **Independently parameterized.** Inspired by the ProjectReport but uses independently derived parameters. Results do not reproduce the appendix tables exactly.
- **No contractual penalties.** Neither model captures contract-specific penalties, reputation damage, or expediting costs beyond the stylized lost-sales parameter.
- **Single-period model.** Plans for one quarter. No multi-period inventory dynamics or demand forecasting.

## References

- Bertsimas, D. & den Hertog, D. *Robust and Adaptive Optimization* (ROText.pdf) -- uncertainty sets, B-S budget formulation, adaptive robust optimization
- Bertsimas, D. & Tsitsiklis, J.N. *Introduction to Linear Optimization* -- LP/MIP foundations
- Boyd, S. & Vandenberghe, L. *Convex Optimization* (bv_cvxbook) -- SOCP, ellipsoidal constraints
- ProjectReport.pdf -- original problem formulation for fabless semiconductor wafer supply chain optimization
