"""Monte Carlo simulator for evaluating supply chain solutions under uncertainty.

Given a fixed wafer allocation (from the optimizer), simulates many random
realizations of uncertain parameters and evaluates actual profit, cost,
and demand fulfillment for each scenario.

Yield and cost/revenue randomness are SEPARATE from the robust set
widths used by the optimizer:
- YIELD randomness: driven by supplier-level yield_std / yield_mean
  (from the Supplier panels). This is the physical process variation.
- COST/REVENUE randomness: driven by the set-width sliders
  (cost_set_width, etc.). These double as simulation spread parameters.
"""

import numpy as np
from .model import (
    ProblemInstance, OptimizationResult, ScenarioRealization,
    SimulationOutput,
)


def generate_scenarios(
    instance: ProblemInstance,
    n_simulations: int = 10000,
    seed: int = 42,
) -> list[ScenarioRealization]:
    """Generate random realizations of uncertain parameters.

    Yield is modeled as a multiplicative factor on good_dies_per_wafer:
      realized_gdpw = good_dies_per_wafer * yield_factor
    where yield_factor ~ TruncatedNormal(1.0, yield_std/yield_mean).
    At yield_factor=1.0 the simulator reproduces the optimizer's planning basis.
    """
    rng = np.random.default_rng(seed)
    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs
    u = instance.uncertainty

    # Yield factor: multiplicative perturbation around 1.0
    yield_cv_a = sa.yield_std / sa.yield_mean if sa.yield_mean > 0 else 0.05
    yield_cv_b = sb.yield_std / sb.yield_mean if sb.yield_mean > 0 else 0.05
    yield_factor_a = rng.normal(1.0, yield_cv_a, n_simulations)
    yield_factor_a = np.clip(yield_factor_a, 0.5, 1.3)
    yield_factor_b = rng.normal(1.0, yield_cv_b, n_simulations)
    yield_factor_b = np.clip(yield_factor_b, 0.5, 1.3)

    # WB overhead cost distributions
    test_cost_samples = rng.normal(
        c.test_cost_per_wafer,
        u.cost_set_width * c.test_cost_per_wafer,
        n_simulations,
    )
    test_cost_samples = np.maximum(test_cost_samples, 0)

    transport_cost_samples = rng.normal(
        c.transport_cost_per_wafer,
        u.cost_set_width * c.transport_cost_per_wafer,
        n_simulations,
    )
    transport_cost_samples = np.maximum(transport_cost_samples, 0)

    eng_cost_samples = rng.normal(
        c.eng_cost_per_wafer,
        u.cost_set_width * c.eng_cost_per_wafer,
        n_simulations,
    )
    eng_cost_samples = np.maximum(eng_cost_samples, 0)

    # Downstream scale (multiplier around 1.0)
    downstream_scale_samples = rng.normal(
        1.0,
        u.downstream_set_width,
        n_simulations,
    )
    downstream_scale_samples = np.maximum(downstream_scale_samples, 0.5)

    # Extra revenue factor (uniform around 1.0)
    half_dev = u.extra_rev_set_width
    extra_rev_a_samples = rng.uniform(1.0 - half_dev, 1.0 + half_dev, n_simulations)
    extra_rev_b_samples = rng.uniform(1.0 - half_dev, 1.0 + half_dev, n_simulations)

    scenarios = []
    for i in range(n_simulations):
        scenarios.append(ScenarioRealization(
            yield_a=float(yield_factor_a[i]),
            yield_b=float(yield_factor_b[i]),
            test_cost_per_wafer=float(test_cost_samples[i]),
            transport_cost_per_wafer=float(transport_cost_samples[i]),
            eng_cost_per_wafer=float(eng_cost_samples[i]),
            downstream_scale=float(downstream_scale_samples[i]),
            extra_rev_factor_a=float(extra_rev_a_samples[i]),
            extra_rev_factor_b=float(extra_rev_b_samples[i]),
        ))

    return scenarios


def evaluate_solution(
    result: OptimizationResult,
    instance: ProblemInstance,
    scenarios: list[ScenarioRealization],
) -> SimulationOutput:
    """Evaluate a static solution against random scenarios.

    Revenue is capped at min(produced, demand) -- you cannot sell what
    you didn't produce. This is the main economic penalty that robust
    planning hedges against.
    """
    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs

    n = len(scenarios)
    profits = np.zeros(n)
    costs = np.zeros(n)
    revenues = np.zeros(n)
    total_dies = np.zeros(n)

    wkgd_a = result.wkgd_a
    wkgd_b = result.wkgd_b
    wwb_a = result.wwb_a
    wwb_b = result.wwb_b
    hw_pcs_a = result.hw_pcs_a
    hw_pcs_b = result.hw_pcs_b

    for i, s in enumerate(scenarios):
        gdpw_a = sa.good_dies_per_wafer * s.yield_a
        gdpw_b = sb.good_dies_per_wafer * s.yield_b
        dies = gdpw_a * (wkgd_a + wwb_a) + gdpw_b * (wkgd_b + wwb_b)
        total_dies[i] = dies

        wb_cost = sa.wb_cost_per_wafer * wwb_a + sb.wb_cost_per_wafer * wwb_b
        kgd_cost = sa.kgd_cost_per_wafer * wkgd_a + sb.kgd_cost_per_wafer * wkgd_b
        wb_overhead = (
            (s.test_cost_per_wafer + s.transport_cost_per_wafer + s.eng_cost_per_wafer)
            * (wwb_a + wwb_b)
        )
        hw_cost = c.hw_piece_cost * (hw_pcs_a + hw_pcs_b)

        actual_processed = min(dies, instance.demand)
        downstream = (
            s.downstream_scale * c.downstream_cost_per_die * actual_processed
            + s.downstream_scale * sa.downstream_extra_per_wb * wwb_a
            + s.downstream_scale * sb.downstream_extra_per_wb * wwb_b
        )
        total_cost = wb_cost + kgd_cost + wb_overhead + hw_cost + downstream
        costs[i] = total_cost

        sellable_dies = min(dies, instance.demand)
        extra_rev = (
            s.extra_rev_factor_a * sa.extra_rev_per_wb_wafer * wwb_a
            + s.extra_rev_factor_b * sb.extra_rev_per_wb_wafer * wwb_b
        )
        revenue = c.selling_price_per_die * sellable_dies + extra_rev
        revenues[i] = revenue
        profits[i] = revenue - total_cost

    return SimulationOutput(
        profits=profits, costs=costs, revenues=revenues,
        total_dies=total_dies, demand=instance.demand,
    )


def evaluate_adaptive_solution(
    result: OptimizationResult,
    instance: ProblemInstance,
    scenarios: list[ScenarioRealization],
    salvage_price_frac: float = 0.30,
    lost_sales_penalty_frac: float = 1.50,
) -> SimulationOutput:
    """Evaluate an adaptive solution with exact optimal recourse.

    Given the payoff structure (selling_price > salvage_price, penalty > 0),
    the optimal stage-2 recourse after observing production is:
      fulfill = min(produced, demand)
      salvage = max(produced - demand, 0)
    This is analytically optimal and does not require policy lookup.

    The profit computation uses the adaptive payoff structure (which
    includes salvage revenue and lost-sales penalty), NOT the static
    payoff (which just caps revenue at min(produced, demand) * price).
    """
    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs

    salvage_price = salvage_price_frac * c.selling_price_per_die
    lost_sales_penalty = lost_sales_penalty_frac * c.selling_price_per_die

    n = len(scenarios)
    profits = np.zeros(n)
    costs = np.zeros(n)
    revenues = np.zeros(n)
    total_dies = np.zeros(n)

    wkgd_a = result.wkgd_a
    wkgd_b = result.wkgd_b
    wwb_a = result.wwb_a
    wwb_b = result.wwb_b
    hw_pcs_a = result.hw_pcs_a
    hw_pcs_b = result.hw_pcs_b

    for i, s in enumerate(scenarios):
        gdpw_a = sa.good_dies_per_wafer * s.yield_a
        gdpw_b = sb.good_dies_per_wafer * s.yield_b
        dies = gdpw_a * (wkgd_a + wwb_a) + gdpw_b * (wkgd_b + wwb_b)
        total_dies[i] = dies

        # Exact optimal recourse
        fulfill = min(dies, instance.demand)
        salvage = max(dies - fulfill, 0.0)
        lost = instance.demand - fulfill

        # Costs
        wb_cost = sa.wb_cost_per_wafer * wwb_a + sb.wb_cost_per_wafer * wwb_b
        kgd_cost = sa.kgd_cost_per_wafer * wkgd_a + sb.kgd_cost_per_wafer * wkgd_b
        wb_overhead = (
            (s.test_cost_per_wafer + s.transport_cost_per_wafer + s.eng_cost_per_wafer)
            * (wwb_a + wwb_b)
        )
        hw_cost = c.hw_piece_cost * (hw_pcs_a + hw_pcs_b)
        downstream = (
            s.downstream_scale * c.downstream_cost_per_die * fulfill
            + s.downstream_scale * sa.downstream_extra_per_wb * wwb_a
            + s.downstream_scale * sb.downstream_extra_per_wb * wwb_b
        )
        total_cost = wb_cost + kgd_cost + wb_overhead + hw_cost + downstream
        costs[i] = total_cost

        # Adaptive payoff: sell + salvage + extra - penalty
        extra_rev = (
            s.extra_rev_factor_a * sa.extra_rev_per_wb_wafer * wwb_a
            + s.extra_rev_factor_b * sb.extra_rev_per_wb_wafer * wwb_b
        )
        revenue = (c.selling_price_per_die * fulfill
                   + salvage_price * salvage
                   + extra_rev
                   - lost_sales_penalty * lost)
        revenues[i] = revenue
        profits[i] = revenue - total_cost

    return SimulationOutput(
        profits=profits, costs=costs, revenues=revenues,
        total_dies=total_dies, demand=instance.demand,
    )


def compare_solutions(
    nominal_result: OptimizationResult,
    robust_result: OptimizationResult,
    instance: ProblemInstance,
    n_simulations: int = 10000,
    seed: int = 42,
    adaptive_result: OptimizationResult = None,
) -> tuple[SimulationOutput, SimulationOutput, SimulationOutput | None]:
    """Compare solutions under the same random scenarios.

    Returns (nominal_sim, robust_sim, adaptive_sim). adaptive_sim is None
    if adaptive_result is not provided.
    """
    scenarios = generate_scenarios(instance, n_simulations, seed)
    nominal_sim = evaluate_solution(nominal_result, instance, scenarios)
    robust_sim = evaluate_solution(robust_result, instance, scenarios)
    adaptive_sim = None
    if adaptive_result is not None:
        adaptive_sim = evaluate_adaptive_solution(adaptive_result, instance, scenarios)
    return nominal_sim, robust_sim, adaptive_sim
