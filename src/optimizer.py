"""Optimization engine for semiconductor wafer supply chain.

Implements nominal, box-robust, and ellipsoidal-robust formulations
using cvxpy for the mixed-integer program.
"""

import time
import math
import numpy as np
import cvxpy as cp
from typing import Optional

from .model import (
    ProblemInstance, OptimizationResult, SupplierParams, CostParams,
    UncertaintyConfig, DEMAND_LEVELS, create_default_instance,
)


def _infeasible_result(instance: ProblemInstance, status: str, solve_time: float, formulation: str) -> OptimizationResult:
    """Return a result object for infeasible/failed solves."""
    return OptimizationResult(
        status=status, robust_objective=0,
        wkgd_a=0, wkgd_b=0, wwb_a=0, wwb_b=0,
        hw_pcs_a=0, hw_pcs_b=0,
        wb_cost=0, kgd_cost=0, wb_overhead=0, hw_cost=0,
        downstream_cost=0, total_cost=0, revenue=0, nominal_profit=0,
        total_good_dies=0, demand=instance.demand,
        demand_surplus=-instance.demand, demand_ratio=0,
        solve_time=solve_time, formulation=formulation,
    )


def _build_result(
    instance: ProblemInstance,
    wkgd_a_val: int, wkgd_b_val: int,
    wwb_a_val: int, wwb_b_val: int,
    status: str, obj_val: float, solve_time: float,
    formulation: str,
) -> OptimizationResult:
    """Build an OptimizationResult from solved variable values."""
    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs

    hw_pcs_a = math.ceil(wwb_a_val / sa.hw_wafers_per_piece) if wwb_a_val > 0 else 0
    hw_pcs_b = math.ceil(wwb_b_val / sb.hw_wafers_per_piece) if wwb_b_val > 0 else 0

    wb_cost = sa.wb_cost_per_wafer * wwb_a_val + sb.wb_cost_per_wafer * wwb_b_val
    kgd_cost = sa.kgd_cost_per_wafer * wkgd_a_val + sb.kgd_cost_per_wafer * wkgd_b_val
    wb_overhead = c.total_wb_overhead_per_wafer * (wwb_a_val + wwb_b_val)
    hw_cost = c.hw_piece_cost * (hw_pcs_a + hw_pcs_b)
    downstream_cost = (
        c.downstream_cost_per_die * instance.demand
        + sa.downstream_extra_per_wb * wwb_a_val
        + sb.downstream_extra_per_wb * wwb_b_val
    )

    total_cost = wb_cost + kgd_cost + wb_overhead + hw_cost + downstream_cost

    extra_rev = sa.extra_rev_per_wb_wafer * wwb_a_val + sb.extra_rev_per_wb_wafer * wwb_b_val
    revenue = c.selling_price_per_die * instance.demand + extra_rev
    nominal_profit = revenue - total_cost

    total_good_dies = (
        sa.good_dies_per_wafer * (wkgd_a_val + wwb_a_val)
        + sb.good_dies_per_wafer * (wkgd_b_val + wwb_b_val)
    )

    return OptimizationResult(
        status=status,
        robust_objective=obj_val,
        wkgd_a=wkgd_a_val,
        wkgd_b=wkgd_b_val,
        wwb_a=wwb_a_val,
        wwb_b=wwb_b_val,
        hw_pcs_a=hw_pcs_a,
        hw_pcs_b=hw_pcs_b,
        wb_cost=wb_cost,
        kgd_cost=kgd_cost,
        wb_overhead=wb_overhead,
        hw_cost=hw_cost,
        downstream_cost=downstream_cost,
        total_cost=total_cost,
        revenue=revenue,
        nominal_profit=nominal_profit,
        total_good_dies=total_good_dies,
        demand=instance.demand,
        demand_surplus=total_good_dies - instance.demand,
        demand_ratio=total_good_dies / instance.demand if instance.demand > 0 else 0,
        solve_time=solve_time,
        formulation=formulation,
    )


def solve_nominal(instance: ProblemInstance, integer: bool = True) -> OptimizationResult:
    """Solve the nominal (deterministic) optimization problem.

    max  Revenue - TotalCost
    s.t. capacity constraints, demand constraint, cost linkages
    """
    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs

    t0 = time.time()

    # Decision variables
    if integer:
        wkgd_a = cp.Variable(integer=True, name="WKGDa")
        wkgd_b = cp.Variable(integer=True, name="WKGDb")
        wwb_a = cp.Variable(integer=True, name="WWBa")
        wwb_b = cp.Variable(integer=True, name="WWBb")
        hw_a = cp.Variable(integer=True, name="HWPcsA")
        hw_b = cp.Variable(integer=True, name="HWPcsB")
    else:
        wkgd_a = cp.Variable(nonneg=True, name="WKGDa")
        wkgd_b = cp.Variable(nonneg=True, name="WKGDb")
        wwb_a = cp.Variable(nonneg=True, name="WWBa")
        wwb_b = cp.Variable(nonneg=True, name="WWBb")
        hw_a = cp.Variable(nonneg=True, name="HWPcsA")
        hw_b = cp.Variable(nonneg=True, name="HWPcsB")

    constraints = [
        # Non-negativity
        wkgd_a >= 0, wkgd_b >= 0, wwb_a >= 0, wwb_b >= 0,
        hw_a >= 0, hw_b >= 0,

        # Capacity constraints
        wkgd_a <= sa.kgd_capacity,
        wkgd_b <= sb.kgd_capacity,
        wwb_a <= sa.wb_capacity,
        wwb_b <= sb.wb_capacity,

        # Hardware piece constraints (ceiling: hw_a >= wwb_a / hw_wafers_per_piece)
        hw_a * sa.hw_wafers_per_piece >= wwb_a,
        hw_b * sb.hw_wafers_per_piece >= wwb_b,

        # Demand constraint: total good dies >= demand
        sa.good_dies_per_wafer * (wkgd_a + wwb_a)
        + sb.good_dies_per_wafer * (wkgd_b + wwb_b)
        >= instance.demand,
    ]

    # Cost components (as expressions)
    wb_cost = sa.wb_cost_per_wafer * wwb_a + sb.wb_cost_per_wafer * wwb_b
    kgd_cost = sa.kgd_cost_per_wafer * wkgd_a + sb.kgd_cost_per_wafer * wkgd_b
    wb_overhead = c.total_wb_overhead_per_wafer * (wwb_a + wwb_b)
    hw_cost = c.hw_piece_cost * (hw_a + hw_b)
    downstream = (
        c.downstream_cost_per_die * instance.demand  # fixed per demand level
        + sa.downstream_extra_per_wb * wwb_a
        + sb.downstream_extra_per_wb * wwb_b
    )

    total_cost = wb_cost + kgd_cost + wb_overhead + hw_cost + downstream

    # Revenue
    extra_rev = sa.extra_rev_per_wb_wafer * wwb_a + sb.extra_rev_per_wb_wafer * wwb_b
    revenue = c.selling_price_per_die * instance.demand + extra_rev

    # Objective: maximize profit
    objective = cp.Maximize(revenue - total_cost)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS, verbose=False)

    solve_time = time.time() - t0

    if prob.status in ("optimal", "optimal_inaccurate"):
        return _build_result(
            instance,
            wkgd_a_val=int(np.round(float(wkgd_a.value))),
            wkgd_b_val=int(np.round(float(wkgd_b.value))),
            wwb_a_val=int(np.round(float(wwb_a.value))),
            wwb_b_val=int(np.round(float(wwb_b.value))),
            status=prob.status,
            obj_val=float(prob.value),
            solve_time=solve_time,
            formulation="nominal",
        )

    return _infeasible_result(instance, prob.status, solve_time, "nominal")


def solve_robust_box(instance: ProblemInstance, integer: bool = True) -> OptimizationResult:
    """Solve static robust optimization with a single 5D Bertsimas-Sim set.

    Uses the SAME uncertainty family as the adaptive model:
        Z = { z in R^5 : |z_l| <= 1 for all l, sum_l |z_l| <= Gamma }

    All 5 uncertain parameters share ONE budget:
      z1: yield_a          (DECREASE is adverse)
      z2: yield_b          (DECREASE is adverse)
      z3: WB overhead cost (INCREASE is adverse)
      z4: downstream cost  (INCREASE is adverse)
      z5: extra revenue    (DECREASE is adverse)

    Because this is a static (no-recourse) model, all decisions are fixed
    before any uncertainty is revealed. The B-S LP reformulation handles
    both the demand constraint AND the profit guarantee under one shared
    budget, using a single set of auxiliary variables (z, p1..p5).

    The demand constraint requires: for all z in Z,
      (gdpw_a - dev_a*z1) * total_a + (gdpw_b - dev_b*z2) * total_b >= demand

    The profit guarantee requires: for all z in Z,
      nominal_profit - yield_penalty - cost_penalty >= t

    Both are reformulated as standard B-S LP duals over the same 5D set.
    """
    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs
    u = instance.uncertainty
    gamma = u.gamma

    t0 = time.time()

    if integer:
        wkgd_a = cp.Variable(integer=True, name="WKGDa")
        wkgd_b = cp.Variable(integer=True, name="WKGDb")
        wwb_a = cp.Variable(integer=True, name="WWBa")
        wwb_b = cp.Variable(integer=True, name="WWBb")
        hw_a = cp.Variable(integer=True, name="HWPcsA")
        hw_b = cp.Variable(integer=True, name="HWPcsB")
    else:
        wkgd_a = cp.Variable(nonneg=True, name="WKGDa")
        wkgd_b = cp.Variable(nonneg=True, name="WKGDb")
        wwb_a = cp.Variable(nonneg=True, name="WWBa")
        wwb_b = cp.Variable(nonneg=True, name="WWBb")
        hw_a = cp.Variable(nonneg=True, name="HWPcsA")
        hw_b = cp.Variable(nonneg=True, name="HWPcsB")

    constraints = [
        wkgd_a >= 0, wkgd_b >= 0, wwb_a >= 0, wwb_b >= 0,
        hw_a >= 0, hw_b >= 0,
        wkgd_a <= sa.kgd_capacity,
        wkgd_b <= sb.kgd_capacity,
        wwb_a <= sa.wb_capacity,
        wwb_b <= sb.wb_capacity,
        hw_a * sa.hw_wafers_per_piece >= wwb_a,
        hw_b * sb.hw_wafers_per_piece >= wwb_b,
    ]

    total_a = wkgd_a + wwb_a
    total_b = wkgd_b + wwb_b
    total_wb = wwb_a + wwb_b

    # ── Deviation magnitudes (same 5 parameters as adaptive) ─────
    yield_dev_a = u.yield_set_width * sa.good_dies_per_wafer   # z1
    yield_dev_b = u.yield_set_width * sb.good_dies_per_wafer   # z2
    overhead_dev = u.cost_set_width * c.total_wb_overhead_per_wafer  # z3
    ds_base_dev = u.downstream_set_width * c.downstream_cost_per_die  # z4
    ds_extra_dev_a = u.downstream_set_width * sa.downstream_extra_per_wb
    ds_extra_dev_b = u.downstream_set_width * sb.downstream_extra_per_wb
    extra_rev_dev_a = u.extra_rev_set_width * sa.extra_rev_per_wb_wafer  # z5
    extra_rev_dev_b = u.extra_rev_set_width * sb.extra_rev_per_wb_wafer

    # ── Robust demand constraint (B-S over full 5D set) ──────────
    # Only z1, z2 affect demand. But the B-S dual is over all 5
    # dimensions under the shared budget. The demand constraint's
    # "sensitivity" to z3-z5 is zero, so their p variables will
    # be driven to zero by the solver. We include them for
    # correctness: the adversary could "spend" budget on z3-z5
    # even though it doesn't help (the solver handles this).
    #
    # h(z) = nominal_dies - dev_a*z1*total_a - dev_b*z2*total_b - demand >= 0
    # Coefficients of z: h1=-dev_a*total_a, h2=-dev_b*total_b, h3=h4=h5=0
    # B-S dual: h0 - Gamma*q - sum r_i >= 0,  r_i + q >= |h_i|
    z_dem = cp.Variable(nonneg=True, name="z_dem")
    p_dem = [cp.Variable(nonneg=True, name=f"p_dem_{i}") for i in range(5)]

    nom_dies = sa.good_dies_per_wafer * total_a + sb.good_dies_per_wafer * total_b
    dem_h = [
        yield_dev_a * total_a,   # |coeff of z1| (adverse is positive z -> decrease yield)
        yield_dev_b * total_b,   # |coeff of z2|
        0,                       # z3 does not affect demand
        0,                       # z4
        0,                       # z5
    ]

    dem_protection = gamma * z_dem
    for i in range(5):
        dem_protection += p_dem[i]
        # B-S requires p_i + z >= |h_i|. For non-negative h_i, this is
        # just p_i + z >= h_i. For zero h_i, trivially satisfied.
        constraints.append(p_dem[i] + z_dem >= dem_h[i])
    constraints.append(nom_dies - dem_protection >= instance.demand)

    # ── Nominal objective terms ──────────────────────────────────
    wb_cost = sa.wb_cost_per_wafer * wwb_a + sb.wb_cost_per_wafer * wwb_b
    kgd_cost = sa.kgd_cost_per_wafer * wkgd_a + sb.kgd_cost_per_wafer * wkgd_b
    wb_overhead = c.total_wb_overhead_per_wafer * total_wb
    hw_cost = c.hw_piece_cost * (hw_a + hw_b)
    downstream = (
        c.downstream_cost_per_die * instance.demand
        + sa.downstream_extra_per_wb * wwb_a
        + sb.downstream_extra_per_wb * wwb_b
    )
    total_cost = wb_cost + kgd_cost + wb_overhead + hw_cost + downstream
    extra_rev = sa.extra_rev_per_wb_wafer * wwb_a + sb.extra_rev_per_wb_wafer * wwb_b
    revenue = c.selling_price_per_die * instance.demand + extra_rev
    nominal_obj = revenue - total_cost

    # ── Robust profit guarantee (B-S over full 5D set) ───────────
    # profit(z) = nominal_obj
    #   - yield_dev_a * z1 * selling_price * total_a   (fewer dies -> less revenue)
    #   - yield_dev_b * z2 * selling_price * total_b
    #   - overhead_dev * z3 * total_wb                  (cost increase)
    #   - (ds_base_dev * z4 * demand + ds_extra_dev * z4 * wwb)  (downstream cost)
    #   - (extra_rev_dev * z5 * wwb)                    (revenue decrease)
    #
    # Note: in static model, revenue = selling_price * demand (fixed).
    # Yield does NOT directly affect static revenue since demand is fixed.
    # Yield only affects the demand CONSTRAINT (can we produce enough?).
    # So the profit sensitivity to z1,z2 is zero in the objective --
    # yield risk is handled entirely by the demand constraint.
    profit_h = [
        0,  # z1: yield doesn't affect static profit (demand is fixed)
        0,  # z2: same
        overhead_dev * total_wb,                                          # z3
        ds_base_dev * instance.demand + ds_extra_dev_a * wwb_a + ds_extra_dev_b * wwb_b,  # z4
        extra_rev_dev_a * wwb_a + extra_rev_dev_b * wwb_b,               # z5
    ]

    z_prof = cp.Variable(nonneg=True, name="z_prof")
    p_prof = [cp.Variable(nonneg=True, name=f"p_prof_{i}") for i in range(5)]

    prof_protection = gamma * z_prof
    for i in range(5):
        prof_protection += p_prof[i]
        constraints.append(p_prof[i] + z_prof >= profit_h[i])

    # ── Epigraph ─────────────────────────────────────────────────
    t = cp.Variable(name="t_profit_guarantee")
    constraints.append(nominal_obj - prof_protection >= t)
    objective = cp.Maximize(t)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS, verbose=False)

    solve_time = time.time() - t0

    if prob.status in ("optimal", "optimal_inaccurate"):
        return _build_result(
            instance,
            wkgd_a_val=int(np.round(float(wkgd_a.value))),
            wkgd_b_val=int(np.round(float(wkgd_b.value))),
            wwb_a_val=int(np.round(float(wwb_a.value))),
            wwb_b_val=int(np.round(float(wwb_b.value))),
            status=prob.status,
            obj_val=float(prob.value),
            solve_time=solve_time,
            formulation="box",
        )

    return _infeasible_result(instance, prob.status, solve_time, "box")


def solve_robust_ellipsoidal(instance: ProblemInstance, integer: bool = True) -> OptimizationResult:
    """Solve the robust optimization with ellipsoidal uncertainty set.

    For the demand constraint a'x >= d, with a_i = a_i_nom + hat_a_i * z_i:
    Robust counterpart: a_nom'x - rho * ||diag(hat_a) * x||_2 >= d

    This is a second-order cone constraint (SOCP).

    NOTE: This is an LP RELAXATION + ROUNDING approximation, not an exact
    mixed-integer SOCP solve. Free MISOCP solvers are limited; we solve
    the continuous relaxation with CLARABEL and round up to integers.
    The rounded solution is feasible but may be suboptimal relative to
    the true integer optimum.
    """
    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs
    u = instance.uncertainty
    rho = u.rho

    t0 = time.time()

    # LP relaxation -- integrality is dropped, solution rounded afterward
    wkgd_a = cp.Variable(nonneg=True, name="WKGDa")
    wkgd_b = cp.Variable(nonneg=True, name="WKGDb")
    wwb_a = cp.Variable(nonneg=True, name="WWBa")
    wwb_b = cp.Variable(nonneg=True, name="WWBb")
    hw_a = cp.Variable(nonneg=True, name="HWPcsA")
    hw_b = cp.Variable(nonneg=True, name="HWPcsB")

    constraints = [
        wkgd_a <= sa.kgd_capacity,
        wkgd_b <= sb.kgd_capacity,
        wwb_a <= sa.wb_capacity,
        wwb_b <= sb.wb_capacity,
        hw_a * sa.hw_wafers_per_piece >= wwb_a,
        hw_b * sb.hw_wafers_per_piece >= wwb_b,
    ]

    # --- Robust demand constraint (ellipsoidal) ---
    # nominal: gdpw_a * total_a + gdpw_b * total_b >= demand
    # deviation: hat_a = yield_dev_frac * gdpw
    # robust: nominal - rho * ||[hat_a_a * total_a, hat_a_b * total_b]||_2 >= demand
    yield_dev_a = u.yield_set_width * sa.good_dies_per_wafer
    yield_dev_b = u.yield_set_width * sb.good_dies_per_wafer

    total_a = wkgd_a + wwb_a
    total_b = wkgd_b + wwb_b

    deviation_vec = cp.vstack([yield_dev_a * total_a, yield_dev_b * total_b])

    constraints.append(
        sa.good_dies_per_wafer * total_a + sb.good_dies_per_wafer * total_b
        - rho * cp.norm(deviation_vec, 2) >= instance.demand
    )

    # --- Robust cost/revenue (same approach as box but with rho scaling) ---
    overhead_nom = c.total_wb_overhead_per_wafer
    overhead_dev = u.cost_set_width * overhead_nom
    robust_overhead_per_wafer = overhead_nom + rho * overhead_dev

    robust_extra_rev_a = max(0, sa.extra_rev_per_wb_wafer - rho * u.extra_rev_set_width * sa.extra_rev_per_wb_wafer)
    robust_extra_rev_b = max(0, sb.extra_rev_per_wb_wafer - rho * u.extra_rev_set_width * sb.extra_rev_per_wb_wafer)

    robust_ds_base = c.downstream_cost_per_die + rho * u.downstream_set_width * c.downstream_cost_per_die
    robust_ds_extra_a = sa.downstream_extra_per_wb + rho * u.downstream_set_width * sa.downstream_extra_per_wb
    robust_ds_extra_b = sb.downstream_extra_per_wb + rho * u.downstream_set_width * sb.downstream_extra_per_wb

    wb_cost = sa.wb_cost_per_wafer * wwb_a + sb.wb_cost_per_wafer * wwb_b
    kgd_cost = sa.kgd_cost_per_wafer * wkgd_a + sb.kgd_cost_per_wafer * wkgd_b
    wb_overhead = robust_overhead_per_wafer * (wwb_a + wwb_b)
    hw_cost = c.hw_piece_cost * (hw_a + hw_b)
    downstream = (
        robust_ds_base * instance.demand
        + robust_ds_extra_a * wwb_a
        + robust_ds_extra_b * wwb_b
    )
    total_cost = wb_cost + kgd_cost + wb_overhead + hw_cost + downstream

    extra_rev = robust_extra_rev_a * wwb_a + robust_extra_rev_b * wwb_b
    revenue = c.selling_price_per_die * instance.demand + extra_rev

    objective = cp.Maximize(revenue - total_cost)

    prob = cp.Problem(objective, constraints)
    # CLARABEL handles SOCP natively
    prob.solve(solver=cp.CLARABEL, verbose=False)

    solve_time = time.time() - t0

    if prob.status in ("optimal", "optimal_inaccurate"):
        # Round to integers for the result (continuous relaxation)
        def _round_val(v):
            return int(math.ceil(v)) if v is not None and v > 0.5 else 0
        return _build_result(
            instance,
            wkgd_a_val=_round_val(wkgd_a.value),
            wkgd_b_val=_round_val(wkgd_b.value),
            wwb_a_val=_round_val(wwb_a.value),
            wwb_b_val=_round_val(wwb_b.value),
            status=prob.status,
            obj_val=float(prob.value),
            solve_time=solve_time,
            formulation="ellipsoidal (approx)",
        )

    return _infeasible_result(instance, prob.status, solve_time, "ellipsoidal (approx)")


def solve(
    instance: ProblemInstance,
    formulation: str = "nominal",
    integer: bool = True,
) -> OptimizationResult:
    """Solve the optimization problem with the specified formulation."""
    if formulation == "nominal":
        return solve_nominal(instance, integer=integer)
    elif formulation == "box":
        return solve_robust_box(instance, integer=integer)
    elif formulation == "ellipsoidal":
        return solve_robust_ellipsoidal(instance, integer=integer)
    elif formulation == "adaptive":
        from .adaptive import solve_two_stage
        return solve_two_stage(instance)
    else:
        raise ValueError(f"Unknown formulation: {formulation}")


def solve_all_demands(
    demands: list[int] = None,
    formulation: str = "nominal",
    uncertainty: Optional[UncertaintyConfig] = None,
    integer: bool = True,
) -> dict[int, OptimizationResult]:
    """Solve for all standard demand levels and return results dict."""
    if demands is None:
        demands = DEMAND_LEVELS

    results = {}
    for d in demands:
        inst = create_default_instance(d)
        if uncertainty:
            inst.uncertainty = uncertainty
        results[d] = solve(inst, formulation=formulation, integer=integer)

    return results
