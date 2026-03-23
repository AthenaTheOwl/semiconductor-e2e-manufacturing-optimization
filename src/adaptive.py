"""Two-stage adaptive robust optimization for the wafer supply chain.

Uses vertex/scenario enumeration over the full 5D Bertsimas-Sim budget set.

Information structure:
  Stage 1 (here-and-now): commit KGD wafers, WB wafers, and hardware.
  Uncertainty revealed: yield (z1, z2) is OBSERVED. Cost/revenue (z3, z4, z5)
    are NOT observed -- they are adversarial.
  Stage 2 (recourse, after observing yield only):
    - fulfill: dies shipped to OEM
    - salvage: surplus dies sold at salvage price
    - lost_sales: demand - fulfill, penalized

Nonanticipativity: recourse decisions depend only on the observed
yield state (z1, z2), not on cost/revenue (z3, z4, z5). Vertices that
share the same (z1, z2) must use the same recourse actions. The profit
guarantee holds for ALL (z3, z4, z5) realizations at each yield state.

Uncertainty parameters (same B-S family as static robust):
  z1: yield_a          (DECREASE is adverse, OBSERVED before recourse)
  z2: yield_b          (DECREASE is adverse, OBSERVED before recourse)
  z3: WB overhead cost (INCREASE is adverse, NOT observed)
  z4: downstream cost  (INCREASE is adverse, NOT observed)
  z5: extra revenue    (DECREASE is adverse, NOT observed)

Budget set: Z = { z in R^5 : |z_i| <= 1, sum |z_i| <= Gamma }
"""

import time
import math
import itertools
import numpy as np
import cvxpy as cp

from .model import (
    ProblemInstance, OptimizationResult, UncertaintyConfig,
)


def _enumerate_budget_vertices(n_dims: int, gamma: float) -> list[tuple[float, ...]]:
    """Enumerate extreme points of the B-S budget set.

    Z = { z in R^n : |z_i| <= 1, sum |z_i| <= Gamma }

    This is the intersection of the L1 ball {sum|z_i| <= Gamma} and
    the L-inf ball {|z_i| <= 1}. Its extreme points are:

    Let f = floor(Gamma), r = Gamma - f (fractional part).

    - For k <= f: choose k coordinates, set each to ±1 (rest 0).
      Budget sum = k <= f <= Gamma. ✓

    - For k = f + 1 (only when r > 0): choose f+1 coordinates,
      set f of them to ±1 and one to ±r. Budget sum = f + r = Gamma. ✓

    This is exact for all Gamma >= 0 (integer or fractional).
    """
    if gamma <= 0:
        return [(0.0,) * n_dims]

    f = int(math.floor(gamma))
    r = gamma - f  # fractional part, 0 <= r < 1

    vertices = set()
    vertices.add((0.0,) * n_dims)

    # Integer-valued vertices: k coordinates at ±1, k <= min(f, n_dims)
    for k in range(1, min(f, n_dims) + 1):
        for indices in itertools.combinations(range(n_dims), k):
            for signs in itertools.product([-1.0, 1.0], repeat=k):
                z = [0.0] * n_dims
                for idx, sgn in zip(indices, signs):
                    z[idx] = sgn
                vertices.add(tuple(z))

    # Fractional vertices: f+1 coordinates active, f at ±1, one at ±r
    # Only needed when r > 1e-12 and f+1 <= n_dims
    if r > 1e-12 and f + 1 <= n_dims:
        for indices in itertools.combinations(range(n_dims), f + 1):
            # Choose which of the f+1 indices gets the fractional value
            for frac_pos in range(f + 1):
                full_indices = [indices[j] for j in range(f + 1) if j != frac_pos]
                frac_idx = indices[frac_pos]
                # Signs for the f full-valued coordinates
                for full_signs in itertools.product([-1.0, 1.0], repeat=f):
                    for frac_sign in [-1.0, 1.0]:
                        z = [0.0] * n_dims
                        for idx, sgn in zip(full_indices, full_signs):
                            z[idx] = sgn
                        z[frac_idx] = frac_sign * r
                        vertices.add(tuple(z))

    return list(vertices)


def solve_two_stage(
    instance: ProblemInstance,
    salvage_price_frac: float = 0.30,
    lost_sales_penalty_frac: float = 1.50,
) -> OptimizationResult:
    """Solve the two-stage adaptive robust optimization.

    Uses vertex enumeration over the full 5D Bertsimas-Sim budget set.
    Stage-2 recourse variables are created per observed production level
    (grouped by yield vertices z1,z2). Cost/revenue coefficients use the
    full vertex realization (z3,z4,z5) -- no bilinearity shortcuts.

    Parameters
    ----------
    salvage_price_frac : float
        Salvage price as fraction of selling price.
    lost_sales_penalty_frac : float
        Lost-sales penalty as fraction of selling price.
    """
    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs
    u = instance.uncertainty
    gamma = u.gamma

    salvage_price = salvage_price_frac * c.selling_price_per_die
    lost_sales_penalty = lost_sales_penalty_frac * c.selling_price_per_die

    t0 = time.time()

    # ── Enumerate vertices of Z (5D budget set) ─────────────────
    vertices = _enumerate_budget_vertices(5, gamma)
    # Each vertex is (z1_yield_a, z2_yield_b, z3_overhead, z4_downstream, z5_extra_rev)

    # ── Stage 1 variables (here-and-now, integer) ────────────────
    wkgd_a = cp.Variable(integer=True, name="WKGDa")
    wkgd_b = cp.Variable(integer=True, name="WKGDb")
    wwb_a = cp.Variable(integer=True, name="WWBa")
    wwb_b = cp.Variable(integer=True, name="WWBb")
    hw_a = cp.Variable(integer=True, name="HWPcsA")
    hw_b = cp.Variable(integer=True, name="HWPcsB")

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

    # ── Stage 1 costs (deterministic, committed) ─────────────────
    wb_cost = sa.wb_cost_per_wafer * wwb_a + sb.wb_cost_per_wafer * wwb_b
    kgd_cost = sa.kgd_cost_per_wafer * wkgd_a + sb.kgd_cost_per_wafer * wkgd_b
    nominal_wb_overhead = c.total_wb_overhead_per_wafer * (wwb_a + wwb_b)
    hw_cost = c.hw_piece_cost * (hw_a + hw_b)

    # ── Deviation magnitudes ─────────────────────────────────────
    yield_dev_a = u.yield_set_width * sa.good_dies_per_wafer  # dies/wafer
    yield_dev_b = u.yield_set_width * sb.good_dies_per_wafer
    overhead_dev = u.cost_set_width * c.total_wb_overhead_per_wafer  # $/wafer
    ds_base_dev = u.downstream_set_width * c.downstream_cost_per_die  # $/die
    ds_extra_dev_a = u.downstream_set_width * sa.downstream_extra_per_wb
    ds_extra_dev_b = u.downstream_set_width * sb.downstream_extra_per_wb
    extra_rev_dev_a = u.extra_rev_set_width * sa.extra_rev_per_wb_wafer
    extra_rev_dev_b = u.extra_rev_set_width * sb.extra_rev_per_wb_wafer

    # ── Group vertices by yield state (z1, z2) for nonanticipativity ─
    # Recourse depends on production, which depends only on (z1, z2).
    # For each unique (z1, z2) group, we create ONE set of recourse vars.
    yield_groups: dict[tuple[float, float], list[int]] = {}
    for i, v in enumerate(vertices):
        key = (v[0], v[1])
        yield_groups.setdefault(key, []).append(i)

    # ── Stage 2 recourse variables per yield group ───────────────
    # fulfill_g, salvage_g for each yield group g
    fulfill_vars = {}  # (z1,z2) -> cp.Variable
    salvage_vars = {}

    for ykey in yield_groups:
        fulfill_vars[ykey] = cp.Variable(nonneg=True, name=f"fulfill_{ykey}")
        salvage_vars[ykey] = cp.Variable(nonneg=True, name=f"salvage_{ykey}")

    # ── Per yield-group: production feasibility ──────────────────
    for ykey in yield_groups:
        z1, z2 = ykey
        # Realized production at this yield state
        prod = ((sa.good_dies_per_wafer + yield_dev_a * z1) * total_a
                + (sb.good_dies_per_wafer + yield_dev_b * z2) * total_b)

        ful = fulfill_vars[ykey]
        sal = salvage_vars[ykey]

        constraints += [
            ful + sal <= prod,           # can't ship + salvage more than produced
            ful <= instance.demand,      # can't ship more than demand
        ]

    # ── Epigraph variable ────────────────────────────────────────
    t = cp.Variable(name="t_adaptive")

    # ── Per vertex: profit lower bound using full realization ─────
    for i, v in enumerate(vertices):
        z1, z2, z3, z4, z5 = v
        ykey = (z1, z2)
        ful = fulfill_vars[ykey]
        sal = salvage_vars[ykey]
        lost = instance.demand - ful

        # Realized WB overhead: nominal + deviation * z3
        realized_overhead = (c.total_wb_overhead_per_wafer + overhead_dev * z3) * (wwb_a + wwb_b)

        # Realized downstream cost: (nominal + dev*z4) applied per die fulfilled
        realized_ds_per_die = c.downstream_cost_per_die + ds_base_dev * z4
        realized_ds_extra_a = sa.downstream_extra_per_wb + ds_extra_dev_a * z4
        realized_ds_extra_b = sb.downstream_extra_per_wb + ds_extra_dev_b * z4
        realized_downstream = (realized_ds_per_die * ful
                               + realized_ds_extra_a * wwb_a
                               + realized_ds_extra_b * wwb_b)

        # Realized extra revenue: (nominal + dev*z5)*wwb  (z5 negative = decrease)
        realized_extra_rev = ((sa.extra_rev_per_wb_wafer - extra_rev_dev_a * z5) * wwb_a
                              + (sb.extra_rev_per_wb_wafer - extra_rev_dev_b * z5) * wwb_b)

        # Total cost at this vertex
        total_cost_v = (wb_cost + kgd_cost + realized_overhead + hw_cost
                        + realized_downstream)

        # Revenue at this vertex
        revenue_v = (c.selling_price_per_die * ful
                     + salvage_price * sal
                     + realized_extra_rev
                     - lost_sales_penalty * lost)

        profit_v = revenue_v - total_cost_v

        constraints.append(profit_v >= t)

    objective = cp.Maximize(t)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS, verbose=False)

    solve_time = time.time() - t0

    if prob.status in ("optimal", "optimal_inaccurate"):
        wkgd_a_val = int(np.round(float(wkgd_a.value)))
        wkgd_b_val = int(np.round(float(wkgd_b.value)))
        wwb_a_val = int(np.round(float(wwb_a.value)))
        wwb_b_val = int(np.round(float(wwb_b.value)))

        # ── Extract solved policy ────────────────────────────────
        # Map yield state (z1, z2) -> (fulfill, salvage) from solved vars
        policy = {}
        for ykey in yield_groups:
            ful_val = float(fulfill_vars[ykey].value)
            sal_val = float(salvage_vars[ykey].value)
            policy[ykey] = (max(0.0, ful_val), max(0.0, sal_val))

        # ── Nominal recourse evaluation ──────────────────────────
        # Evaluate the policy at z = (0,0,0,0,0) = nominal scenario
        nominal_ykey = (0.0, 0.0)
        if nominal_ykey in fulfill_vars:
            nom_fulfill = float(fulfill_vars[nominal_ykey].value)
            nom_salvage = float(salvage_vars[nominal_ykey].value)
        else:
            # Nominal might not be a vertex; compute from production
            nom_dies = (sa.good_dies_per_wafer * (wkgd_a_val + wwb_a_val)
                        + sb.good_dies_per_wafer * (wkgd_b_val + wwb_b_val))
            nom_fulfill = min(nom_dies, instance.demand)
            nom_salvage = max(nom_dies - nom_fulfill, 0)

        nom_lost = instance.demand - nom_fulfill

        # Nominal costs
        wb_c = sa.wb_cost_per_wafer * wwb_a_val + sb.wb_cost_per_wafer * wwb_b_val
        kgd_c = sa.kgd_cost_per_wafer * wkgd_a_val + sb.kgd_cost_per_wafer * wkgd_b_val
        wb_oh = c.total_wb_overhead_per_wafer * (wwb_a_val + wwb_b_val)
        hw_c = c.hw_piece_cost * (
            (math.ceil(wwb_a_val / sa.hw_wafers_per_piece) if wwb_a_val > 0 else 0)
            + (math.ceil(wwb_b_val / sb.hw_wafers_per_piece) if wwb_b_val > 0 else 0)
        )
        ds_c = (c.downstream_cost_per_die * nom_fulfill
                + sa.downstream_extra_per_wb * wwb_a_val
                + sb.downstream_extra_per_wb * wwb_b_val)
        total_cost_nom = wb_c + kgd_c + wb_oh + hw_c + ds_c

        extra_rev_nom = (sa.extra_rev_per_wb_wafer * wwb_a_val
                         + sb.extra_rev_per_wb_wafer * wwb_b_val)
        revenue_nom = (c.selling_price_per_die * nom_fulfill
                       + salvage_price * nom_salvage
                       + extra_rev_nom
                       - lost_sales_penalty * nom_lost)
        nominal_profit = revenue_nom - total_cost_nom

        total_good_dies = (sa.good_dies_per_wafer * (wkgd_a_val + wwb_a_val)
                           + sb.good_dies_per_wafer * (wkgd_b_val + wwb_b_val))

        hw_pcs_a = math.ceil(wwb_a_val / sa.hw_wafers_per_piece) if wwb_a_val > 0 else 0
        hw_pcs_b = math.ceil(wwb_b_val / sb.hw_wafers_per_piece) if wwb_b_val > 0 else 0

        return OptimizationResult(
            status=prob.status,
            robust_objective=float(prob.value),
            wkgd_a=wkgd_a_val, wkgd_b=wkgd_b_val,
            wwb_a=wwb_a_val, wwb_b=wwb_b_val,
            hw_pcs_a=hw_pcs_a, hw_pcs_b=hw_pcs_b,
            wb_cost=wb_c, kgd_cost=kgd_c, wb_overhead=wb_oh,
            hw_cost=hw_c, downstream_cost=ds_c,
            total_cost=total_cost_nom, revenue=revenue_nom,
            nominal_profit=nominal_profit,
            total_good_dies=total_good_dies,
            demand=instance.demand,
            demand_surplus=total_good_dies - instance.demand,
            demand_ratio=total_good_dies / instance.demand if instance.demand > 0 else 0,
            solve_time=solve_time,
            formulation="two-stage adaptive (vertex)",
            adaptive_policy=policy,
        )

    return OptimizationResult(
        status=prob.status, robust_objective=0,
        wkgd_a=0, wkgd_b=0, wwb_a=0, wwb_b=0,
        hw_pcs_a=0, hw_pcs_b=0,
        wb_cost=0, kgd_cost=0, wb_overhead=0, hw_cost=0,
        downstream_cost=0, total_cost=0, revenue=0, nominal_profit=0,
        total_good_dies=0, demand=instance.demand,
        demand_surplus=-instance.demand, demand_ratio=0,
        solve_time=solve_time, formulation="two-stage adaptive (vertex)",
    )
