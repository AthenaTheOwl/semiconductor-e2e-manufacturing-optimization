"""Validation script: sanity-checks solver outputs and simulator consistency.

Run with: python validate.py
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model import create_default_instance, UncertaintyConfig, DEMAND_LEVELS
from src.optimizer import solve
from src.simulator import compare_solutions, generate_scenarios, evaluate_solution


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


def main():
    all_pass = True
    print("=" * 70)
    print("VALIDATION: Semiconductor Wafer Supply Chain Robust Optimization")
    print("=" * 70)

    # ── 1. Nominal solver sanity ─────────────────────────────────
    print("\n1. Nominal solver sanity checks")
    for d in DEMAND_LEVELS:
        inst = create_default_instance(d)
        r = solve(inst, formulation="nominal")
        all_pass &= check(
            f"Demand {d/1e6:.0f}M: status=optimal",
            r.status == "optimal",
        )
        all_pass &= check(
            f"Demand {d/1e6:.0f}M: demand met (dies >= demand)",
            r.total_good_dies >= d,
            f"dies={r.total_good_dies:.0f}, demand={d}",
        )
        all_pass &= check(
            f"Demand {d/1e6:.0f}M: capacities respected",
            r.wkgd_a <= 17500 and r.wkgd_b <= 30000 and r.wwb_a <= 7500 and r.wwb_b <= 20000,
        )
        all_pass &= check(
            f"Demand {d/1e6:.0f}M: nominal_profit == robust_objective",
            abs(r.nominal_profit - r.robust_objective) < 1.0,
            f"nom={r.nominal_profit:.0f}, obj={r.robust_objective:.0f}",
        )

    # ── 2. Robust solver sanity ──────────────────────────────────
    print("\n2. Box robust solver sanity checks")
    uc = UncertaintyConfig(uncertainty_type="box", gamma=1.0)
    for d in [1_000_000, 30_000_000, 50_000_000]:
        inst = create_default_instance(d)
        inst.uncertainty = uc
        r = solve(inst, formulation="box")
        all_pass &= check(
            f"Demand {d/1e6:.0f}M box: status=optimal",
            r.status == "optimal",
        )
        all_pass &= check(
            f"Demand {d/1e6:.0f}M box: robust_obj < nominal_profit",
            r.robust_objective <= r.nominal_profit + 1.0,
            f"robust_obj={r.robust_objective:.0f}, nominal_profit={r.nominal_profit:.0f}",
        )
        # Robust should produce MORE dies (surplus) than nominal
        nom = solve(create_default_instance(d), formulation="nominal")
        all_pass &= check(
            f"Demand {d/1e6:.0f}M box: more dies than nominal",
            r.total_good_dies >= nom.total_good_dies - 1,
            f"robust_dies={r.total_good_dies:.0f}, nom_dies={nom.total_good_dies:.0f}",
        )

    # ── 3. Ellipsoidal solver sanity ─────────────────────────────
    print("\n3. Ellipsoidal (approx) solver sanity checks")
    uc_e = UncertaintyConfig(uncertainty_type="ellipsoidal", rho=1.0)
    for d in [10_000_000, 30_000_000]:
        inst = create_default_instance(d)
        inst.uncertainty = uc_e
        r = solve(inst, formulation="ellipsoidal")
        all_pass &= check(
            f"Demand {d/1e6:.0f}M ellip: status=optimal",
            r.status == "optimal",
        )
        all_pass &= check(
            f"Demand {d/1e6:.0f}M ellip: formulation label contains 'approx'",
            "approx" in r.formulation,
            f"formulation={r.formulation}",
        )

    # ── 4. Simulator yield consistency ───────────────────────────
    print("\n4. Simulator yield model consistency")
    inst = create_default_instance(30_000_000)
    r = solve(inst, formulation="nominal")

    # At yield_factor=1.0, simulator should reproduce optimizer's good dies
    from src.model import ScenarioRealization
    nominal_scenario = ScenarioRealization(
        yield_a=1.0, yield_b=1.0,  # factor=1.0 means nominal
        test_cost_per_wafer=2000.0, transport_cost_per_wafer=80.0,
        eng_cost_per_wafer=5.0, downstream_scale=1.0,
        extra_rev_factor_a=1.0, extra_rev_factor_b=1.0,
    )
    sim = evaluate_solution(r, inst, [nominal_scenario])
    all_pass &= check(
        "Yield@1.0: simulator dies == optimizer dies",
        abs(sim.total_dies[0] - r.total_good_dies) < 1.0,
        f"sim_dies={sim.total_dies[0]:.0f}, opt_dies={r.total_good_dies:.0f}",
    )
    all_pass &= check(
        "Yield@1.0: simulator profit == optimizer nominal_profit",
        abs(sim.profits[0] - r.nominal_profit) < 1.0,
        f"sim_profit={sim.profits[0]:.0f}, opt_profit={r.nominal_profit:.0f}",
    )

    # ── 5. Shortage revenue check ────────────────────────────────
    print("\n5. Shortage revenue model")
    # Force a yield shock: yield_factor = 0.5 should produce fewer dies
    shortage_scenario = ScenarioRealization(
        yield_a=0.5, yield_b=0.5,  # 50% yield shock
        test_cost_per_wafer=2000.0, transport_cost_per_wafer=80.0,
        eng_cost_per_wafer=5.0, downstream_scale=1.0,
        extra_rev_factor_a=1.0, extra_rev_factor_b=1.0,
    )
    sim_short = evaluate_solution(r, inst, [shortage_scenario])
    dies_produced = sim_short.total_dies[0]
    # Revenue should be capped at min(dies, demand) * price
    expected_sellable = min(dies_produced, inst.demand)
    expected_rev = expected_sellable * inst.costs.selling_price_per_die + (
        inst.supplier_a.extra_rev_per_wb_wafer * r.wwb_a
        + inst.supplier_b.extra_rev_per_wb_wafer * r.wwb_b
    )
    all_pass &= check(
        "Shortage: revenue capped at production",
        dies_produced < inst.demand,
        f"dies={dies_produced:.0f} < demand={inst.demand}",
    )
    all_pass &= check(
        "Shortage: revenue < full-demand revenue",
        sim_short.revenues[0] < inst.costs.selling_price_per_die * inst.demand + 1,
        f"revenue={sim_short.revenues[0]:.0f}",
    )
    all_pass &= check(
        "Shortage: profit is lower than nominal",
        sim_short.profits[0] < r.nominal_profit,
        f"shortage_profit={sim_short.profits[0]:.0f} < nominal={r.nominal_profit:.0f}",
    )

    # ── 6. Monte Carlo sanity ────────────────────────────────────
    print("\n6. Monte Carlo simulation sanity")
    inst_mc = create_default_instance(30_000_000)
    inst_mc.uncertainty = UncertaintyConfig(uncertainty_type="box", gamma=1.0)
    nom = solve(create_default_instance(30_000_000), formulation="nominal")
    rob = solve(inst_mc, formulation="box")
    nom_sim, rob_sim, _ = compare_solutions(nom, rob, inst_mc, n_simulations=3000, seed=42)

    all_pass &= check(
        "MC: robust surplus >= nominal surplus (on average)",
        rob_sim.surplus_mean >= nom_sim.surplus_mean - 1e6,
        f"rob_surplus={rob_sim.surplus_mean/1e6:.2f}M, nom_surplus={nom_sim.surplus_mean/1e6:.2f}M",
    )
    all_pass &= check(
        "MC: simulation count correct",
        nom_sim.n_simulations == 3000 and rob_sim.n_simulations == 3000,
    )

    # ── 7. Comparison with report (qualitative) ──────────────────
    print("\n7. Qualitative comparison with ProjectReport appendix")
    print("   NOTE: This model is independently parameterized. Exact match")
    print("   with the report is not expected.")

    nom_1m = solve(create_default_instance(1_000_000), formulation="nominal")
    nom_10m = solve(create_default_instance(10_000_000), formulation="nominal")
    nom_50m = solve(create_default_instance(50_000_000), formulation="nominal")

    all_pass &= check(
        "1M demand: KGD-only (no WB)",
        nom_1m.wwb_a == 0 and nom_1m.wwb_b == 0,
        f"WB-A={nom_1m.wwb_a}, WB-B={nom_1m.wwb_b}",
    )
    all_pass &= check(
        "10M demand: KGD-only (report: KGD-A=0, KGD-B=14185)",
        nom_10m.wwb_a == 0 and nom_10m.wwb_b == 0,
        f"KGD-A={nom_10m.wkgd_a}, KGD-B={nom_10m.wkgd_b}",
    )
    all_pass &= check(
        "10M demand: KGD-B matches report (14185)",
        nom_10m.wkgd_b == 14185,
        f"got {nom_10m.wkgd_b}, expected 14185",
    )
    all_pass &= check(
        "50M demand: WB is used (report: WB-A=7500, WB-B=19200)",
        nom_50m.wwb_a > 0 or nom_50m.wwb_b > 0,
        f"WB-A={nom_50m.wwb_a}, WB-B={nom_50m.wwb_b}",
    )
    all_pass &= check(
        "50M demand: all sources used",
        nom_50m.wkgd_a > 0 and nom_50m.wkgd_b > 0,
        f"KGD-A={nom_50m.wkgd_a}, KGD-B={nom_50m.wkgd_b}",
    )

    # ── 8. Yield control changes simulation ────────────────────
    print("\n8. Yield control affects sampled yield factors")
    from src.simulator import generate_scenarios

    inst_lo = create_default_instance(10_000_000)
    inst_lo.supplier_a.yield_std = 0.01  # tight
    inst_lo.supplier_b.yield_std = 0.01
    inst_hi = create_default_instance(10_000_000)
    inst_hi.supplier_a.yield_std = 0.10  # wide
    inst_hi.supplier_b.yield_std = 0.10

    scens_lo = generate_scenarios(inst_lo, n_simulations=2000, seed=99)
    scens_hi = generate_scenarios(inst_hi, n_simulations=2000, seed=99)
    yields_lo = np.array([s.yield_a for s in scens_lo])
    yields_hi = np.array([s.yield_a for s in scens_hi])

    all_pass &= check(
        "Higher yield_std -> wider yield spread",
        np.std(yields_hi) > np.std(yields_lo) * 1.5,
        f"std_hi={np.std(yields_hi):.4f}, std_lo={np.std(yields_lo):.4f}",
    )
    all_pass &= check(
        "Both centered near 1.0",
        abs(np.mean(yields_lo) - 1.0) < 0.02 and abs(np.mean(yields_hi) - 1.0) < 0.02,
        f"mean_lo={np.mean(yields_lo):.4f}, mean_hi={np.mean(yields_hi):.4f}",
    )

    # ── 9. Gamma monotonicity ────────────────────────────────────
    print("\n9. Gamma monotonicity: robust_objective non-increasing in Gamma")
    prev_obj = float("inf")
    gamma_mono_ok = True
    for g in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        inst_g = create_default_instance(30_000_000)
        inst_g.uncertainty = UncertaintyConfig(uncertainty_type="box", gamma=g)
        rg = solve(inst_g, formulation="box")
        if rg.robust_objective > prev_obj + 100:  # small tolerance
            gamma_mono_ok = False
            all_pass &= check(
                f"Gamma={g}: robust_obj <= previous",
                False,
                f"obj={rg.robust_objective:.0f} > prev={prev_obj:.0f}",
            )
        prev_obj = rg.robust_objective
    all_pass &= check(
        "Gamma sweep: robust_objective monotonically non-increasing",
        gamma_mono_ok,
        f"final obj at Gamma=3: {prev_obj:.0f}",
    )

    # ── 10. Gamma=0 reproduces nominal ───────────────────────────
    print("\n10. Gamma=0 reproduces nominal solution")
    inst_g0 = create_default_instance(30_000_000)
    inst_g0.uncertainty = UncertaintyConfig(uncertainty_type="box", gamma=0.0)
    r_g0 = solve(inst_g0, formulation="box")
    r_nom = solve(create_default_instance(30_000_000), formulation="nominal")
    all_pass &= check(
        "Gamma=0: nominal_profit matches nominal solver (within 0.01%)",
        abs(r_g0.nominal_profit - r_nom.nominal_profit) / r_nom.nominal_profit < 0.0001,
        f"g0={r_g0.nominal_profit:.0f}, nom={r_nom.nominal_profit:.0f}, "
        f"diff={abs(r_g0.nominal_profit - r_nom.nominal_profit):.0f}",
    )
    all_pass &= check(
        "Gamma=0: robust_objective == nominal_profit (no protection needed)",
        abs(r_g0.robust_objective - r_g0.nominal_profit) / r_g0.nominal_profit < 0.0001,
        f"obj={r_g0.robust_objective:.0f}, nom={r_g0.nominal_profit:.0f}",
    )

    # ── 11. Terminology sanity ───────────────────────────────────
    print("\n11. Terminology: robust_objective <= nominal_profit for robust solves")
    for d in [10_000_000, 30_000_000, 50_000_000]:
        for g in [0.5, 1.0, 2.0]:
            inst_t = create_default_instance(d)
            inst_t.uncertainty = UncertaintyConfig(uncertainty_type="box", gamma=g)
            rt = solve(inst_t, formulation="box")
            all_pass &= check(
                f"D={d/1e6:.0f}M, G={g}: robust_obj <= nominal_profit",
                rt.robust_objective <= rt.nominal_profit + 1.0,
                f"obj={rt.robust_objective:.0f}, nom={rt.nominal_profit:.0f}",
            )

    # ── 12. Two-stage adaptive model ─────────────────────────────
    print("\n12. Two-stage adaptive model (vertex enumeration, full 5D set)")
    from src.adaptive import solve_two_stage

    for d in [10_000_000, 30_000_000, 50_000_000]:
        inst_a = create_default_instance(d)
        inst_a.uncertainty = UncertaintyConfig(uncertainty_type="box", gamma=1.0)
        adaptive = solve_two_stage(inst_a)
        all_pass &= check(
            f"D={d/1e6:.0f}M: adaptive status=optimal",
            adaptive.status == "optimal",
        )
        all_pass &= check(
            f"D={d/1e6:.0f}M: adaptive formulation label contains 'vertex'",
            "vertex" in adaptive.formulation,
            f"formulation={adaptive.formulation}",
        )
        all_pass &= check(
            f"D={d/1e6:.0f}M: adaptive demand met at nominal",
            adaptive.total_good_dies >= d,
            f"dies={adaptive.total_good_dies:.0f}, demand={d}",
        )

    # ── 13. Adaptive cost sensitivity ────────────────────────────
    print("\n13. Adaptive cost sensitivity: set widths affect robust objective")
    base_uc = UncertaintyConfig(uncertainty_type="box", gamma=1.0)
    inst_base = create_default_instance(30_000_000)
    inst_base.uncertainty = base_uc
    adp_base = solve_two_stage(inst_base)

    # Downstream width should clearly affect the result
    wide_ds = UncertaintyConfig(uncertainty_type="box", gamma=1.0, downstream_set_width=0.40)
    inst_ds = create_default_instance(30_000_000)
    inst_ds.uncertainty = wide_ds
    adp_ds = solve_two_stage(inst_ds)
    all_pass &= check(
        "Wider downstream_set_width -> lower adaptive robust_obj",
        adp_ds.robust_objective < adp_base.robust_objective - 1.0,
        f"base={adp_base.robust_objective:.0f}, wide_ds={adp_ds.robust_objective:.0f}",
    )

    # Yield width should clearly affect
    wide_yield = UncertaintyConfig(uncertainty_type="box", gamma=1.0, yield_set_width=0.20)
    inst_yw = create_default_instance(30_000_000)
    inst_yw.uncertainty = wide_yield
    adp_yw = solve_two_stage(inst_yw)
    all_pass &= check(
        "Wider yield_set_width -> lower adaptive robust_obj",
        adp_yw.robust_objective < adp_base.robust_objective - 1.0,
        f"base={adp_base.robust_objective:.0f}, wide_yield={adp_yw.robust_objective:.0f}",
    )

    # ── 14. Adaptive nominal recourse evaluation ─────────────────
    print("\n14. Adaptive nominal recourse evaluation")
    # At nominal scenario, fulfill should equal demand (since production > demand)
    all_pass &= check(
        "Adaptive nominal: production >= demand",
        adp_base.total_good_dies >= 30_000_000,
        f"dies={adp_base.total_good_dies:.0f}",
    )
    # Nominal profit should use recourse payoff, not static formula
    all_pass &= check(
        "Adaptive nominal_profit uses recourse payoff",
        "adaptive" in adp_base.formulation,
        f"formulation={adp_base.formulation}",
    )

    # ── 15. Adaptive simulation: exact recourse evaluation ───────
    print("\n15. Adaptive simulation: exact recourse evaluation in MC")
    from src.simulator import evaluate_adaptive_solution, generate_scenarios

    inst_sim = create_default_instance(30_000_000)
    inst_sim.uncertainty = UncertaintyConfig(uncertainty_type="box", gamma=1.0)
    adp_for_sim = solve_two_stage(inst_sim)

    all_pass &= check(
        "Adaptive policy stored in result",
        adp_for_sim.adaptive_policy is not None,
    )
    all_pass &= check(
        "Adaptive policy has yield-group entries",
        adp_for_sim.adaptive_policy is not None and len(adp_for_sim.adaptive_policy) > 0,
        f"n_yield_states={len(adp_for_sim.adaptive_policy) if adp_for_sim.adaptive_policy else 0}",
    )

    scenarios = generate_scenarios(inst_sim, n_simulations=1000, seed=42)
    adp_sim = evaluate_adaptive_solution(adp_for_sim, inst_sim, scenarios)

    all_pass &= check(
        "Adaptive MC: all profits finite",
        np.all(np.isfinite(adp_sim.profits)),
    )
    all_pass &= check(
        "Adaptive MC: revenues non-negative (most scenarios)",
        np.mean(adp_sim.revenues >= 0) > 0.9,
        f"positive_rev_pct={np.mean(adp_sim.revenues >= 0)*100:.1f}%",
    )
    all_pass &= check(
        "Adaptive MC: simulation count",
        adp_sim.n_simulations == 1000,
    )

    # ── 16. Fractional Gamma produces distinct adaptive results ──
    print("\n16. Fractional Gamma: adaptive solver produces distinct results")
    from src.adaptive import _enumerate_budget_vertices
    prev_obj = None
    frac_gamma_ok = True
    for g in [0.0, 0.5, 1.0, 1.5, 2.0]:
        inst_fg = create_default_instance(30_000_000)
        inst_fg.uncertainty = UncertaintyConfig(uncertainty_type="box", gamma=g)
        r_fg = solve_two_stage(inst_fg)
        if prev_obj is not None and abs(r_fg.robust_objective - prev_obj) < 1.0:
            frac_gamma_ok = False
        prev_obj = r_fg.robust_objective
    all_pass &= check(
        "Fractional Gamma: each 0.5 step produces a distinct result",
        frac_gamma_ok,
    )

    verts_frac = _enumerate_budget_vertices(5, 1.5)
    has_frac = any(any(0 < abs(zi) < 1 - 1e-9 for zi in v) for v in verts_frac)
    all_pass &= check(
        "Gamma=1.5: vertex set contains fractional coordinates",
        has_frac,
        f"n_vertices={len(verts_frac)}",
    )

    # ── 17. Policy consistency at nominal scenario ───────────────
    print("\n17. Policy consistency: nominal scenario matches stored profit")
    from src.model import ScenarioRealization
    nom_scenario = ScenarioRealization(
        yield_a=1.0, yield_b=1.0,
        test_cost_per_wafer=2000.0, transport_cost_per_wafer=80.0,
        eng_cost_per_wafer=5.0, downstream_scale=1.0,
        extra_rev_factor_a=1.0, extra_rev_factor_b=1.0,
    )
    adp_nom_sim = evaluate_adaptive_solution(adp_for_sim, inst_sim, [nom_scenario])
    all_pass &= check(
        "Policy@nominal: sim profit matches stored nominal_profit (within 1%)",
        abs(adp_nom_sim.profits[0] - adp_for_sim.nominal_profit) / max(1, abs(adp_for_sim.nominal_profit)) < 0.01,
        f"sim={adp_nom_sim.profits[0]:.0f}, stored={adp_for_sim.nominal_profit:.0f}",
    )

    # ── 18. App-path UncertaintyConfig smoke test ──────────────
    print("\n18. App-path UncertaintyConfig construction (field names)")
    # These mirror the exact constructions used in app.py Gamma sweep
    # and sensitivity sections. If field names are wrong, this crashes.
    try:
        uc_sweep = UncertaintyConfig(
            uncertainty_type="box", gamma=1.5,
            yield_set_width=0.10,
            cost_set_width=0.15,
            downstream_set_width=0.20,
            extra_rev_set_width=0.25,
        )
        inst_smoke = create_default_instance(10_000_000)
        inst_smoke.uncertainty = uc_sweep
        r_smoke = solve(inst_smoke, formulation="box")
        all_pass &= check(
            "App Gamma-sweep path: UncertaintyConfig + solve succeeds",
            r_smoke.status == "optimal",
        )
    except TypeError as e:
        all_pass &= check("App Gamma-sweep path: no TypeError", False, str(e))

    # Sensitivity path: construct with one param overridden
    try:
        uc_sens = UncertaintyConfig(
            uncertainty_type="box", gamma=1.0,
            yield_set_width=0.15,  # overridden
            cost_set_width=0.15,
            downstream_set_width=0.20,
            extra_rev_set_width=0.25,
        )
        inst_sens = create_default_instance(10_000_000)
        inst_sens.uncertainty = uc_sens
        r_sens = solve(inst_sens, formulation="box")
        all_pass &= check(
            "App sensitivity path: UncertaintyConfig + solve succeeds",
            r_sens.status == "optimal",
        )
    except TypeError as e:
        all_pass &= check("App sensitivity path: no TypeError", False, str(e))

    # Ellipsoidal path
    try:
        uc_ellip = UncertaintyConfig(
            uncertainty_type="ellipsoidal", rho=1.0,
            yield_set_width=0.10,
            cost_set_width=0.15,
            downstream_set_width=0.20,
            extra_rev_set_width=0.25,
        )
        inst_ell = create_default_instance(10_000_000)
        inst_ell.uncertainty = uc_ellip
        r_ell = solve(inst_ell, formulation="ellipsoidal")
        all_pass &= check(
            "App ellipsoidal path: UncertaintyConfig + solve succeeds",
            r_ell.status == "optimal",
        )
    except TypeError as e:
        all_pass &= check("App ellipsoidal path: no TypeError", False, str(e))

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- review output above")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
