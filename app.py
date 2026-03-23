"""Semiconductor Wafer Supply Chain Robust Optimization Dashboard.

Interactive Streamlit application demonstrating how robust optimization
protects against uncertainty in a fabless semiconductor supply chain.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.model import (
    create_default_instance, create_default_suppliers,
    UncertaintyConfig, DEMAND_LEVELS,
    ProblemInstance, SupplierParams, CostParams,
)
from src.optimizer import solve, solve_all_demands
from src.simulator import compare_solutions
from src.visualization import (
    plot_supply_chain_diagram, plot_wafer_allocation,
    plot_demand_sweep, plot_cost_breakdown,
    plot_nominal_vs_robust, plot_uncertainty_sets_2d,
    plot_monte_carlo_histograms, plot_demand_met_comparison,
    plot_cost_of_robustness, plot_sensitivity_tornado,
)

st.set_page_config(
    page_title="Semiconductor Wafer Supply Chain - Robust Optimization",
    page_icon="🔬",
    layout="wide",
)


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("Configuration")

demand_m = st.sidebar.selectbox(
    "Demand Level (Millions of Dies)",
    [1, 5, 10, 20, 30, 40, 50],
    index=4,
)
demand = demand_m * 1_000_000

# ── Supplier Parameters ──
defaults_a, defaults_b = create_default_suppliers()
defaults_c = CostParams()

with st.sidebar.expander("Supplier A (Premium)", expanded=False):
    a_kgd_cap = st.number_input("KGD Capacity (wafers)", value=defaults_a.kgd_capacity, step=500, key="a_kgd_cap")
    a_wb_cap = st.number_input("WB Capacity (wafers)", value=defaults_a.wb_capacity, step=500, key="a_wb_cap")
    a_kgd_cost = st.number_input("KGD Cost ($/wafer)", value=defaults_a.kgd_cost_per_wafer, step=500.0, key="a_kgd_cost")
    a_wb_cost = st.number_input("WB Cost ($/wafer)", value=defaults_a.wb_cost_per_wafer, step=500.0, key="a_wb_cost")
    a_gdpw = st.number_input("Good Dies/Wafer", value=defaults_a.good_dies_per_wafer, step=10, key="a_gdpw")
    a_yield = st.slider("Yield Mean", 0.50, 1.00, defaults_a.yield_mean, 0.01, key="a_yield")
    a_yield_std = st.slider("Yield Std Dev", 0.001, 0.100, defaults_a.yield_std, 0.005, key="a_yield_std")
    a_hw_tput = st.number_input("HW Wafers/Piece", value=defaults_a.hw_wafers_per_piece, step=100, key="a_hw_tput")
    a_extra_rev = st.number_input("Extra Rev per WB Wafer ($)", value=defaults_a.extra_rev_per_wb_wafer, step=50.0, key="a_extra_rev")
    a_ds_extra = st.number_input("Downstream Extra per WB ($)", value=defaults_a.downstream_extra_per_wb, step=10.0, key="a_ds_extra")

with st.sidebar.expander("Supplier B (Economy)", expanded=False):
    b_kgd_cap = st.number_input("KGD Capacity (wafers)", value=defaults_b.kgd_capacity, step=500, key="b_kgd_cap")
    b_wb_cap = st.number_input("WB Capacity (wafers)", value=defaults_b.wb_capacity, step=500, key="b_wb_cap")
    b_kgd_cost = st.number_input("KGD Cost ($/wafer)", value=defaults_b.kgd_cost_per_wafer, step=500.0, key="b_kgd_cost")
    b_wb_cost = st.number_input("WB Cost ($/wafer)", value=defaults_b.wb_cost_per_wafer, step=500.0, key="b_wb_cost")
    b_gdpw = st.number_input("Good Dies/Wafer", value=defaults_b.good_dies_per_wafer, step=10, key="b_gdpw")
    b_yield = st.slider("Yield Mean", 0.50, 1.00, defaults_b.yield_mean, 0.01, key="b_yield")
    b_yield_std = st.slider("Yield Std Dev", 0.001, 0.100, defaults_b.yield_std, 0.005, key="b_yield_std")
    b_hw_tput = st.number_input("HW Wafers/Piece", value=defaults_b.hw_wafers_per_piece, step=100, key="b_hw_tput")
    b_extra_rev = st.number_input("Extra Rev per WB Wafer ($)", value=defaults_b.extra_rev_per_wb_wafer, step=50.0, key="b_extra_rev")
    b_ds_extra = st.number_input("Downstream Extra per WB ($)", value=defaults_b.downstream_extra_per_wb, step=10.0, key="b_ds_extra")

with st.sidebar.expander("Fixed Costs", expanded=False):
    test_cost = st.number_input("Test Cost per WB Wafer ($)", value=defaults_c.test_cost_per_wafer, step=100.0, key="test_cost")
    transport_cost = st.number_input("Transport Cost per WB Wafer ($)", value=defaults_c.transport_cost_per_wafer, step=10.0, key="transport_cost")
    eng_cost = st.number_input("Engineering Cost per WB Wafer ($)", value=defaults_c.eng_cost_per_wafer, step=1.0, key="eng_cost")
    hw_piece_cost = st.number_input("HW Piece Cost ($)", value=defaults_c.hw_piece_cost, step=10000.0, key="hw_piece_cost")
    selling_price = st.number_input("Selling Price per Die ($)", value=defaults_c.selling_price_per_die, step=1.0, key="sell_price")
    ds_cost_per_die = st.number_input("Downstream Cost per Die ($)", value=defaults_c.downstream_cost_per_die, step=0.5, key="ds_cost")

st.sidebar.markdown("---")
st.sidebar.subheader("Uncertainty Parameters")

uncertainty_type = st.sidebar.radio(
    "Uncertainty Set Type",
    ["Nominal", "Box (Budget)", "Ellipsoidal"],
    index=1,
)

if uncertainty_type == "Box (Budget)":
    gamma = st.sidebar.slider("Gamma (Budget Parameter)", 0.0, 3.0, 1.0, 0.1)
    rho = 1.0
    unc_type_key = "box"
elif uncertainty_type == "Ellipsoidal":
    gamma = 1.0
    rho = st.sidebar.slider("Rho (Ellipsoid Radius)", 0.0, 3.0, 1.0, 0.1)
    unc_type_key = "ellipsoidal"
else:
    gamma = 0.0
    rho = 0.0
    unc_type_key = "nominal"

st.sidebar.caption("Robust set half-widths (planner conservatism). "
                    "Simulation yield spread uses supplier Yield Std above.")
yield_dev = st.sidebar.slider("Yield Set Width (%)", 1, 25, 10, 1, help="Robust planner's yield uncertainty half-width") / 100
cost_dev = st.sidebar.slider("Cost Set Width (%)", 1, 30, 15, 1, help="Also used as simulation cost spread") / 100
downstream_dev = st.sidebar.slider("Downstream Set Width (%)", 1, 40, 20, 1) / 100
extra_rev_dev = st.sidebar.slider("Revenue Set Width (%)", 1, 50, 25, 1) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation")
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 500, 20000, 5000, 500)
mc_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)


# ── Build instance from sidebar values ───────────────────────────────
def build_instance(d: int) -> ProblemInstance:
    """Build a ProblemInstance from current sidebar parameter values."""
    supplier_a = SupplierParams(
        name="A", kgd_capacity=a_kgd_cap, wb_capacity=a_wb_cap,
        wb_cost_per_wafer=a_wb_cost, kgd_cost_per_wafer=a_kgd_cost,
        good_dies_per_wafer=a_gdpw, yield_mean=a_yield, yield_std=a_yield_std,
        dies_per_wafer=int(a_gdpw / a_yield) if a_yield > 0 else a_gdpw,
        hw_wafers_per_piece=a_hw_tput, extra_rev_per_wb_wafer=a_extra_rev,
        downstream_extra_per_wb=a_ds_extra,
    )
    supplier_b = SupplierParams(
        name="B", kgd_capacity=b_kgd_cap, wb_capacity=b_wb_cap,
        wb_cost_per_wafer=b_wb_cost, kgd_cost_per_wafer=b_kgd_cost,
        good_dies_per_wafer=b_gdpw, yield_mean=b_yield, yield_std=b_yield_std,
        dies_per_wafer=int(b_gdpw / b_yield) if b_yield > 0 else b_gdpw,
        hw_wafers_per_piece=b_hw_tput, extra_rev_per_wb_wafer=b_extra_rev,
        downstream_extra_per_wb=b_ds_extra,
    )
    costs = CostParams(
        test_cost_per_wafer=test_cost, transport_cost_per_wafer=transport_cost,
        eng_cost_per_wafer=eng_cost, hw_piece_cost=hw_piece_cost,
        selling_price_per_die=selling_price, downstream_cost_per_die=ds_cost_per_die,
    )
    return ProblemInstance(
        supplier_a=supplier_a, supplier_b=supplier_b,
        costs=costs, demand=d, uncertainty=unc_config,
    )


unc_config = UncertaintyConfig(
    uncertainty_type=unc_type_key,
    gamma=gamma, rho=rho,
    yield_set_width=yield_dev,
    cost_set_width=cost_dev,
    downstream_set_width=downstream_dev,
    extra_rev_set_width=extra_rev_dev,
)

instance = build_instance(demand)


def solve_all_custom(formulation: str, unc: UncertaintyConfig = None) -> dict[int, any]:
    """Solve for all demand levels using current sidebar parameters."""
    results = {}
    for d in DEMAND_LEVELS:
        inst = build_instance(d)
        if unc:
            inst.uncertainty = unc
        results[d] = solve(inst, formulation=formulation)
    return results


# ── Main Content ─────────────────────────────────────────────────────
st.title("Semiconductor Wafer Supply Chain")
st.markdown("### Static Robust Optimization Under Uncertainty")
st.caption(
    "Compares **static** robust planning (all decisions before uncertainty) with "
    "**two-stage adaptive** optimization (stage-1 allocation, then stage-2 recourse "
    "after observing yield; cost/revenue remain adversarial). Both use the same "
    "5D Bertsimas-Sim budget set. The model is independently parameterized, "
    "inspired by the ProjectReport but not a reproduction of it."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Supply Chain Overview",
    "Optimization Results",
    "Uncertainty Analysis",
    "Monte Carlo Simulation",
    "Sensitivity & Insights",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: Supply Chain Overview
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.header("The Fabless Semiconductor Supply Chain")

    st.markdown("""
    **Fabless** semiconductor companies (like Qualcomm) design chips but don't own
    fabrication plants. They must source wafers from external fabs and manage a complex
    supply chain spanning fabrication, testing, packaging, and delivery.
    """)

    st.plotly_chart(plot_supply_chain_diagram(), use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("KGD Model (Known Good Die)")
        st.markdown("""
        - **Pay per confirmed good die** (fixed price per wafer)
        - Fab handles testing logistics, yield analysis, transport
        - Lower risk, but higher per-die cost
        - No access to partially good dies
        - Preferred at **lower demand** levels
        """)

    with col2:
        st.subheader("WB Model (Wafer Buy)")
        st.markdown("""
        - **Pay per raw wafer** (lower price per wafer)
        - Company handles testing, transport, engineering
        - Can harvest **partially good dies** for extra revenue
        - Requires hardware investment (probe cards)
        - Becomes economic at **higher demand** levels
        """)

    st.markdown("---")
    st.subheader("Current Supplier Parameters")

    sa = instance.supplier_a
    sb = instance.supplier_b
    c = instance.costs

    param_df = pd.DataFrame({
        "Parameter": [
            "KGD Capacity (wafers/qtr)", "WB Capacity (wafers/qtr)",
            "KGD Cost ($/wafer)", "WB Cost ($/wafer)",
            "Good Dies/Wafer", "Yield Mean", "Yield Std Dev",
            "HW Test Throughput (wafers/piece)", "Extra Rev per WB Wafer ($)",
            "Downstream Extra per WB Wafer ($)",
        ],
        "Supplier A (Premium)": [
            f"{sa.kgd_capacity:,}", f"{sa.wb_capacity:,}",
            f"${sa.kgd_cost_per_wafer:,.0f}", f"${sa.wb_cost_per_wafer:,.0f}",
            f"{sa.good_dies_per_wafer:,}", f"{sa.yield_mean:.2f}", f"{sa.yield_std:.3f}",
            f"{sa.hw_wafers_per_piece:,}", f"${sa.extra_rev_per_wb_wafer:,.2f}",
            f"${sa.downstream_extra_per_wb:,.2f}",
        ],
        "Supplier B (Economy)": [
            f"{sb.kgd_capacity:,}", f"{sb.wb_capacity:,}",
            f"${sb.kgd_cost_per_wafer:,.0f}", f"${sb.wb_cost_per_wafer:,.0f}",
            f"{sb.good_dies_per_wafer:,}", f"{sb.yield_mean:.2f}", f"{sb.yield_std:.3f}",
            f"{sb.hw_wafers_per_piece:,}", f"${sb.extra_rev_per_wb_wafer:,.2f}",
            f"${sb.downstream_extra_per_wb:,.2f}",
        ],
    })
    st.dataframe(param_df, hide_index=True, use_container_width=True)

    with st.expander("Fixed Cost Parameters"):
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Test Cost per WB Wafer | ${c.test_cost_per_wafer:,.0f} |
        | Transport Cost per WB Wafer | ${c.transport_cost_per_wafer:,.0f} |
        | Engineering Cost per WB Wafer | ${c.eng_cost_per_wafer:,.0f} |
        | Hardware Piece Cost (Probe Card) | ${c.hw_piece_cost:,.0f} |
        | Selling Price per Die (to OEM) | ${c.selling_price_per_die:,.0f} |
        | Downstream Cost per Die | ${c.downstream_cost_per_die:,.0f} |
        """)

    with st.expander("Mathematical Formulation"):
        st.latex(r"""
        \max \quad \text{Revenue} - \text{TotalCost}
        """)
        st.latex(r"""
        \text{s.t.} \quad \text{gdpw}_A \cdot (W_{KGD,A} + W_{WB,A}) + \text{gdpw}_B \cdot (W_{KGD,B} + W_{WB,B}) \geq D
        """)
        st.latex(r"""
        W_{KGD,A} \leq C^{KGD}_A, \quad W_{KGD,B} \leq C^{KGD}_B, \quad W_{WB,A} \leq C^{WB}_A, \quad W_{WB,B} \leq C^{WB}_B
        """)
        st.markdown(f"""
        Where:
        - **Revenue** = D x ${c.selling_price_per_die:.0f} + Extra revenue from WB partial dies
        - **TotalCost** = KGD cost + WB cost + WB overhead + HW cost + Downstream cost
        - **gdpw** = good dies per wafer ({sa.good_dies_per_wafer} for A, {sb.good_dies_per_wafer} for B)
        - **D** = demand in dies
        """)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: Optimization Results
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Optimization Results")

    with st.spinner("Solving optimization..."):
        nom_inst = build_instance(demand)
        nom_inst.uncertainty = UncertaintyConfig()  # nominal = no uncertainty
        nom_result = solve(nom_inst, formulation="nominal")

        box_result = solve(build_instance(demand), formulation="box")

        ellip_inst = build_instance(demand)
        ellip_inst.uncertainty = UncertaintyConfig(
            uncertainty_type="ellipsoidal", rho=rho,
            yield_set_width=yield_dev, cost_set_width=cost_dev,
            downstream_set_width=downstream_dev,
            extra_rev_set_width=extra_rev_dev,
        )
        ellip_result = solve(ellip_inst, formulation="ellipsoidal")

        adaptive_result = solve(build_instance(demand), formulation="adaptive")

    col1, col2, col3, col4 = st.columns(4)

    def show_result_card(col, result, label):
        with col:
            st.subheader(label)
            status_color = "green" if result.status == "optimal" else "red"
            st.markdown(f"Status: :{status_color}[{result.status}] ({result.formulation})")

            st.metric("Nominal Profit", f"${result.nominal_profit/1e6:.1f}M",
                      help="Profit evaluated at nominal (expected) parameter values")
            if result.formulation != "nominal":
                st.metric("Robust Objective", f"${result.robust_objective/1e6:.1f}M",
                          help="Worst-case guaranteed profit under the uncertainty set")
            st.metric("Total Cost", f"${result.total_cost/1e6:.1f}M")
            st.metric("Revenue", f"${result.revenue/1e6:.1f}M")

            st.markdown(f"""
            | Wafers | Count |
            |---|---|
            | KGD-A | {result.wkgd_a:,} |
            | KGD-B | {result.wkgd_b:,} |
            | WB-A | {result.wwb_a:,} |
            | WB-B | {result.wwb_b:,} |
            | **Total** | **{result.wkgd_a + result.wkgd_b + result.wwb_a + result.wwb_b:,}** |
            """)

            surplus_pct = (result.demand_ratio - 1) * 100
            surplus_color = "green" if surplus_pct >= 0 else "red"
            st.markdown(f"Good Dies: {result.total_good_dies/1e6:.2f}M "
                       f"(:{surplus_color}[{surplus_pct:+.1f}%])")

    show_result_card(col1, nom_result, "Nominal")
    show_result_card(col2, box_result, "Static Robust")
    show_result_card(col3, ellip_result, "Ellipsoidal (approx)")
    show_result_card(col4, adaptive_result, "Two-Stage Adaptive")

    st.markdown("---")

    st.plotly_chart(
        plot_nominal_vs_robust(nom_result, box_result, "Nominal vs Box Robust"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_cost_breakdown(nom_result, "Nominal Cost Breakdown"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plot_cost_breakdown(box_result, "Box Robust Cost Breakdown"),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Demand Sweep: KGD vs WB Allocation")

    with st.spinner("Solving for all demand levels..."):
        nom_sweep = solve_all_custom(formulation="nominal", unc=UncertaintyConfig())
        rob_sweep = solve_all_custom(formulation="box", unc=unc_config)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_demand_sweep(nom_sweep, "Nominal: KGD vs WB by Demand"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plot_demand_sweep(rob_sweep, "Box Robust: KGD vs WB by Demand"),
            use_container_width=True,
        )

    st.subheader("Full Results Table")
    table_rows = []
    for d in DEMAND_LEVELS:
        nr = nom_sweep[d]
        rr = rob_sweep[d]
        cost_inc = (rr.total_cost - nr.total_cost) / nr.total_cost * 100 if nr.total_cost > 0 else 0
        table_rows.append({
            "Demand (M)": f"{d/1e6:.0f}",
            "Nom KGD": f"{nr.wkgd_a + nr.wkgd_b:,}",
            "Nom WB": f"{nr.wwb_a + nr.wwb_b:,}",
            "Nom Cost ($M)": f"{nr.total_cost/1e6:.1f}",
            "Nom Profit ($M)": f"{nr.nominal_profit/1e6:.1f}",
            "Rob KGD": f"{rr.wkgd_a + rr.wkgd_b:,}",
            "Rob WB": f"{rr.wwb_a + rr.wwb_b:,}",
            "Rob Cost ($M)": f"{rr.total_cost/1e6:.1f}",
            "Rob Nom Profit ($M)": f"{rr.nominal_profit/1e6:.1f}",
            "Rob Objective ($M)": f"{rr.robust_objective/1e6:.1f}",
            "Cost of Robustness": f"{cost_inc:.1f}%",
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3: Uncertainty Analysis
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Uncertainty Analysis")

    st.markdown("""
    Robust optimization protects against worst-case realizations of uncertain parameters.
    The **uncertainty set** defines the region of possible parameter values.
    Larger sets provide more protection but at higher cost.
    """)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Uncertainty Set Visualization")
        st.plotly_chart(
            plot_uncertainty_sets_2d(gamma=gamma, rho=rho),
            use_container_width=True,
        )

    with c2:
        st.subheader("How Uncertainty Sets Work")
        st.markdown(f"""
        **Budget Uncertainty / Bertsimas-Sim (Gamma = {gamma:.1f})**
        - Each normalized deviation |z_i| <= 1, with budget sum |z_i| <= Gamma
        - At most Gamma parameters hit their worst case simultaneously
        - The diamond shape reflects the L1 budget constraint
        - Result: Linear robust counterpart (LP/MIP)

        **Ellipsoidal (Rho = {rho:.1f})**
        - Parameters jointly constrained: ||z||_2 <= Rho
        - Less conservative than budget: allows partial deviations to trade off
        - Solved as SOCP with **LP relaxation + rounding** (approximate for MIP)

        **Robust Set Widths (planner conservatism):**
        - Yield: +/- {yield_dev*100:.0f}%
        - Cost: +/- {cost_dev*100:.0f}%
        - Downstream: +/- {downstream_dev*100:.0f}%
        - Revenue: +/- {extra_rev_dev*100:.0f}%

        **Simulation yield spread** uses supplier Yield Std / Yield Mean
        from the supplier panels (separate from robust set widths).
        """)

    st.markdown("---")
    st.subheader("Impact of Gamma on Cost and Profit")

    gamma_range = np.arange(0.0, 3.1, 0.25)

    nom_base = solve(build_instance(demand), formulation="nominal")
    sweep_nom_costs = [nom_base.total_cost] * len(gamma_range)
    sweep_rob_costs = []

    for g in gamma_range:
        inst_g = build_instance(demand)
        inst_g.uncertainty = UncertaintyConfig(
            uncertainty_type="box", gamma=g,
            yield_set_width=yield_dev,
            cost_set_width=cost_dev,
            downstream_set_width=downstream_dev,
            extra_rev_set_width=extra_rev_dev,
        )
        r = solve(inst_g, formulation="box")
        sweep_rob_costs.append(r.total_cost)

    st.plotly_chart(
        plot_cost_of_robustness(
            list(gamma_range), sweep_nom_costs, sweep_rob_costs,
        ),
        use_container_width=True,
    )

    st.markdown("""
    As Gamma increases:
    - The robust solution becomes more conservative (higher cost)
    - More wafers are ordered to protect against yield uncertainty
    - Supplier mix may shift toward more reliable (but expensive) options
    - At Gamma=0, the robust solution equals the nominal solution
    """)


# ══════════════════════════════════════════════════════════════════════
# TAB 4: Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Monte Carlo Simulation")

    st.markdown(f"""
    Evaluating both nominal and robust solutions under **{n_simulations:,} random scenarios**
    with uncertain yields, costs, and revenues. The allocation is fixed (from the optimizer);
    only the uncertain parameters are randomized.
    """)

    with st.spinner(f"Running {n_simulations:,} simulations..."):
        nom_inst_sim = build_instance(demand)
        nom_inst_sim.uncertainty = UncertaintyConfig()
        nom_for_sim = solve(nom_inst_sim, formulation="nominal")

        rob_for_sim = solve(build_instance(demand), formulation="box")
        adp_for_sim = solve(build_instance(demand), formulation="adaptive")

        sim_instance = build_instance(demand)
        nom_sim, rob_sim, adp_sim = compare_solutions(
            nom_for_sim, rob_for_sim, sim_instance,
            n_simulations=n_simulations, seed=int(mc_seed),
            adaptive_result=adp_for_sim,
        )

    st.subheader("Key Metrics")
    st.caption("Nominal/Static use full-demand revenue; Adaptive uses recourse payoff "
               "(salvage + lost-sales penalty). Profit definitions differ.")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Nominal: Demand Met", f"{nom_sim.demand_met_pct:.1f}%")
        st.metric("Nominal: Avg Profit", f"${nom_sim.expected_profit/1e6:.1f}M")
    with c2:
        st.metric("Static Robust: Demand Met", f"{rob_sim.demand_met_pct:.1f}%")
        st.metric("Static Robust: Avg Profit", f"${rob_sim.expected_profit/1e6:.1f}M")
    with c3:
        if adp_sim:
            st.metric("Adaptive: Demand Met", f"{adp_sim.demand_met_pct:.1f}%")
            st.metric("Adaptive: Avg Profit", f"${adp_sim.expected_profit/1e6:.1f}M",
                      help="Uses recourse payoff: salvage revenue - lost-sales penalty")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_monte_carlo_histograms(nom_sim, rob_sim),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plot_demand_met_comparison(nom_sim, rob_sim),
            use_container_width=True,
        )

    def _sim_col(sim):
        return [
            f"{sim.expected_profit/1e6:.1f}",
            f"{sim.profit_std/1e6:.1f}",
            f"{sim.worst_case_profit/1e6:.1f}",
            f"{sim.best_case_profit/1e6:.1f}",
            f"{sim.var_95/1e6:.1f}",
            f"{sim.cvar_95/1e6:.1f}",
            f"{sim.demand_met_pct:.1f}%",
            f"{sim.surplus_mean/1e6:.2f}",
            f"{np.mean(sim.deficit_scenarios)/1e6:.2f}" if len(sim.deficit_scenarios) > 0 else "N/A",
        ]

    st.subheader("Detailed Statistics")
    stats_data = {
        "Metric": [
            "Expected Profit ($M)", "Profit Std Dev ($M)",
            "Worst Case Profit ($M)", "Best Case Profit ($M)",
            "VaR 95% ($M)", "CVaR 95% ($M)",
            "Demand Met %", "Avg Surplus (M dies)",
            "Avg Deficit (M dies)",
        ],
        "Nominal": _sim_col(nom_sim),
        "Static Robust": _sim_col(rob_sim),
    }
    if adp_sim:
        stats_data["Adaptive (recourse)"] = _sim_col(adp_sim)
    st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5: Sensitivity & Insights
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Sensitivity Analysis & Insights")

    st.subheader("Cost of Robustness Across Demand Levels")
    rob_cost_data = []
    for d in DEMAND_LEVELS:
        nr = nom_sweep[d]
        rr = rob_sweep[d]
        cost_inc = (rr.total_cost - nr.total_cost) / nr.total_cost * 100 if nr.total_cost > 0 else 0
        profit_dec = (nr.nominal_profit - rr.nominal_profit) / nr.nominal_profit * 100 if nr.nominal_profit > 0 else 0
        rob_cost_data.append({
            "Demand (M)": f"{d/1e6:.0f}",
            "Nominal Cost ($M)": f"{nr.total_cost/1e6:.1f}",
            "Robust Cost ($M)": f"{rr.total_cost/1e6:.1f}",
            "Cost Increase (%)": f"{cost_inc:.1f}%",
            "Nominal Profit ($M)": f"{nr.nominal_profit/1e6:.1f}",
            "Robust Profit ($M)": f"{rr.nominal_profit/1e6:.1f}",
            "Profit Reduction (%)": f"{profit_dec:.1f}%",
            "Nom Surplus (%)": f"{(nr.demand_ratio-1)*100:.1f}%",
            "Rob Surplus (%)": f"{(rr.demand_ratio-1)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rob_cost_data), hide_index=True, use_container_width=True)

    st.markdown("---")

    st.subheader("Parameter Sensitivity")
    st.markdown("How does profit change when each uncertainty parameter varies by +/- 50% from current settings?")

    base_result = solve(instance, formulation="box")
    base_profit = base_result.nominal_profit

    param_configs = [
        ("Yield Set Width", "yield_set_width", yield_dev),
        ("Cost Set Width", "cost_set_width", cost_dev),
        ("Downstream Set Width", "downstream_set_width", downstream_dev),
        ("Extra Rev Set Width", "extra_rev_set_width", extra_rev_dev),
        ("Gamma", "gamma", gamma),
    ]

    param_names = []
    low_profits = []
    high_profits = []

    for pname, pattr, pval in param_configs:
        for factor, target in [(0.5, "low"), (1.5, "high")]:
            new_uc = UncertaintyConfig(
                uncertainty_type="box",
                gamma=gamma if pattr != "gamma" else pval * factor,
                yield_set_width=yield_dev if pattr != "yield_set_width" else pval * factor,
                cost_set_width=cost_dev if pattr != "cost_set_width" else pval * factor,
                downstream_set_width=downstream_dev if pattr != "downstream_set_width" else pval * factor,
                extra_rev_set_width=extra_rev_dev if pattr != "extra_rev_set_width" else pval * factor,
            )
            inst_s = build_instance(demand)
            inst_s.uncertainty = new_uc
            r = solve(inst_s, formulation="box")
            if target == "low":
                low_profits.append(r.nominal_profit)
            else:
                high_profits.append(r.nominal_profit)
        param_names.append(pname)

    st.plotly_chart(
        plot_sensitivity_tornado(param_names, low_profits, high_profits, base_profit),
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Key Takeaways")

    st.info("""
    **1. KGD dominates at low demand; WB enters at high demand**

    At low demand levels, the cheaper Supplier B KGD model alone can satisfy
    demand. The Wafer Buy model only becomes economic when KGD capacity is exhausted,
    typically above 20-30M die demand.
    """)

    st.warning("""
    **2. Robustness has a cost -- but provides insurance**

    The robust solution costs more than the nominal solution (depending on uncertainty
    parameters and Gamma). However, it provides a buffer of surplus dies, protecting against
    yield variability and supply disruptions. The "Robust Objective" shows the worst-case
    guaranteed profit; the "Nominal Profit" shows what you'd realize if parameters come
    in at their expected values.
    """)

    st.success("""
    **3. The surplus from robust optimization isn't wasted**

    Semiconductor ICs have a "long tail" product lifecycle. The same chips may be sold in
    smaller volumes for up to a decade. Surplus inventory from robust planning can fill
    future orders, recovering some of the robustness cost.
    """)

    st.info("""
    **4. Yield uncertainty is the most impactful parameter**

    Sensitivity analysis shows that yield uncertainty has the largest impact on the optimal
    allocation and total cost. This is consistent with industry experience where fab yield
    excursions are the primary supply chain risk.
    """)

    st.warning("""
    **5. Supplier diversification increases with uncertainty**

    Under higher uncertainty (larger Gamma/Rho), the robust optimizer diversifies across
    both suppliers and both sourcing models. This multi-sourcing strategy provides natural
    hedging against correlated supply disruptions.
    """)

    st.markdown("---")
    st.subheader("Model Limitations")
    st.markdown("""
    - **Static vs adaptive**: Both models use the same 5D Bertsimas-Sim budget set.
      The static model makes all decisions before any uncertainty is revealed.
      The adaptive model observes **yield only** (z1, z2) before making recourse
      decisions (fulfill, salvage, lost sales). Cost/revenue parameters (z3-z5)
      remain adversarial and unobserved. Because the adaptive payoff includes
      salvage revenue and lost-sales penalties (which the static model does not),
      their robust objectives are not directly comparable.
    - **Ellipsoidal formulation is approximate**: The SOCP is solved as an LP relaxation
      and rounded to integers. This is a feasible but potentially suboptimal approximation.
    - **Independently parameterized**: This model is inspired by the ProjectReport but
      uses independently derived parameters. Results differ from the appendix tables
      because the cost structure and downstream model are constructed from partial data.
    - **Shortage modeling differs by formulation**: The static model loses revenue on
      unproduced dies but has no explicit penalty. The adaptive model adds a lost-sales
      penalty (default 150% of selling price) and salvage revenue (30%). Neither model
      captures contractual penalties, reputation damage, or expediting costs beyond
      these stylized parameters.
    """)
