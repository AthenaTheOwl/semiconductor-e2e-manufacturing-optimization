"""Plotly visualization utilities for the semiconductor wafer supply chain dashboard."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .model import OptimizationResult, SimulationOutput


# Color palette
COLORS = {
    "kgd_a": "#2196F3",   # Blue
    "kgd_b": "#42A5F5",   # Light blue
    "wb_a": "#FF9800",    # Orange
    "wb_b": "#FFB74D",    # Light orange
    "nominal": "#2196F3",
    "robust_box": "#FF5722",
    "robust_ellip": "#9C27B0",
    "profit": "#4CAF50",
    "cost": "#F44336",
    "revenue": "#2196F3",
}


def plot_supply_chain_diagram() -> go.Figure:
    """Create an interactive supply chain flow diagram using Plotly shapes."""
    fig = go.Figure()

    # Box positions (x, y, width, height)
    boxes = {
        "IC Design":       (0.02, 0.55, 0.12, 0.12),
        "Masking":         (0.02, 0.30, 0.12, 0.10),
        "Fabrication\nHouse": (0.22, 0.45, 0.14, 0.15),
        "Wafer Test\nSite":  (0.42, 0.45, 0.14, 0.15),
        "Packaging\nHouse":  (0.62, 0.45, 0.14, 0.15),
        "Final Test\nSite":  (0.82, 0.45, 0.14, 0.15),
        "OEM":             (0.82, 0.18, 0.14, 0.10),
        "End User":        (0.82, 0.02, 0.14, 0.10),
    }

    equipment = {
        "Mfg\nEquipment":   (0.22, 0.75, 0.12, 0.10),
        "Testing\nEquipment": (0.42, 0.75, 0.12, 0.10),
        "Packaging\nEquipment": (0.62, 0.75, 0.12, 0.10),
        "Raw Wafer":       (0.22, 0.18, 0.10, 0.08),
        "Chemicals":       (0.22, 0.05, 0.10, 0.08),
        "Probe Card":      (0.42, 0.18, 0.10, 0.08),
        "Load Board":      (0.62, 0.18, 0.10, 0.08),
    }

    # Draw main process boxes
    for label, (x, y, w, h) in boxes.items():
        fig.add_shape(
            type="rect", x0=x, y0=y, x1=x+w, y1=y+h,
            fillcolor="#B0BEC5", line=dict(color="#455A64", width=2),
            layer="below",
        )
        fig.add_annotation(
            x=x+w/2, y=y+h/2, text=f"<b>{label}</b>",
            showarrow=False, font=dict(size=11, color="#212121"),
        )

    # Draw equipment/input boxes
    for label, (x, y, w, h) in equipment.items():
        fig.add_shape(
            type="rect", x0=x, y0=y, x1=x+w, y1=y+h,
            fillcolor="#E3F2FD", line=dict(color="#90CAF9", width=1.5),
            layer="below",
        )
        fig.add_annotation(
            x=x+w/2, y=y+h/2, text=label,
            showarrow=False, font=dict(size=9, color="#1565C0"),
        )

    # Draw arrows (main flow)
    arrows = [
        (0.14, 0.61, 0.22, 0.55),   # IC Design -> Fab
        (0.14, 0.35, 0.22, 0.48),   # Masking -> Fab
        (0.36, 0.52, 0.42, 0.52),   # Fab -> Wafer Test
        (0.56, 0.52, 0.62, 0.52),   # Wafer Test -> Packaging
        (0.76, 0.52, 0.82, 0.52),   # Packaging -> Final Test
        (0.89, 0.45, 0.89, 0.28),   # Final Test -> OEM
        (0.89, 0.18, 0.89, 0.12),   # OEM -> End User
    ]

    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=2, arrowcolor="#455A64",
        )

    # Equipment arrows (down to process)
    equip_arrows = [
        (0.28, 0.75, 0.29, 0.60),
        (0.48, 0.75, 0.49, 0.60),
        (0.68, 0.75, 0.69, 0.60),
        (0.27, 0.26, 0.27, 0.45),
        (0.27, 0.13, 0.27, 0.18),
        (0.47, 0.26, 0.47, 0.45),
        (0.67, 0.26, 0.67, 0.45),
    ]

    for x0, y0, x1, y1 in equip_arrows:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor="#90CAF9",
        )

    fig.update_layout(
        xaxis=dict(range=[-0.02, 1.02], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.02, 0.92], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )

    return fig


def plot_wafer_allocation(result: OptimizationResult, title: str = "") -> go.Figure:
    """Stacked bar chart of wafer allocation by supplier and model."""
    categories = ["KGD-A", "KGD-B", "WB-A", "WB-B"]
    values = [result.wkgd_a, result.wkgd_b, result.wwb_a, result.wwb_b]
    colors = [COLORS["kgd_a"], COLORS["kgd_b"], COLORS["wb_a"], COLORS["wb_b"]]

    fig = go.Figure(data=[
        go.Bar(
            x=categories, y=values,
            marker_color=colors,
            text=[f"{v:,}" for v in values],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title=title or f"Wafer Allocation (Demand={result.demand/1e6:.0f}M)",
        yaxis_title="Number of Wafers",
        height=350,
        margin=dict(l=50, r=20, t=40, b=30),
    )
    return fig


def plot_demand_sweep(results: dict[int, OptimizationResult], title: str = "") -> go.Figure:
    """Line chart of KGD vs WB wafer counts as a function of demand."""
    demands = sorted(results.keys())
    demand_labels = [d / 1e6 for d in demands]

    total_kgd = [results[d].wkgd_a + results[d].wkgd_b for d in demands]
    total_wb = [results[d].wwb_a + results[d].wwb_b for d in demands]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=demand_labels, y=total_kgd, mode="lines+markers",
        name="Total KGD", line=dict(color=COLORS["nominal"], width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=demand_labels, y=total_wb, mode="lines+markers",
        name="Total WB", line=dict(color=COLORS["robust_box"], width=3),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title=title or "Wafer Buy and KGD Wafer Division as a Function of Demand",
        xaxis_title="Modeled Demand (Millions)",
        yaxis_title="Number of Wafers",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def plot_cost_breakdown(result: OptimizationResult, title: str = "") -> go.Figure:
    """Waterfall chart showing cost breakdown and profit."""
    categories = ["KGD Cost", "WB Cost", "WB Overhead", "HW Cost",
                   "Downstream", "Total Cost", "Revenue", "Profit"]
    values = [
        result.kgd_cost, result.wb_cost, result.wb_overhead,
        result.hw_cost, result.downstream_cost,
        result.total_cost, result.revenue, result.nominal_profit,
    ]

    measures = ["relative", "relative", "relative", "relative",
                "relative", "total", "total", "total"]

    fig = go.Figure(go.Waterfall(
        x=categories,
        y=[result.kgd_cost, result.wb_cost, result.wb_overhead,
           result.hw_cost, result.downstream_cost,
           0, 0, 0],
        measure=measures,
        text=[f"${v/1e6:.1f}M" for v in values],
        textposition="outside",
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
    ))

    # Replace with simpler grouped bar
    fig = go.Figure()
    labels = ["KGD Cost", "WB Cost", "WB Overhead", "HW Cost", "Downstream"]
    vals = [result.kgd_cost, result.wb_cost, result.wb_overhead,
            result.hw_cost, result.downstream_cost]
    colors_list = ["#1976D2", "#FF9800", "#FFC107", "#795548", "#9E9E9E"]

    fig.add_trace(go.Bar(
        x=labels, y=vals,
        marker_color=colors_list,
        text=[f"${v/1e6:.1f}M" for v in vals],
        textposition="auto",
    ))

    fig.update_layout(
        title=title or f"Cost Breakdown (Demand={result.demand/1e6:.0f}M)",
        yaxis_title="Cost ($)",
        height=350,
        margin=dict(l=60, r=20, t=40, b=30),
    )
    return fig


def plot_nominal_vs_robust(
    nominal: OptimizationResult,
    robust: OptimizationResult,
    title: str = "",
) -> go.Figure:
    """Side-by-side comparison of nominal and robust solutions."""
    categories = ["Total Cost", "Revenue", "Profit", "Good Dies"]
    nom_values = [nominal.total_cost, nominal.revenue, nominal.nominal_profit, nominal.total_good_dies]
    rob_values = [robust.total_cost, robust.revenue, robust.nominal_profit, robust.total_good_dies]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Financials ($)", "Die Output"])

    # Financials
    fig.add_trace(go.Bar(
        x=["Total Cost", "Revenue", "Profit"],
        y=[nominal.total_cost, nominal.revenue, nominal.nominal_profit],
        name="Nominal", marker_color=COLORS["nominal"],
        text=[f"${v/1e6:.0f}M" for v in [nominal.total_cost, nominal.revenue, nominal.nominal_profit]],
        textposition="auto",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=["Total Cost", "Revenue", "Profit"],
        y=[robust.total_cost, robust.revenue, robust.nominal_profit],
        name="Robust", marker_color=COLORS["robust_box"],
        text=[f"${v/1e6:.0f}M" for v in [robust.total_cost, robust.revenue, robust.nominal_profit]],
        textposition="auto",
    ), row=1, col=1)

    # Die output
    fig.add_trace(go.Bar(
        x=["Demand", "Nominal Dies", "Robust Dies"],
        y=[nominal.demand, nominal.total_good_dies, robust.total_good_dies],
        marker_color=["#9E9E9E", COLORS["nominal"], COLORS["robust_box"]],
        text=[f"{v/1e6:.1f}M" for v in [nominal.demand, nominal.total_good_dies, robust.total_good_dies]],
        textposition="auto",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=title or "Nominal vs Robust Comparison",
        height=380,
        margin=dict(l=50, r=20, t=60, b=30),
        barmode="group",
    )
    return fig


def plot_uncertainty_sets_2d(gamma: float = 1.0, rho: float = 1.0) -> go.Figure:
    """Visualize Bertsimas-Sim budget and ellipsoidal uncertainty sets in 2D.

    Budget set (Bertsimas-Sim): { z : |z1| + |z2| <= Gamma, |z_i| <= 1 }
    This is the intersection of an L1 ball (diamond) and an L-inf ball (box).
    For Gamma < 1: a diamond with vertices at (+/-Gamma, 0) and (0, +/-Gamma).
    For 1 <= Gamma < 2: a hexagon (truncated diamond).
    For Gamma >= 2: the full [-1,1]^2 box.

    Ellipsoidal set: { z : ||z||_2 <= rho }
    """
    fig = go.Figure()

    # Bertsimas-Sim budget set: |z1| + |z2| <= Gamma AND |z_i| <= 1
    # Compute the intersection polygon
    g = min(gamma, 2.0)  # cap at 2 for 2D (max L parameters)
    if g <= 0:
        # Degenerate: just the origin
        budget_x, budget_y = [0], [0]
    elif g <= 1.0:
        # Diamond: vertices at (g,0), (0,g), (-g,0), (0,-g)
        budget_x = [g, 0, -g, 0, g]
        budget_y = [0, g, 0, -g, 0]
    elif g < 2.0:
        # Truncated diamond (hexagon): L1 ball clipped by |z_i| <= 1
        # Vertices where |z1|+|z2|=Gamma intersects |z_i|=1
        r = g - 1  # the remaining budget when one is at 1
        budget_x = [1, r, -r, -1, -1, -r, r, 1, 1]
        budget_y = [g-1, 1, 1, g-1, -(g-1), -1, -1, -(g-1), g-1]
        # Simpler: just trace the 8 vertices
        budget_x = [1, g-1, 0, -(g-1), -1, -1, -(g-1), 0, g-1, 1, 1]
        budget_y = [0, 1, 1*(g<=1.5)+min(g,1)*(g>1.5), 1, g-1, -(g-1), -1, -min(g,1)*(g>1.5)-1*(g<=1.5), -1, 0, 0]
        # Actually let me just compute this properly
        # The set is { z : |z1|+|z2| <= g } intersect [-1,1]^2
        # In quadrant 1: z1+z2 <= g, z1 <= 1, z2 <= 1
        # Corner points in Q1: (min(g,1), 0), if g>1: (1, g-1), (g-1, 1) else: skip, (0, min(g,1))
        pts = []
        if g <= 1:
            pts = [(g,0), (0,g), (-g,0), (0,-g)]
        else:
            pts = [
                (1, 0), (1, g-1), (g-1, 1), (0, 1),
                (-g+1, 1), (-1, g-1), (-1, 0),
                (-1, -(g-1)), (-(g-1), -1), (0, -1),
                (g-1, -1), (1, -(g-1)),
            ]
        pts.append(pts[0])  # close polygon
        budget_x = [p[0] for p in pts]
        budget_y = [p[1] for p in pts]
    else:
        # Full box [-1,1]^2
        budget_x = [-1, 1, 1, -1, -1]
        budget_y = [-1, -1, 1, 1, -1]

    fig.add_trace(go.Scatter(
        x=budget_x, y=budget_y, mode="lines",
        name=f"Budget (Gamma={gamma:.1f})",
        line=dict(color=COLORS["robust_box"], width=2),
        fill="toself", fillcolor="rgba(255,87,34,0.1)",
    ))

    # Ellipsoidal uncertainty set
    theta = np.linspace(0, 2*np.pi, 100)
    ellip_x = rho * np.cos(theta)
    ellip_y = rho * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=ellip_x.tolist(), y=ellip_y.tolist(), mode="lines",
        name=f"Ellipsoidal (rho={rho:.1f})",
        line=dict(color=COLORS["robust_ellip"], width=2),
        fill="toself", fillcolor="rgba(156,39,176,0.1)",
    ))

    # Box for reference (interval uncertainty |z_i| <= 1)
    fig.add_trace(go.Scatter(
        x=[-1, 1, 1, -1, -1], y=[-1, -1, 1, 1, -1],
        mode="lines", name="Interval (|z|<=1)",
        line=dict(color="#9E9E9E", width=1, dash="dot"),
    ))

    # Origin
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        name="Nominal",
        marker=dict(size=10, color="black", symbol="x"),
    ))

    max_range = max(gamma, rho, 1.0) * 1.3
    fig.update_layout(
        title="Uncertainty Sets (2D Projection onto Yield Coordinates z1, z2)",
        xaxis_title="Supplier A Yield Coordinate (z1)",
        yaxis_title="Supplier B Yield Coordinate (z2)",
        xaxis=dict(range=[-max_range, max_range]),
        yaxis=dict(range=[-max_range, max_range], scaleanchor="x"),
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_monte_carlo_histograms(
    nominal_sim: SimulationOutput,
    robust_sim: SimulationOutput,
) -> go.Figure:
    """Overlaid histograms of profit distribution for nominal vs robust."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=nominal_sim.profits / 1e6,
        name="Nominal", opacity=0.6,
        marker_color=COLORS["nominal"],
        nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=robust_sim.profits / 1e6,
        name="Robust", opacity=0.6,
        marker_color=COLORS["robust_box"],
        nbinsx=50,
    ))

    fig.update_layout(
        title="Profit Distribution Under Uncertainty",
        xaxis_title="Profit ($ Millions)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400,
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def plot_demand_met_comparison(
    nominal_sim: SimulationOutput,
    robust_sim: SimulationOutput,
) -> go.Figure:
    """Histogram comparing demand fulfillment (total dies / demand ratio)."""
    fig = go.Figure()

    nom_ratio = nominal_sim.total_dies / nominal_sim.demand
    rob_ratio = robust_sim.total_dies / robust_sim.demand

    fig.add_trace(go.Histogram(
        x=nom_ratio,
        name="Nominal", opacity=0.6,
        marker_color=COLORS["nominal"],
        nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=rob_ratio,
        name="Robust", opacity=0.6,
        marker_color=COLORS["robust_box"],
        nbinsx=50,
    ))

    # Add demand line
    fig.add_vline(x=1.0, line_dash="dash", line_color="red",
                  annotation_text="Demand = 100%")

    fig.update_layout(
        title="Demand Fulfillment Ratio Distribution",
        xaxis_title="Total Dies / Demand",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400,
        margin=dict(l=50, r=20, t=40, b=30),
    )
    return fig


def plot_cost_of_robustness(
    gamma_values: list[float],
    nominal_costs: list[float],
    robust_costs: list[float],
) -> go.Figure:
    """Line chart showing cost increase as a function of uncertainty budget."""
    pct_increase = [
        (r - n) / n * 100 if n > 0 else 0
        for n, r in zip(nominal_costs, robust_costs)
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=gamma_values, y=[c/1e6 for c in nominal_costs],
        name="Nominal Cost", mode="lines",
        line=dict(color=COLORS["nominal"], width=2, dash="dash"),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=gamma_values, y=[c/1e6 for c in robust_costs],
        name="Robust Cost", mode="lines+markers",
        line=dict(color=COLORS["robust_box"], width=2),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=gamma_values, y=pct_increase,
        name="Cost Increase %", mode="lines+markers",
        line=dict(color=COLORS["profit"], width=2),
    ), secondary_y=True)

    fig.update_layout(
        title="Cost of Robustness vs Uncertainty Budget (Gamma)",
        xaxis_title="Gamma",
        height=400,
        margin=dict(l=60, r=60, t=50, b=40),
    )
    fig.update_yaxes(title_text="Total Cost ($ Millions)", secondary_y=False)
    fig.update_yaxes(title_text="Cost Increase (%)", secondary_y=True)

    return fig


def plot_sensitivity_tornado(
    param_names: list[str],
    low_profits: list[float],
    high_profits: list[float],
    base_profit: float,
) -> go.Figure:
    """Tornado chart showing sensitivity of profit to each parameter."""
    # Sort by range
    ranges = [abs(h - l) for l, h in zip(low_profits, high_profits)]
    sorted_idx = np.argsort(ranges)

    fig = go.Figure()

    for i, idx in enumerate(sorted_idx):
        low_delta = (low_profits[idx] - base_profit) / 1e6
        high_delta = (high_profits[idx] - base_profit) / 1e6

        fig.add_trace(go.Bar(
            y=[param_names[idx]], x=[low_delta],
            orientation="h", marker_color=COLORS["cost"],
            name="Low" if i == 0 else None, showlegend=(i == 0),
            text=f"${low_profits[idx]/1e6:.1f}M",
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            y=[param_names[idx]], x=[high_delta],
            orientation="h", marker_color=COLORS["profit"],
            name="High" if i == 0 else None, showlegend=(i == 0),
            text=f"${high_profits[idx]/1e6:.1f}M",
            textposition="outside",
        ))

    fig.add_vline(x=0, line_color="black", line_width=1)

    fig.update_layout(
        title="Sensitivity Analysis (Profit Change from Base)",
        xaxis_title="Profit Change ($ Millions)",
        barmode="overlay",
        height=max(300, 50 * len(param_names)),
        margin=dict(l=120, r=80, t=40, b=30),
    )
    return fig


def plot_comparison_table_data(results: dict[str, OptimizationResult]) -> list[dict]:
    """Convert optimization results to table data for display."""
    rows = []
    for label, r in results.items():
        rows.append({
            "Formulation": label,
            "KGD-A": r.wkgd_a,
            "KGD-B": r.wkgd_b,
            "WB-A": r.wwb_a,
            "WB-B": r.wwb_b,
            "Total Cost ($M)": f"{r.total_cost/1e6:.1f}",
            "Revenue ($M)": f"{r.revenue/1e6:.1f}",
            "Nominal Profit ($M)": f"{r.nominal_profit/1e6:.1f}",
            "Robust Obj ($M)": f"{r.robust_objective/1e6:.1f}",
            "Good Dies (M)": f"{r.total_good_dies/1e6:.2f}",
            "Surplus %": f"{(r.demand_ratio - 1)*100:.1f}%",
        })
    return rows
