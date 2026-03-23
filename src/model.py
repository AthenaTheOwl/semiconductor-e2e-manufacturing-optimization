"""Data model for the semiconductor wafer supply chain robust optimization problem."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SupplierParams:
    """Parameters for a single wafer supplier (fab).

    Each supplier has capacity limits for both KGD and WB sourcing models,
    different cost structures, and different yield characteristics.
    """
    name: str
    kgd_capacity: int              # Max KGD wafers per quarter
    wb_capacity: int               # Max WB wafers per quarter
    wb_cost_per_wafer: float       # WB wafer purchase price ($)
    kgd_cost_per_wafer: float      # KGD cost per wafer (includes yield guarantee) ($)
    good_dies_per_wafer: int       # Expected good dies per wafer (yield * die sites)
    yield_mean: float              # Mean yield ratio for simulation (0 to 1)
    yield_std: float               # Yield std dev for simulation
    dies_per_wafer: int            # Total die sites on wafer (raw, pre-yield)
    hw_wafers_per_piece: int       # WB wafers one HW piece can test per quarter
    extra_rev_per_wb_wafer: float  # Revenue from partially good dies per WB wafer ($)
    downstream_extra_per_wb: float # Extra downstream cost per WB wafer ($)


@dataclass
class CostParams:
    """Fixed cost parameters for the supply chain."""
    test_cost_per_wafer: float = 2000.0       # Wafer sort test cost per WB wafer
    transport_cost_per_wafer: float = 80.0    # Transport cost per WB wafer
    eng_cost_per_wafer: float = 5.0           # Engineering support cost per WB wafer
    hw_piece_cost: float = 300000.0           # Cost per hardware piece (probe card)
    selling_price_per_die: float = 35.0       # Selling price per good die to OEM
    downstream_cost_per_die: float = 7.0      # Base downstream cost per die (applied to demand)

    @property
    def total_wb_overhead_per_wafer(self) -> float:
        """Total per-wafer overhead for WB model (test + transport + engineering)."""
        return self.test_cost_per_wafer + self.transport_cost_per_wafer + self.eng_cost_per_wafer


@dataclass
class UncertaintyConfig:
    """Configuration for the robust optimization uncertainty sets.

    These parameters define the PLANNING uncertainty sets used by the
    optimizer to construct robust counterparts. They are SEPARATE from
    the simulation randomness used in Monte Carlo, which is driven by
    the supplier-level yield_mean/yield_std parameters.

    The set-width fractions define the half-width of each uncertain
    coefficient's interval: a_i in [a_nom * (1 - dev), a_nom * (1 + dev)].
    The budget parameter Gamma (or radius Rho) then controls how many
    of these intervals are simultaneously active.
    """
    uncertainty_type: str = "nominal"  # "nominal", "box", "ellipsoidal"
    gamma: float = 1.0                # Box uncertainty budget parameter (Gamma)
    rho: float = 1.0                  # Ellipsoidal uncertainty radius

    # Robust set half-widths (fraction of nominal). These control how
    # conservative the PLANNER is, not the actual randomness in simulation.
    yield_set_width: float = 0.10            # Yield set half-width +/- 10%
    cost_set_width: float = 0.15             # WB overhead cost set half-width
    downstream_set_width: float = 0.20       # Downstream cost set half-width
    extra_rev_set_width: float = 0.25        # Partial-die revenue set half-width


@dataclass
class ProblemInstance:
    """Complete problem instance combining all parameters."""
    supplier_a: SupplierParams
    supplier_b: SupplierParams
    costs: CostParams
    demand: int  # Target good die demand
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)


@dataclass
class OptimizationResult:
    """Result of an optimization solve.

    Two profit figures are reported:
    - `nominal_profit`: Revenue minus cost computed with NOMINAL coefficients.
      This is what would actually be realized if all parameters hit their
      nominal values. Use this for apples-to-apples comparison across
      formulations.
    - `robust_objective`: The value the solver actually optimized. For nominal
      solves this equals nominal_profit. For robust solves this is the
      worst-case guaranteed profit under the uncertainty set -- always lower
      than nominal_profit for the same allocation.
    """
    status: str                # "optimal", "infeasible", etc.
    robust_objective: float    # Solver's objective value (worst-case for robust)

    # Decision variables
    wkgd_a: int                # KGD wafers from supplier A
    wkgd_b: int                # KGD wafers from supplier B
    wwb_a: int                 # WB wafers from supplier A
    wwb_b: int                 # WB wafers from supplier B
    hw_pcs_a: int              # Hardware pieces for A
    hw_pcs_b: int              # Hardware pieces for B

    # Cost breakdown (all evaluated at NOMINAL coefficients)
    wb_cost: float             # Total WB wafer purchase cost
    kgd_cost: float            # Total KGD wafer cost
    wb_overhead: float         # Test + transport + engineering for WB
    hw_cost: float             # Hardware piece cost
    downstream_cost: float     # Downstream processing cost
    total_cost: float          # Sum of all costs (nominal)
    revenue: float             # Sales revenue + partial die revenue (nominal)
    nominal_profit: float      # Revenue - TotalCost at nominal coefficients

    # Demand analysis
    total_good_dies: float     # Total good dies produced (at nominal yield)
    demand: int                # Target demand
    demand_surplus: float      # total_good_dies - demand (negative = deficit)
    demand_ratio: float        # total_good_dies / demand

    solve_time: float = 0.0
    formulation: str = "nominal"  # "nominal", "box", "ellipsoidal (approx)"

    # Adaptive policy: maps yield state (z1, z2) -> (fulfill, salvage).
    # Only populated for two-stage adaptive solves. Stored for
    # introspection; the simulator uses the exact greedy recourse rule
    # (which is analytically optimal for the current payoff structure).
    # None for static formulations.
    adaptive_policy: dict[tuple[float, float], tuple[float, float]] | None = field(
        default=None, repr=False,
    )


@dataclass
class ScenarioRealization:
    """A single realization of uncertain parameters for Monte Carlo simulation."""
    yield_a: float
    yield_b: float
    test_cost_per_wafer: float
    transport_cost_per_wafer: float
    eng_cost_per_wafer: float
    downstream_scale: float    # Multiplier on downstream costs
    extra_rev_factor_a: float  # Multiplier on nominal extra revenue
    extra_rev_factor_b: float


@dataclass
class SimulationOutput:
    """Output from Monte Carlo simulation of a single solution."""
    profits: np.ndarray
    costs: np.ndarray
    revenues: np.ndarray
    total_dies: np.ndarray
    demand: int

    @property
    def n_simulations(self) -> int:
        return len(self.profits)

    @property
    def demand_met_count(self) -> int:
        return int(np.sum(self.total_dies >= self.demand))

    @property
    def demand_met_pct(self) -> float:
        return self.demand_met_count / self.n_simulations * 100

    @property
    def expected_profit(self) -> float:
        return float(np.mean(self.profits))

    @property
    def worst_case_profit(self) -> float:
        return float(np.min(self.profits))

    @property
    def best_case_profit(self) -> float:
        return float(np.max(self.profits))

    @property
    def profit_std(self) -> float:
        return float(np.std(self.profits))

    @property
    def var_95(self) -> float:
        """Value at Risk at 95% confidence (5th percentile of profit)."""
        return float(np.percentile(self.profits, 5))

    @property
    def cvar_95(self) -> float:
        """Conditional VaR at 95% (expected profit in worst 5%)."""
        threshold = np.percentile(self.profits, 5)
        worst = self.profits[self.profits <= threshold]
        return float(np.mean(worst)) if len(worst) > 0 else float(threshold)

    @property
    def surplus_mean(self) -> float:
        return float(np.mean(self.total_dies - self.demand))

    @property
    def deficit_scenarios(self) -> np.ndarray:
        """Array of deficit amounts for scenarios where demand is NOT met."""
        deficits = self.demand - self.total_dies
        return deficits[deficits > 0]


def create_default_suppliers() -> tuple[SupplierParams, SupplierParams]:
    """Create default supplier A and B with parameters derived from the report.

    Key relationships (from the report's formulation and results):
    - A: WB:KGD capacity ratio = 3:7 -> 7500:17500
    - B: WB:KGD capacity ratio = 2:3 -> 20000:30000
    - Total A:B capacity = 1:2 -> 25000:50000
    - A: higher quality, tighter process, higher cost
    - B: cheaper, lower quality, broader variation, higher capacity
    - HW test throughput: A=1500 wafers/piece, B=1200 wafers/piece
    - Extra revenue per WB wafer: A=$787.50, B=$656.25

    Good dies per wafer derived from report results:
    - B: ~705 (verified: ceil(10M/705)=14185 matching report's KGD-B for 10M demand)
    - A: ~882 (verified across multiple demand levels)

    KGD cost per wafer derived from report:
    - B: $15,750 (verified: 15750*14185 = 223,413,750 matching report)
    - A: $22,500 (verified: residual cost matches across demand levels)
    """
    supplier_a = SupplierParams(
        name="A",
        kgd_capacity=17500,
        wb_capacity=7500,
        wb_cost_per_wafer=20000.0,
        kgd_cost_per_wafer=22500.0,
        good_dies_per_wafer=882,
        yield_mean=0.92,
        yield_std=0.03,
        dies_per_wafer=1500,
        hw_wafers_per_piece=1500,
        extra_rev_per_wb_wafer=787.50,
        downstream_extra_per_wb=157.50,
    )

    supplier_b = SupplierParams(
        name="B",
        kgd_capacity=30000,
        wb_capacity=20000,
        wb_cost_per_wafer=15000.0,
        kgd_cost_per_wafer=15750.0,
        good_dies_per_wafer=705,
        yield_mean=0.85,
        yield_std=0.06,
        dies_per_wafer=1200,
        hw_wafers_per_piece=1200,
        extra_rev_per_wb_wafer=656.25,
        downstream_extra_per_wb=131.25,
    )

    return supplier_a, supplier_b


def create_default_instance(demand: int = 1_000_000) -> ProblemInstance:
    """Create a default problem instance for a given demand level."""
    supplier_a, supplier_b = create_default_suppliers()
    return ProblemInstance(
        supplier_a=supplier_a,
        supplier_b=supplier_b,
        costs=CostParams(),
        demand=demand,
        uncertainty=UncertaintyConfig(),
    )


# Standard demand levels from the report (in units of dies)
DEMAND_LEVELS = [
    1_000_000, 5_000_000, 10_000_000, 20_000_000,
    30_000_000, 40_000_000, 50_000_000,
]
