# Stress the shared cause, then compare the sourcing policies

Method update: October 18, 2026. Proposed experiments below are not reported as executed results.

The sourcing simulator compares formulations under uncertain wafer yields. Independent yield draws can understate a shared disruption when several suppliers depend on one process input, region or equipment source. Correlation should be an explicit scenario assumption rather than inferred from the number of supplier names.

The next proposed synthetic experiment holds supplier means and marginal variation fixed while changing a common-shock component. Use the same realized yields for every policy at each seed. Evaluate delivered good units, unmet demand, expedite cost if modeled and total cost; report demand shortfall in every draw, and report any optimizer runs that come back infeasible across the common-shock sweep rather than dropping them.

Keep training scenarios separate from held-out evaluation draws when choosing uncertainty-set sizes or adaptive rules. The adaptive model observes yield only, so it must not see the cost/revenue draws (z3–z5) or any future shock information unavailable to its competitors. A lower simulated mean cost remains conditional on that information structure, solver status and recourse assumptions. Save the scenario generator and policy configuration with the scorecard.

## References

- https://doi.org/10.1287/opre.1030.0065
