# Stress the shared cause, then compare the sourcing policies

Method update: October 18, 2026. Proposed experiments below are not reported as executed results.

The sourcing simulator compares formulations under uncertain wafer yields. Independent yield draws can understate a shared disruption when several suppliers depend on one process input, region or equipment source. Correlation should be an explicit scenario assumption rather than inferred from the number of supplier names.

The next proposed synthetic experiment holds supplier means and marginal variation fixed while changing a common-shock component. Use the same realized yields for every policy at each seed. Evaluate delivered good units, unmet demand, expedite cost if modeled and total cost; include infeasible draws in the report rather than averaging only successful solves.

Keep training scenarios separate from held-out evaluation draws when choosing uncertainty-set sizes or adaptive rules. The adaptive model observes yield only, so it must not receive later demand or disruption information unavailable to its competitors. A lower simulated mean cost remains conditional on that information structure, solver status and recourse assumptions. Save the scenario generator and policy configuration with the scorecard.

## References

- https://doi.org/10.1287/opre.1030.0065
