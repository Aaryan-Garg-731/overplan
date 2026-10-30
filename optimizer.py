"""
OverPlan - Allocation Optimizer Module
Formulates and solves the constrained integer optimization problem to allocate
exactly 20 overs (6 Powerplay, 9 Middle, 5 Death) across a chosen roster of bowlers,
minimizing expected runs conceded while adhering to standard T20 constraints.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

PHASES = ["Powerplay", "Middle", "Death"]
PHASE_TARGETS = {"Powerplay": 6, "Middle": 9, "Death": 5}
TOTAL_OVERS_REQUIRED = 20
MAX_OVERS_PER_BOWLER = 4


class AllocationResult:
    def __init__(
        self,
        is_feasible: bool,
        message: str,
        total_estimated_runs: float = 0.0,
        allocation_matrix: Optional[Dict[str, Dict[str, int]]] = None,
        allocation_df: Optional[pd.DataFrame] = None,
        phase_runs: Optional[Dict[str, float]] = None,
        diagnostics: Optional[List[str]] = None,
    ):
        self.is_feasible = is_feasible
        self.message = message
        self.total_estimated_runs = round(total_estimated_runs, 2)
        self.projected_economy = round(total_estimated_runs / TOTAL_OVERS_REQUIRED, 2) if is_feasible else 0.0
        self.allocation_matrix = allocation_matrix or {}
        self.allocation_df = allocation_df if allocation_df is not None else pd.DataFrame()
        self.phase_runs = phase_runs or {}
        self.diagnostics = diagnostics or []


def validate_roster_feasibility(
    roster: List[str],
    bowler_limits: Optional[Dict[str, int]] = None,
    phase_permissions: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[bool, List[str]]:
    """
    Perform pre-flight feasibility checks before calling the solver.
    Returns (is_valid, list_of_error_reasons).
    """
    errors = []
    num_bowlers = len(roster)

    if num_bowlers < 5:
        errors.append(
            f"Insufficient bowlers selected: {num_bowlers} selected. At least 5 bowlers "
            f"are mathematically required to complete 20 overs (maximum {MAX_OVERS_PER_BOWLER} overs per bowler)."
        )

    limits = bowler_limits or {b: MAX_OVERS_PER_BOWLER for b in roster}
    total_capacity = sum(limits.get(b, MAX_OVERS_PER_BOWLER) for b in roster)

    if total_capacity < TOTAL_OVERS_REQUIRED:
        errors.append(
            f"Total available bowling capacity is {total_capacity} overs, but a full T20 "
            f"innings requires {TOTAL_OVERS_REQUIRED} overs. Increase bowler limits or add more bowlers."
        )

    # Check phase capacities
    perms = phase_permissions or {b: set(PHASES) for b in roster}
    for phase, target in PHASE_TARGETS.items():
        phase_cap = sum(
            min(limits.get(b, MAX_OVERS_PER_BOWLER), target)
            for b in roster
            if phase in perms.get(b, set(PHASES))
        )
        if phase_cap < target:
            errors.append(
                f"Insufficient capacity for {phase} overs: Available bowlers can only bowl "
                f"up to {phase_cap} overs in {phase}, but exactly {target} overs are required."
            )

    return len(errors) == 0, errors


def solve_optimal_allocation(
    roster: List[str],
    phase_metrics_df: pd.DataFrame,
    bowler_limits: Optional[Dict[str, int]] = None,
    phase_permissions: Optional[Dict[str, Set[str]]] = None,
    comp_baselines: Optional[Dict[str, Dict[str, float]]] = None,
) -> AllocationResult:
    """
    Solve for the optimal 20-over allocation using Mixed-Integer Linear Programming.
    Objective: Minimize sum of (overs * adjusted_phase_economy).
    Constraints:
      1) PP overs == 6
      2) Middle overs == 9
      3) Death overs == 5
      4) Total overs per bowler <= limit (default 4)
      5) Over counts are non-negative integers
      6) Bowler phase exclusions respected
    """
    if not roster:
        return AllocationResult(
            is_feasible=False,
            message="No bowlers provided in roster.",
            diagnostics=["Please select at least 5 bowlers from the roster picker."],
        )

    # Pre-check feasibility
    is_valid, validation_errors = validate_roster_feasibility(
        roster, bowler_limits, phase_permissions
    )
    if not is_valid:
        return AllocationResult(
            is_feasible=False,
            message="Roster constraints cannot be satisfied.",
            diagnostics=validation_errors,
        )

    limits = bowler_limits or {b: MAX_OVERS_PER_BOWLER for b in roster}
    perms = phase_permissions or {b: set(PHASES) for b in roster}

    n_bowlers = len(roster)
    n_phases = len(PHASES)
    n_vars = n_bowlers * n_phases

    # Build cost vector c (adjusted economy rates)
    c = np.zeros(n_vars)
    cost_lookup = {}

    for i, bowler in enumerate(roster):
        b_metrics = phase_metrics_df[phase_metrics_df["bowler"] == bowler]
        for j, phase in enumerate(PHASES):
            var_idx = i * n_phases + j
            p_metric = b_metrics[b_metrics["phase"] == phase]

            if not p_metric.empty and pd.notna(p_metric.iloc[0]["adjusted_economy"]):
                econ = float(p_metric.iloc[0]["adjusted_economy"])
            elif comp_baselines and phase in comp_baselines:
                econ = float(comp_baselines[phase]["economy"])
            else:
                econ = 8.0

            # If bowler is forbidden from bowling in this phase, set high penalty
            if phase not in perms.get(bowler, set(PHASES)):
                econ = 1e6

            c[var_idx] = econ
            cost_lookup[(bowler, phase)] = econ

    # Construct constraints:
    # 1. Phase totals (3 equality constraints)
    A_phase = np.zeros((n_phases, n_vars))
    for j, phase in enumerate(PHASES):
        for i in range(n_bowlers):
            A_phase[j, i * n_phases + j] = 1.0
    b_l_phase = np.array([PHASE_TARGETS[p] for p in PHASES], dtype=float)
    b_u_phase = np.array([PHASE_TARGETS[p] for p in PHASES], dtype=float)

    # 2. Bowler maximum limits (n_bowlers inequality constraints)
    A_bowler = np.zeros((n_bowlers, n_vars))
    for i in range(n_bowlers):
        A_bowler[i, i * n_phases : (i + 1) * n_phases] = 1.0
    b_l_bowler = np.zeros(n_bowlers, dtype=float)
    b_u_bowler = np.array([float(limits.get(b, MAX_OVERS_PER_BOWLER)) for b in roster])

    A = np.vstack([A_phase, A_bowler])
    b_l = np.concatenate([b_l_phase, b_l_bowler])
    b_u = np.concatenate([b_u_phase, b_u_bowler])

    constraints = LinearConstraint(A, b_l, b_u)

    # Upper bound for each individual variable:
    # cannot exceed bowler's limit or phase target, and 0 if excluded
    var_upper_bounds = np.zeros(n_vars)
    for i, bowler in enumerate(roster):
        b_max = limits.get(bowler, MAX_OVERS_PER_BOWLER)
        for j, phase in enumerate(PHASES):
            var_idx = i * n_phases + j
            if phase in perms.get(bowler, set(PHASES)):
                var_upper_bounds[var_idx] = min(b_max, PHASE_TARGETS[phase])
            else:
                var_upper_bounds[var_idx] = 0

    bounds = Bounds(lb=np.zeros(n_vars), ub=var_upper_bounds)
    integrality = np.ones(n_vars)  # All integer variables

    # Run SciPy MILP
    try:
        res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
    except Exception as e:
        return AllocationResult(
            is_feasible=False,
            message=f"MILP solver error: {str(e)}",
            diagnostics=["Solver encountered an unexpected numerical or environment issue."],
        )

    if not res.success:
        return AllocationResult(
            is_feasible=False,
            message=f"Optimization failed: {res.message}",
            diagnostics=[
                "The solver could not satisfy all over constraints with the current roster limits.",
                "Try relaxing phase restrictions or increasing individual bowler over limits.",
            ],
        )

    # Parse solution
    raw_sol = np.round(res.x).astype(int).reshape((n_bowlers, n_phases))
    total_est_runs = 0.0
    phase_runs = {p: 0.0 for p in PHASES}
    allocation_matrix = {}
    rows = []

    for i, bowler in enumerate(roster):
        b_alloc = {}
        bowler_total_ov = int(raw_sol[i].sum())
        bowler_est_runs = 0.0

        b_metrics = phase_metrics_df[phase_metrics_df["bowler"] == bowler]

        # Extract career / sample statistics
        total_career_wickets = int(b_metrics["wickets"].sum()) if not b_metrics.empty else 0
        total_career_overs = float(b_metrics["exact_overs"].sum()) if not b_metrics.empty else 0.0

        # Sample size confidence indicator
        if total_career_overs == 0:
            sample_confidence = "⚠️ Default (No historical data)"
        elif total_career_overs < 4.0:
            sample_confidence = "⚠️ Low sample (<4 ov)"
        elif total_career_overs < 10.0:
            sample_confidence = "ℹ️ Moderate sample"
        else:
            sample_confidence = "✅ High sample (10+ ov)"

        for j, phase in enumerate(PHASES):
            overs_in_phase = int(raw_sol[i, j])
            b_alloc[phase] = overs_in_phase
            econ = cost_lookup.get((bowler, phase), 8.0)
            if econ >= 1e5:  # excluded phase
                econ = 8.0
            phase_cost = overs_in_phase * econ
            bowler_est_runs += phase_cost
            phase_runs[phase] += phase_cost

        total_est_runs += bowler_est_runs
        allocation_matrix[bowler] = b_alloc

        # Tactical description
        pp_ov = b_alloc["Powerplay"]
        mid_ov = b_alloc["Middle"]
        death_ov = b_alloc["Death"]

        roles = []
        if pp_ov >= 2:
            roles.append(f"Powerplay Anchor ({pp_ov} ov)")
        elif pp_ov == 1:
            roles.append("PP support")
        if mid_ov >= 3:
            roles.append(f"Middle-over Controller ({mid_ov} ov)")
        elif mid_ov >= 1:
            roles.append(f"Middle ({mid_ov} ov)")
        if death_ov >= 2:
            roles.append(f"Death Specialist ({death_ov} ov)")
        elif death_ov == 1:
            roles.append("Death support")

        role_desc = ", ".join(roles) if roles else "Bench / Unallocated"

        pp_econ = cost_lookup.get((bowler, "Powerplay"), 8.0)
        mid_econ = cost_lookup.get((bowler, "Middle"), 8.0)
        death_econ = cost_lookup.get((bowler, "Death"), 8.0)

        rows.append({
            "Bowler": bowler,
            "PP (Overs 1-6)": pp_ov,
            "Middle (Overs 7-15)": mid_ov,
            "Death (Overs 16-20)": death_ov,
            "Total Overs": bowler_total_ov,
            "Est. Runs Conceded": round(bowler_est_runs, 1),
            "PP Adj Econ": round(pp_econ, 2),
            "Middle Adj Econ": round(mid_econ, 2),
            "Death Adj Econ": round(death_econ, 2),
            "Career Wickets": total_career_wickets,
            "Career Overs": round(total_career_overs, 1),
            "Sample Confidence": sample_confidence,
            "Suggested Role": role_desc,
        })

    allocation_df = pd.DataFrame(rows)
    # Sort bowlers: allocated bowlers first by total overs, then estimated runs
    allocation_df = allocation_df.sort_values(
        by=["Total Overs", "Est. Runs Conceded"], ascending=[False, True]
    ).reset_index(drop=True)

    return AllocationResult(
        is_feasible=True,
        message="Optimal 20-over allocation computed successfully.",
        total_estimated_runs=total_est_runs,
        allocation_matrix=allocation_matrix,
        allocation_df=allocation_df,
        phase_runs=phase_runs,
        diagnostics=[],
    )
