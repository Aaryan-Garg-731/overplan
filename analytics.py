"""
OverPlan - Analytics Module
Calculates phase-by-phase bowling metrics (Powerplay, Middle, Death),
runs conceded, legal overs, wickets, and Empirical Bayes shrinkage
toward competition phase averages.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

PHASES = ["Powerplay", "Middle", "Death"]
PHASE_TARGET_OVERS = {"Powerplay": 6, "Middle": 9, "Death": 5}

# Default league baseline economy fallbacks if phase data is sparse
DEFAULT_PHASE_BASELINES = {
    "Powerplay": 7.40,
    "Middle": 7.80,
    "Death": 9.60,
}

# Default Empirical Bayes shrinkage weight in overs (pseudo-overs prior)
DEFAULT_SHRINKAGE_WEIGHT = 4.0

# Low sample threshold in overs
LOW_SAMPLE_THRESHOLD_OVERS = 3.0


def format_cricket_overs(legal_balls: int) -> str:
    """
    Format legal balls into traditional cricket notation (e.g. 14 balls -> '2.2 ov').
    """
    if legal_balls <= 0:
        return "0.0"
    completed = legal_balls // 6
    remainder = legal_balls % 6
    return f"{completed}.{remainder}"


def compute_competition_phase_averages(
    deliveries_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Compute competition-wide / dataset-wide totals and economy rate for each phase.
    """
    phase_stats = {}
    for phase in PHASES:
        sub = deliveries_df[deliveries_df["phase"] == phase]
        if sub.empty:
            phase_stats[phase] = {
                "legal_balls": 0,
                "overs": 0.0,
                "runs": 0,
                "wickets": 0,
                "economy": DEFAULT_PHASE_BASELINES.get(phase, 8.0),
            }
            continue

        legal_balls = int(sub["is_legal"].sum())
        runs = int(sub["bowler_conceded_runs"].sum())
        wickets = int(sub["is_bowler_wicket"].sum())
        exact_overs = legal_balls / 6.0

        if exact_overs > 0:
            econ = runs / exact_overs
        else:
            econ = DEFAULT_PHASE_BASELINES.get(phase, 8.0)

        phase_stats[phase] = {
            "legal_balls": legal_balls,
            "overs": round(exact_overs, 2),
            "runs": runs,
            "wickets": wickets,
            "economy": round(econ, 2),
        }

    return phase_stats


def compute_bowler_phase_metrics(
    deliveries_df: pd.DataFrame,
    shrinkage_weight: float = DEFAULT_SHRINKAGE_WEIGHT,
    low_sample_threshold: float = LOW_SAMPLE_THRESHOLD_OVERS,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Calculate raw and adjusted metrics for every bowler across Powerplay, Middle, Death, and Overall.
    Returns:
        (phase_metrics_df, competition_baselines, bowler_summary_df)
    """
    if deliveries_df.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    comp_baselines = compute_competition_phase_averages(deliveries_df)

    # Group by bowler and phase
    records = []
    unique_bowlers = sorted(deliveries_df["bowler"].unique().tolist())

    # Pre-group for fast aggregation
    grouped = deliveries_df.groupby(["bowler", "phase"]).agg(
        legal_balls=("is_legal", "sum"),
        total_balls=("is_legal", "count"),
        runs_conceded=("bowler_conceded_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        total_wickets=("is_wicket", "sum"),
        dot_balls=("bowler_conceded_runs", lambda s: ((s == 0) & (deliveries_df.loc[s.index, "is_legal"])).sum()),
    ).reset_index()

    # Fast lookup dictionary
    group_lookup = {}
    for _, row in grouped.iterrows():
        group_lookup[(row["bowler"], row["phase"])] = row

    for bowler in unique_bowlers:
        bowler_total_balls = 0
        bowler_total_runs = 0
        bowler_total_wickets = 0

        for phase in PHASES:
            baseline_econ = comp_baselines.get(phase, {}).get(
                "economy", DEFAULT_PHASE_BASELINES[phase]
            )
            key = (bowler, phase)

            if key in group_lookup:
                row = group_lookup[key]
                legal_balls = int(row["legal_balls"])
                runs = int(row["runs_conceded"])
                wickets = int(row["wickets"])
                dots = int(row["dot_balls"])
                exact_overs = legal_balls / 6.0
                formatted_overs = format_cricket_overs(legal_balls)

                if exact_overs > 0:
                    raw_econ = runs / exact_overs
                else:
                    raw_econ = baseline_econ

                # Empirical Bayes shrinkage:
                # E_adj = (runs + m * baseline_econ) / (overs + m)
                adj_econ = (runs + shrinkage_weight * baseline_econ) / (
                    exact_overs + shrinkage_weight
                )

                # Confidence label & sample warning
                if exact_overs == 0:
                    confidence = "No Phase History"
                    warning = "⚠️ No Data (League Average)"
                    has_low_sample = True
                elif exact_overs < low_sample_threshold:
                    confidence = f"Low Sample ({formatted_overs} ov)"
                    warning = f"⚠️ Low Sample (<{low_sample_threshold:g} ov)"
                    has_low_sample = True
                elif exact_overs < 8.0:
                    confidence = "Moderate Sample"
                    warning = "ℹ️ Moderate Sample"
                    has_low_sample = False
                else:
                    confidence = "High Sample"
                    warning = "✅ Sufficient Sample"
                    has_low_sample = False

                dot_pct = (dots / legal_balls * 100.0) if legal_balls > 0 else 0.0
                strike_rate = (legal_balls / wickets) if wickets > 0 else np.nan

                bowler_total_balls += legal_balls
                bowler_total_runs += runs
                bowler_total_wickets += wickets

            else:
                # Bowler has never bowled in this phase in the dataset
                legal_balls = 0
                runs = 0
                wickets = 0
                dots = 0
                exact_overs = 0.0
                formatted_overs = "0.0"
                raw_econ = np.nan
                # Shrinkage fallback = baseline economy
                adj_econ = baseline_econ
                confidence = "No Phase History"
                warning = "⚠️ No Data (League Baseline)"
                has_low_sample = True
                dot_pct = 0.0
                strike_rate = np.nan

            records.append({
                "bowler": bowler,
                "phase": phase,
                "legal_balls": legal_balls,
                "exact_overs": round(exact_overs, 3),
                "overs_formatted": formatted_overs,
                "runs_conceded": runs,
                "wickets": wickets,
                "raw_economy": round(raw_econ, 2) if not np.isnan(raw_econ) else None,
                "adjusted_economy": round(adj_econ, 2),
                "baseline_economy": baseline_econ,
                "shrinkage_overs_weight": shrinkage_weight,
                "confidence": confidence,
                "warning": warning,
                "is_low_sample": has_low_sample,
                "dot_pct": round(dot_pct, 1),
                "strike_rate": round(strike_rate, 1) if not np.isnan(strike_rate) else None,
            })

    phase_metrics_df = pd.DataFrame(records)

    # Compute overall bowler summary
    summary_records = []
    overall_grouped = deliveries_df.groupby("bowler").agg(
        total_legal=("is_legal", "sum"),
        total_runs=("bowler_conceded_runs", "sum"),
        total_wickets=("is_bowler_wicket", "sum"),
        total_matches=("source_name", "nunique"),
    ).reset_index()

    for _, row in overall_grouped.iterrows():
        b = row["bowler"]
        legal = int(row["total_legal"])
        runs = int(row["total_runs"])
        wkts = int(row["total_wickets"])
        matches = int(row["total_matches"])
        exact_ov = legal / 6.0
        econ = (runs / exact_ov) if exact_ov > 0 else 0.0
        sr = (legal / wkts) if wkts > 0 else np.nan

        # Phase adjusted economies
        b_phases = phase_metrics_df[phase_metrics_df["bowler"] == b]
        pp_adj = b_phases[b_phases["phase"] == "Powerplay"]["adjusted_economy"].values
        mid_adj = b_phases[b_phases["phase"] == "Middle"]["adjusted_economy"].values
        death_adj = b_phases[b_phases["phase"] == "Death"]["adjusted_economy"].values

        summary_records.append({
            "bowler": b,
            "matches": matches,
            "legal_balls": legal,
            "overs": round(exact_ov, 1),
            "overs_formatted": format_cricket_overs(legal),
            "runs_conceded": runs,
            "wickets": wkts,
            "overall_economy": round(econ, 2),
            "strike_rate": round(sr, 1) if not np.isnan(sr) else None,
            "pp_adjusted_econ": pp_adj[0] if len(pp_adj) > 0 else None,
            "mid_adjusted_econ": mid_adj[0] if len(mid_adj) > 0 else None,
            "death_adjusted_econ": death_adj[0] if len(death_adj) > 0 else None,
        })

    bowler_summary_df = pd.DataFrame(summary_records)

    return phase_metrics_df, comp_baselines, bowler_summary_df
