"""
OverPlan - Unit and Integration Tests
Tests data loading, cricket accounting rules, Empirical Bayes shrinkage,
and 20-over MILP allocation.
"""

import unittest
import os
import io
import json
import zipfile
import pandas as pd
import numpy as np

import data_loader
import analytics
import optimizer


class TestDataLoader(unittest.TestCase):
    def test_phase_mapping(self):
        # Cricsheet 0-indexed overs
        # 0 to 5 -> Powerplay (overs 1 to 6)
        for ov in range(6):
            self.assertEqual(data_loader.get_phase_from_over(ov), "Powerplay")
        # 6 to 14 -> Middle (overs 7 to 15)
        for ov in range(6, 15):
            self.assertEqual(data_loader.get_phase_from_over(ov), "Middle")
        # 15 to 19 -> Death (overs 16 to 20)
        for ov in range(15, 20):
            self.assertEqual(data_loader.get_phase_from_over(ov), "Death")

    def test_bowler_conceded_runs_and_extras(self):
        # Match with custom deliveries testing cricket accounting rules
        mock_match = {
            "info": {
                "match_type": "T20",
                "overs": 20,
                "teams": ["Team A", "Team B"],
                "dates": ["2026-01-01"],
            },
            "innings": [
                {
                    "team": "Team A",
                    "overs": [
                        {
                            "over": 0,
                            "deliveries": [
                                # 1. Normal single off the bat
                                {"bowler": "Bowler 1", "batter": "Batter 1", "runs": {"batter": 1, "extras": 0, "total": 1}},
                                # 2. Wide ball (1 extra run, not legal, charged to bowler)
                                {"bowler": "Bowler 1", "batter": "Batter 1", "extras": {"wides": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}},
                                # 3. No-ball with 2 off the bat (1 noball + 2 batter = 3 runs charged to bowler, not legal)
                                {"bowler": "Bowler 1", "batter": "Batter 1", "extras": {"noballs": 1}, "runs": {"batter": 2, "extras": 1, "total": 3}},
                                # 4. Leg-bye: 4 runs (legal ball, 0 runs charged to bowler!)
                                {"bowler": "Bowler 1", "batter": "Batter 1", "extras": {"legbyes": 4}, "runs": {"batter": 0, "extras": 4, "total": 4}},
                                # 5. Bye: 1 run (legal ball, 0 runs charged to bowler!)
                                {"bowler": "Bowler 1", "batter": "Batter 1", "extras": {"byes": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}},
                                # 6. Bowler wicket (bowled)
                                {"bowler": "Bowler 1", "batter": "Batter 1", "runs": {"batter": 0, "extras": 0, "total": 0}, "wickets": [{"kind": "bowled", "player_out": "Batter 1"}]},
                                # 7. Non-bowler wicket (run out)
                                {"bowler": "Bowler 1", "batter": "Batter 2", "runs": {"batter": 1, "extras": 0, "total": 1}, "wickets": [{"kind": "run out", "player_out": "Batter 1"}]},
                            ],
                        }
                    ],
                }
            ],
        }

        meta, delivs, err = data_loader.parse_single_match_json(mock_match, "test_mock")
        self.assertIsNone(err)
        self.assertEqual(len(delivs), 7)

        df = pd.DataFrame(delivs)
        # Check legal deliveries: deliveries 1, 4, 5, 6, 7 are legal (5 total).
        # deliveries 2 (wide) and 3 (noball) are illegal.
        self.assertEqual(df["is_legal"].sum(), 5)

        # Check bowler runs:
        # Deliv 1: 1
        # Deliv 2 (wide): 1
        # Deliv 3 (noball + 2 batter): 3
        # Deliv 4 (legbyes 4): 0
        # Deliv 5 (byes 1): 0
        # Deliv 6 (bowled): 0
        # Deliv 7 (run out, 1 batter): 1
        # Total bowler runs = 1 + 1 + 3 + 0 + 0 + 0 + 1 = 6
        self.assertEqual(df["bowler_conceded_runs"].sum(), 6)

        # Total wickets = 2, but bowler wickets = 1 (bowled only; run out excluded!)
        self.assertEqual(df["is_wicket"].sum(), 2)
        self.assertEqual(df["is_bowler_wicket"].sum(), 1)


class TestAnalytics(unittest.TestCase):
    def test_cricket_overs_formatting(self):
        self.assertEqual(analytics.format_cricket_overs(0), "0.0")
        self.assertEqual(analytics.format_cricket_overs(6), "1.0")
        self.assertEqual(analytics.format_cricket_overs(7), "1.1")
        self.assertEqual(analytics.format_cricket_overs(23), "3.5")
        self.assertEqual(analytics.format_cricket_overs(24), "4.0")

    def test_empirical_bayes_shrinkage(self):
        # Create deliveries with 1 over (6 balls) for 2 runs (raw econ = 2.00)
        # Baseline econ = 8.00, weight m = 4.0
        # Adjusted econ = (2 + 4 * 8.0) / (1.0 + 4.0) = (2 + 32) / 5 = 34 / 5 = 6.80
        deliv_records = []
        for _ in range(6):
            deliv_records.append({
                "bowler": "Fluke Bowler",
                "phase": "Powerplay",
                "is_legal": True,
                "bowler_conceded_runs": 0 if _ < 4 else 1, # 2 runs total
                "is_bowler_wicket": 0,
                "is_wicket": 0,
                "source_name": "match1",
            })
        
        # Add league deliveries to establish baseline
        for _ in range(60): # 10 overs, 80 runs -> 8.0 baseline
            deliv_records.append({
                "bowler": "League Bowler",
                "phase": "Powerplay",
                "is_legal": True,
                "bowler_conceded_runs": 1 if _ % 3 != 0 else 2,
                "is_bowler_wicket": 0,
                "is_wicket": 0,
                "source_name": "match1",
            })

        df = pd.DataFrame(deliv_records)
        phase_df, baselines, summary_df = analytics.compute_bowler_phase_metrics(
            df, shrinkage_weight=4.0, low_sample_threshold=3.0
        )

        fluke = phase_df[(phase_df["bowler"] == "Fluke Bowler") & (phase_df["phase"] == "Powerplay")].iloc[0]
        self.assertEqual(fluke["exact_overs"], 1.0)
        self.assertEqual(fluke["runs_conceded"], 2)
        self.assertEqual(fluke["raw_economy"], 2.00)
        # Adjusted economy should be pulled up toward baseline (> 2.00)
        self.assertGreater(fluke["adjusted_economy"], 2.00)
        self.assertTrue(fluke["is_low_sample"])


class TestOptimizer(unittest.TestCase):
    def test_optimal_allocation_satisfies_all_constraints(self):
        # 6 bowlers with different strengths across phases
        bowlers = [f"Bowler {i}" for i in range(1, 7)]
        records = []
        for i, b in enumerate(bowlers):
            records.append({"bowler": b, "phase": "Powerplay", "adjusted_economy": 7.0 + i * 0.3, "exact_overs": 5.0, "wickets": 2})
            records.append({"bowler": b, "phase": "Middle", "adjusted_economy": 7.5 - i * 0.2, "exact_overs": 5.0, "wickets": 3})
            records.append({"bowler": b, "phase": "Death", "adjusted_economy": 9.0 + (i % 2) * 1.5, "exact_overs": 5.0, "wickets": 4})

        phase_df = pd.DataFrame(records)
        res = optimizer.solve_optimal_allocation(
            roster=bowlers,
            phase_metrics_df=phase_df,
        )

        self.assertTrue(res.is_feasible)
        alloc_df = res.allocation_df

        # Check total overs allocated across entire team is exactly 20
        self.assertEqual(alloc_df["Total Overs"].sum(), 20)
        # Check phase sums
        self.assertEqual(alloc_df["PP (Overs 1-6)"].sum(), 6)
        self.assertEqual(alloc_df["Middle (Overs 7-15)"].sum(), 9)
        self.assertEqual(alloc_df["Death (Overs 16-20)"].sum(), 5)
        # Check no bowler exceeds 4 overs
        self.assertTrue((alloc_df["Total Overs"] <= 4).all())

    def test_infeasible_when_fewer_than_five_bowlers(self):
        bowlers = ["Bowler 1", "Bowler 2", "Bowler 3", "Bowler 4"]
        phase_df = pd.DataFrame([
            {"bowler": b, "phase": p, "adjusted_economy": 8.0, "exact_overs": 5.0, "wickets": 2}
            for b in bowlers for p in ["Powerplay", "Middle", "Death"]
        ])
        res = optimizer.solve_optimal_allocation(
            roster=bowlers,
            phase_metrics_df=phase_df,
        )
        self.assertFalse(res.is_feasible)
        self.assertTrue(any("At least 5 bowlers" in d for d in res.diagnostics))


if __name__ == "__main__":
    unittest.main()
