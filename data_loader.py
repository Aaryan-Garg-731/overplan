"""
OverPlan - Data Loader Module
Handles loading, parsing, validating, and extracting delivery records
from Cricsheet ball-by-ball JSON files and ZIP archives.
"""

import json
import zipfile
import io
import os
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

# Standard bowler-credited wicket types in official cricket rules
BOWLER_WICKET_KINDS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}

# Non-bowler-credited dismissals (must be excluded from bowler wicket tally)
NON_BOWLER_WICKET_KINDS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
    "timed out",
    "hit the ball twice",
}


def get_phase_from_over(over_zero_indexed: int) -> str:
    """
    Convert Cricsheet's 0-indexed over number to T20 match phase:
    - 0 to 5 (Overs 1-6): Powerplay
    - 6 to 14 (Overs 7-15): Middle
    - 15 to 19 (Overs 16-20): Death
    - 20+ (Super overs / extended): Death
    """
    if over_zero_indexed < 6:
        return "Powerplay"
    elif over_zero_indexed < 15:
        return "Middle"
    else:
        return "Death"


def parse_single_match_json(
    raw_data: Any, source_name: str = "unknown"
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """
    Parse a single Cricsheet match JSON object or string.
    Returns:
        (metadata_dict, deliveries_list, error_message)
    """
    if isinstance(raw_data, (str, bytes)):
        try:
            match = json.loads(raw_data)
        except Exception as e:
            return None, [], f"Invalid JSON syntax in {source_name}: {str(e)}"
    elif isinstance(raw_data, dict):
        match = raw_data
    else:
        return None, [], f"Unsupported data type for {source_name}: {type(raw_data)}"

    if not isinstance(match, dict):
        return None, [], f"Expected JSON dictionary at root of {source_name}"

    info = match.get("info", {})
    if not isinstance(info, dict):
        return None, [], f"Missing or invalid 'info' dictionary in {source_name}"

    # Extract metadata safely
    match_type = info.get("match_type", "Unknown")
    scheduled_overs = info.get("overs", 20)
    dates = info.get("dates", [])
    match_date = str(dates[0]) if dates else "Unknown"
    
    # Competition / Event
    event = info.get("event")
    event_name = "Independent Match"
    if isinstance(event, dict):
        event_name = event.get("name", "Independent Match")
    elif isinstance(event, str) and event.strip():
        event_name = event.strip()

    season = str(info.get("season", "Unknown"))
    gender = info.get("gender", "male")
    teams = info.get("teams", [])
    venue = info.get("venue", "Unknown")
    city = info.get("city", venue)
    
    # Outcome
    outcome = info.get("outcome", {})
    winner = outcome.get("winner", "No Result / Tie")
    by_dict = outcome.get("by", {})
    by_summary = ""
    if by_dict:
        k = list(by_dict.keys())[0]
        by_summary = f"by {by_dict[k]} {k}"

    meta = {
        "source_name": source_name,
        "match_type": match_type,
        "scheduled_overs": scheduled_overs,
        "date": match_date,
        "event": event_name,
        "season": season,
        "gender": gender,
        "teams": teams,
        "venue": venue,
        "city": city,
        "winner": winner,
        "result_detail": by_summary,
    }

    innings_list = match.get("innings", [])
    if not isinstance(innings_list, list):
        return meta, [], f"Innings field is not a list in {source_name}"

    deliveries = []

    for inn_idx, innings in enumerate(innings_list):
        if not isinstance(innings, dict):
            continue
        
        # Super overs can be flagged or handled
        is_super_over = innings.get("super_over", False)
        if is_super_over:
            continue  # Exclude super overs from standard 20-over bowling allocation

        batting_team = innings.get("team", "Unknown")
        # Identify bowling team from match teams
        bowling_team = "Unknown"
        if len(teams) >= 2:
            bowling_team = teams[1] if batting_team == teams[0] else teams[0]
        elif len(teams) == 1:
            bowling_team = "Opponent"

        overs_data = innings.get("overs", [])
        if not isinstance(overs_data, list):
            continue

        for over_item in overs_data:
            if not isinstance(over_item, dict):
                continue
            
            raw_over = over_item.get("over")
            if raw_over is None:
                continue

            try:
                over_num = int(raw_over)
            except (ValueError, TypeError):
                continue

            # Standard phase conversion
            phase = get_phase_from_over(over_num)
            over_display = over_num + 1  # 1-indexed over number (1-20)

            ball_deliveries = over_item.get("deliveries", [])
            if not isinstance(ball_deliveries, list):
                continue

            for ball_idx, deliv in enumerate(ball_deliveries):
                if not isinstance(deliv, dict):
                    continue

                bowler = deliv.get("bowler")
                batter = deliv.get("batter")
                non_striker = deliv.get("non_striker")

                if not bowler:
                    continue

                # Runs parsing
                runs_dict = deliv.get("runs", {})
                batter_runs = runs_dict.get("batter", 0) if isinstance(runs_dict, dict) else 0
                total_runs = runs_dict.get("total", 0) if isinstance(runs_dict, dict) else 0

                # Extras parsing
                extras_dict = deliv.get("extras", {})
                if not isinstance(extras_dict, dict):
                    extras_dict = {}

                wides = extras_dict.get("wides", 0)
                noballs = extras_dict.get("noballs", 0)
                byes = extras_dict.get("byes", 0)
                legbyes = extras_dict.get("legbyes", 0)
                penalty = extras_dict.get("penalty", 0)

                # Legal delivery definition: not wide and not no-ball
                is_legal = (wides == 0) and (noballs == 0)

                # Bowler-conceded runs: batter runs + wides + no-balls
                # Explicitly exclude byes, legbyes, and penalty runs
                bowler_conceded = int(batter_runs) + int(wides) + int(noballs)

                # Dismissal parsing
                wickets_info = deliv.get("wickets", [])
                is_wicket = False
                is_bowler_wicket = False
                dismissal_kind = None
                player_out = None

                if isinstance(wickets_info, list) and len(wickets_info) > 0:
                    for w in wickets_info:
                        if isinstance(w, dict):
                            is_wicket = True
                            k = str(w.get("kind", "")).lower().strip()
                            dismissal_kind = k
                            player_out = w.get("player_out")
                            if k in BOWLER_WICKET_KINDS:
                                is_bowler_wicket = True

                deliveries.append({
                    "source_name": source_name,
                    "date": match_date,
                    "event": event_name,
                    "season": season,
                    "gender": gender,
                    "match_type": match_type,
                    "scheduled_overs": scheduled_overs,
                    "innings": inn_idx + 1,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over_0indexed": over_num,
                    "over_number": over_display,
                    "phase": phase,
                    "ball_index": ball_idx + 1,
                    "bowler": bowler,
                    "batter": batter,
                    "non_striker": non_striker,
                    "batter_runs": int(batter_runs),
                    "wides": int(wides),
                    "noballs": int(noballs),
                    "byes": int(byes),
                    "legbyes": int(legbyes),
                    "penalty": int(penalty),
                    "bowler_conceded_runs": bowler_conceded,
                    "is_legal": is_legal,
                    "is_wicket": is_wicket,
                    "is_bowler_wicket": is_bowler_wicket,
                    "dismissal_kind": dismissal_kind,
                    "player_out": player_out,
                })

    return meta, deliveries, None


def load_matches_from_files(
    file_inputs: List[Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load and parse a list of file inputs (file paths, stream buffers, or UploadedFiles).
    Handles individual .json files and .zip archives.
    Returns:
        (matches_df, deliveries_df, parsing_warnings)
    """
    all_matches = []
    all_deliveries = []
    warnings = []

    for item in file_inputs:
        filename = getattr(item, "name", str(item))
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".zip":
            # Extract and parse all JSON files inside ZIP
            try:
                # Support both path string and bytes/buffer
                if isinstance(item, (str, os.PathLike)):
                    z = zipfile.ZipFile(item)
                else:
                    item.seek(0)
                    z = zipfile.ZipFile(io.BytesIO(item.read()))

                with z:
                    for inner_name in z.namelist():
                        if inner_name.endswith(".json") and not inner_name.startswith("__MACOSX"):
                            try:
                                raw_bytes = z.read(inner_name)
                                meta, delivs, err = parse_single_match_json(
                                    raw_bytes, source_name=inner_name
                                )
                                if err:
                                    warnings.append(err)
                                elif meta:
                                    all_matches.append(meta)
                                    all_deliveries.extend(delivs)
                            except Exception as e:
                                warnings.append(
                                    f"Failed parsing {inner_name} in {filename}: {str(e)}"
                                )
            except Exception as e:
                warnings.append(f"Failed opening ZIP file {filename}: {str(e)}")

        elif ext == ".json":
            try:
                if isinstance(item, (str, os.PathLike)):
                    with open(item, "r", encoding="utf-8") as f:
                        raw_content = f.read()
                else:
                    item.seek(0)
                    raw_content = item.read().decode("utf-8", errors="replace")

                meta, delivs, err = parse_single_match_json(
                    raw_content, source_name=filename
                )
                if err:
                    warnings.append(err)
                elif meta:
                    all_matches.append(meta)
                    all_deliveries.extend(delivs)
            except Exception as e:
                warnings.append(f"Error reading JSON file {filename}: {str(e)}")
        else:
            warnings.append(f"Ignored non-JSON/non-ZIP file: {filename}")

    matches_df = pd.DataFrame(all_matches)
    deliveries_df = pd.DataFrame(all_deliveries)

    return matches_df, deliveries_df, warnings


def load_demo_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load the bundled Cricsheet demo matches from data/demo directory.
    """
    demo_dir = os.path.join(os.path.dirname(__file__), "data", "demo")
    if not os.path.exists(demo_dir):
        return pd.DataFrame(), pd.DataFrame(), ["Demo data directory not found."]

    json_files = [
        os.path.join(demo_dir, f)
        for f in os.listdir(demo_dir)
        if f.endswith(".json")
    ]

    if not json_files:
        return pd.DataFrame(), pd.DataFrame(), ["No demo JSON match files found."]

    return load_matches_from_files(json_files)
