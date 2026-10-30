# OverPlan 🏏

**OverPlan** is an open-source cricket decision-support web application for T20 captains, bowling coaches, and performance analysts. It ingests historical ball-by-ball match data from [Cricsheet](https://cricsheet.org/), evaluates bowler performance across T20 phases (**Powerplay**, **Middle**, and **Death**), and uses **Empirical Bayes shrinkage** and **Mixed-Integer Linear Programming (MILP)** to generate a data-backed 20-over bowling allocation.

---

> ### ⚠️ Decision-Support Disclaimer
> OverPlan is a **decision-support prototype** grounded in historical ball-by-ball frequency data. Its recommendations represent baseline statistical expectations rather than performance guarantees or causal claims. In live matches, critical factors such as pitch condition, weather, dew, batter handedness matchups (LHB vs RHB), wicket-taking leverage, game situation, and the non-consecutive over rule must guide real-time tactical decisions.

---

## The Problem & Intended Users

### The Problem
In modern 20-over T20 cricket, assigning 20 overs across five or more bowlers is one of the captain's most consequential tactical challenges:
- **Phase Specialization**: A bowler with an impressive overall economy rate might excel in the middle overs but prove disastrous at the death.
- **Small-Sample Distortion**: Traditional economy rates severely mislead when sample sizes are small (e.g. 1 over for 2 runs looks elite at 2.00 RPO, while 1 over for 18 runs looks unplayable at 18.00 RPO).
- **Hard Optimization Constraints**: Captains must satisfy strict rules (exactly 20 overs, 6 in the Powerplay, 9 in the Middle, 5 at the Death, with no bowler exceeding 4 overs) while minimizing expected runs conceded.

### Intended Users
- **Cricket Captains & Team Management**: To construct pre-match bowling plans and contingency scenarios.
- **Performance & Data Analysts**: To assess squad strengths across phases and communicate clear, data-driven rationale to coaching staff.
- **Cricket Enthusiasts & Researchers**: To explore empirical phase dynamics and constrained optimization in cricket analytics.

---

## Key Features

1. **Robust Cricsheet Data Ingestion**:
   - Ingests individual match `.json` files and `.zip` archives.
   - Bundles a real 25-match international T20 demo dataset directly from Cricsheet.
   - Graceful error recovery: skips malformed entries and reports file-level warnings without crashing.
   - Clean empty state with instructions and sample download capabilities.

2. **Accurate Cricket Accounting Rules**:
   - **Legal Balls Only**: Deliveries with wides and no-balls are flagged as illegal deliveries and excluded from completed over tallies.
   - **Bowler-Conceded Runs**: Strictly tallies batter runs plus wides and no-balls; byes, leg-byes, and penalty runs are properly excluded.
   - **Bowler-Credited Wickets**: Isolates bowled, caught, caught and bowled, lbw, stumped, and hit wicket dismissals. Excludes run-outs, obstructing the field, and retirements.
   - **Correct Phase Boundaries**:
     - **Powerplay**: Overs 1–6 (Cricsheet 0-indexed overs 0 to 5)
     - **Middle**: Overs 7–15 (Cricsheet 0-indexed overs 6 to 14)
     - **Death**: Overs 16–20 (Cricsheet 0-indexed overs 15 to 19)

3. **Empirical Bayes Shrinkage**:
   - Mitigates extreme small-sample noise by regressing bowler phase economy rates toward the competition phase baseline.
   - Configurable prior weight parameter ($m$, default 4.0 overs).
   - Clear confidence indicators: *No History / League Default*, *Low Sample (<3 ov)*, *Moderate Sample (3–8 ov)*, and *Sufficient Sample (8+ ov)*.
   - Displays both raw and adjusted figures side-by-side with plain-language explanations.

4. **MILP Constrained 20-Over Allocation**:
   - Formulates an exact Mixed-Integer Linear Program solved via `scipy.optimize.milp` (HiGHS solver).
   - Enforces standard T20 constraints: exactly 6 PP overs, 9 Middle overs, 5 Death overs, and $\le 4$ overs per bowler.
   - Supports custom bowler limits (e.g. limiting part-timers to 1–2 overs) and phase eligibility restrictions.
   - Descriptive wicket reporting kept distinct from runs conceded (no arbitrary combined scores).
   - Feasibility pre-checks with diagnostic remedies when rosters cannot satisfy constraints.

5. **Polished Analyst Dashboard**:
   - Dark navy theme (`#080e1a`) with teal (`#00f5d4`) and lime accents.
   - High-contrast interactive Plotly visualizations.
   - Responsive layout suited for desktops and mobile devices.

---

## Analytical Methodology

### 1. Empirical Bayes Shrinkage Formula

$$\mu_p = \frac{\sum_{b} \text{Runs}_{b, p}}{\sum_{b} \text{Overs}_{b, p}}$$

$$E_{\text{adjusted}}(b, p) = \frac{\text{Runs}_{b, p} + m \cdot \mu_p}{\text{Overs}_{b, p} + m} = \left(\frac{\text{Overs}_{b, p}}{\text{Overs}_{b, p} + m}\right) E_{\text{raw}}(b, p) + \left(\frac{m}{\text{Overs}_{b, p} + m}\right) \mu_p$$

- When a bowler has no phase history ($\text{Overs} = 0$), $E_{\text{adjusted}} = \mu_p$ (league phase norm).
- As sample size increases, the bowler's empirical track record progressively dominates the prior.

### 2. Allocation Optimization Model

Let $x_{b, p} \in \{0, 1, 2, 3, 4\}$ be the integer number of overs allocated to bowler $b$ in phase $p \in \{\text{Powerplay}, \text{Middle}, \text{Death}\}$.

$$\min \sum_{b \in \text{Roster}} \sum_{p \in \{\text{PP}, \text{MID}, \text{DEATH}\}} x_{b, p} \cdot E_{\text{adjusted}}(b, p)$$

Subject to:
$$\sum_{b \in \text{Roster}} x_{b, \text{PP}} = 6$$
$$\sum_{b \in \text{Roster}} x_{b, \text{MID}} = 9$$
$$\sum_{b \in \text{Roster}} x_{b, \text{DEATH}} = 5$$
$$\sum_{p \in \text{Phases}} x_{b, p} \le 4 \quad \forall b \in \text{Roster}$$
$$x_{b, p} \in \mathbb{Z}_{\ge 0}$$

---

## Data Source, Attribution & Licensing

### Source
Match data is sourced from **Cricsheet** ([cricsheet.org](https://cricsheet.org/)), which publishes ball-by-ball data in JSON format:
- Downloads: https://cricsheet.org/downloads/
- JSON Format Specification: https://cricsheet.org/format/json/
- Licensing Information: https://cricsheet.org/licensing/

### Attribution Notice
> Data provided by **Cricsheet** ([cricsheet.org](https://cricsheet.org/)), licensed under the **Open Data Commons Attribution License (ODC-By v1.0)**.
> Full license available at [https://opendatacommons.org/licenses/by/1-0/](https://opendatacommons.org/licenses/by/1-0/) and included in `data/demo/LICENSE.txt`.

### How to Provide Your Own Data
1. Visit https://cricsheet.org/downloads/ and download any T20 tournament archive (e.g. `t20s_json.zip`, `ipl_json.zip`, `bbl_json.zip`) or individual match JSON files.
2. Launch OverPlan and choose **Upload Custom Files (.json / .zip)** in the sidebar.
3. Drag and drop your `.json` match files or `.zip` archives. OverPlan parses and aggregates all valid matches automatically.

---

## Installation & Setup

### Prerequisites
- Python 3.9+ (tested on Python 3.10 – 3.15)
- Standard virtual environment tool (`venv` or `conda`)

### 1. Clone or Navigate to the Repository
```bash
cd overplan
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
The dashboard will open automatically in your default browser at `http://localhost:8501`.

### 5. Run the Test Suite
```bash
python -m unittest tests/test_overplan.py
```

---

## Project Structure

```
overplan/
├── app.py                      # Main Streamlit web application & analyst UI
├── data_loader.py              # Cricsheet JSON/ZIP parser, validation & legal ball logic
├── analytics.py                # Phase grouping, economy rates & Empirical Bayes shrinkage
├── optimizer.py                # MILP optimization model & feasibility diagnostics
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License for OverPlan software
├── README.md                   # Project documentation & methodology guide
├── data/
│   └── demo/                   # Real Cricsheet sample data
│       ├── 1528722.json        # Australia vs New Zealand
│       ├── 1529231.json        # India vs West Indies
│       ├── 1539432.json        # New Zealand vs India
│       ├── ...                 # Additional international T20 matches (25 total)
│       ├── demo_matches.zip    # Sample ZIP archive for testing upload
│       ├── README.txt          # Original Cricsheet distribution notice
│       └── LICENSE.txt         # Cricsheet ODC-By v1.0 license file
└── tests/
    └── test_overplan.py        # Automated test suite (accounting, shrinkage, MILP)
```

---

## Important Assumptions & Tactical Limitations

1. **Consecutive Over Rule**: Under MCC Law 17.6, no bowler can bowl consecutive overs. OverPlan computes the total phase quota; the captain must sequence the overs alternately during play.
2. **Rain & Shortened Matches**: OverPlan evaluates standard 20-over T20 matches. Matches shortened under DLS change phase allocations in real life.
3. **Batter Matchups**: Left-hand/right-hand matchups, entry points, and boundary dimensions are qualitative factors that captains should layer onto OverPlan's recommendations.
4. **Historical vs. Current Form**: Career figures do not account for recent injuries, pitch conditions on the day, or tactical role changes.

---

## License

- **Codebase**: Licensed under the [MIT License](LICENSE).
- **Match Data**: Sourced from [Cricsheet](https://cricsheet.org/) under [Open Data Commons Attribution License (ODC-By v1.0)](https://opendatacommons.org/licenses/by/1-0/).
