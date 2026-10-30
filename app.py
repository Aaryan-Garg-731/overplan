"""
OverPlan - T20 Bowling Innings Decision Support System
A complete, open-source sports analytics web application for cricket captains
and performance analysts. Powered by Cricsheet ball-by-ball data, Empirical Bayes
shrinkage, and Mixed-Integer Linear Programming.
"""

import os
import io
import json
import zipfile
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_loader
import analytics
import optimizer

# Page configuration
st.set_page_config(
    page_title="OverPlan | T20 Bowling Innings Optimizer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Dark Navy background, teal & lime accents, high-contrast text
DARK_NAVY_CSS = """
<style>
    /* Dark Navy Theme Styling */
    .stApp {
        background-color: #080e1a;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1527 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Top Header Banner */
    .overplan-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .overplan-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .overplan-badge {
        background-color: #0f766e;
        color: #5eead4;
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .overplan-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 8px;
        line-height: 1.5;
    }
    
    /* Decision Support Notice Box */
    .notice-box {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #06b6d4;
        border-radius: 6px;
        padding: 12px 18px;
        margin: 14px 0 24px 0;
        font-size: 0.92rem;
        color: #cbd5e1;
        border: 1px solid #1e293b;
        border-left: 4px solid #00f5d4;
    }
    
    /* Analytics Metric Cards */
    .metric-card {
        background: #111c33;
        border: 1px solid #1e2e4f;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00f5d4;
    }
    
    .metric-val-lime {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a3e635;
    }
    
    .metric-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Badges & Tables */
    .tag-low {
        color: #f59e0b;
        background: rgba(245, 158, 11, 0.15);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .tag-ok {
        color: #10b981;
        background: rgba(16, 185, 129, 0.15);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1e293b;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #00f5d4 !important;
        border-bottom: 2px solid #00f5d4 !important;
    }

    /* Hide Streamlit Deploy button */
    .stDeployButton, [data-testid="stDeployButton"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide Main Menu (Hamburger menu with Settings & Theme options) */
    #MainMenu, [data-testid="stMainMenu"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide Top Toolbar & Header action buttons */
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    header [data-testid="baseButton-header"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide Footer */
    footer {
        display: none !important;
        visibility: hidden !important;
    }
</style>
"""
st.markdown(DARK_NAVY_CSS, unsafe_allow_html=True)


# Helper function to cache demo loading
@st.cache_data(show_spinner=False)
def get_demo_data():
    return data_loader.load_demo_dataset()


# Helper function to parse uploaded files
def process_uploaded_files(uploaded_files):
    return data_loader.load_matches_from_files(uploaded_files)


# App Header
st.markdown(
    """
    <div class="overplan-header">
        <div class="overplan-title">
            <span>🏏 OverPlan</span>
            <span class="overplan-badge">Decision Support</span>
            <span class="overplan-badge" style="background-color: #1e3a8a; color: #93c5fd;">T20 Analytics</span>
        </div>
        <div class="overplan-subtitle">
            An open-source cricket decision-support tool helping captains and performance analysts evaluate historical bowling phase performance and determine an optimal 20-over allocation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory decision-support disclaimer banner
st.markdown(
    """
    <div class="notice-box">
        <strong>⚠️ Decision-Support Advisory:</strong> OverPlan's recommendations are historical statistical baselines designed to support tactical discussion. They do not constitute performance guarantees, causal claims, or automated captaincy. In-game variables—including pitch degradation, weather, batter matchups (LHB/RHB), match situation, and the non-consecutive over rule—must always guide live decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# SIDEBAR: DATA INGESTION & FILTER CONTROLS
# ==========================================
st.sidebar.title("Data & Settings")

data_source_mode = st.sidebar.radio(
    "Data Source Mode",
    options=["Bundled Demo Data (Real Cricsheet T20s)", "Upload Custom Files (.json / .zip)"],
    index=0,
    help="Select whether to use the bundled Cricsheet demo dataset or upload your own match files.",
)

matches_df = pd.DataFrame()
deliveries_df = pd.DataFrame()
parsing_warnings = []

if data_source_mode.startswith("Bundled Demo Data"):
    with st.spinner("Loading bundled Cricsheet demo matches..."):
        matches_df, deliveries_df, parsing_warnings = get_demo_data()

    if matches_df.empty:
        st.sidebar.error("Bundled demo files not found. Please upload match files below.")
    else:
        st.sidebar.success(f"Loaded {len(matches_df)} demo matches ({len(deliveries_df):,} balls).")

        # Allow user to download the sample zip for offline inspection
        demo_zip_path = os.path.join(os.path.dirname(__file__), "data", "demo", "demo_matches.zip")
        if os.path.exists(demo_zip_path):
            with open(demo_zip_path, "rb") as zf:
                st.sidebar.download_button(
                    label="📥 Download Demo Matches (.zip)",
                    data=zf.read(),
                    file_name="cricsheet_t20_demo_matches.zip",
                    mime="application/zip",
                    help="Download the exact real Cricsheet T20 matches used in this demo.",
                )

else:
    # Upload Mode
    st.sidebar.markdown("### Upload Cricsheet Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Cricsheet JSON matches or ZIP archive",
        type=["json", "zip"],
        accept_multiple_files=True,
        help="Upload individual match JSON files or a ZIP archive containing match JSON files from https://cricsheet.org/downloads/",
    )

    if uploaded_files:
        with st.spinner("Parsing uploaded files..."):
            matches_df, deliveries_df, parsing_warnings = process_uploaded_files(uploaded_files)
        if not matches_df.empty:
            st.sidebar.success(f"Parsed {len(matches_df)} valid matches ({len(deliveries_df):,} balls).")
    else:
        # Empty state prompt
        st.sidebar.info("Upload Cricsheet .json or .zip files to begin analysis.")


# Show parsing warnings if any
if parsing_warnings:
    with st.sidebar.expander(f"⚠️ Parsing Warnings ({len(parsing_warnings)})", expanded=False):
        for w in parsing_warnings[:15]:
            st.write(f"- {w}")
        if len(parsing_warnings) > 15:
            st.write(f"... and {len(parsing_warnings) - 15} more warnings.")


# Empty state handling
if matches_df.empty or deliveries_df.empty:
    st.markdown(
        """
        <div style="background: #111c33; border: 2px dashed #334155; border-radius: 12px; padding: 48px; text-align: center; margin-top: 30px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">📁</div>
            <h3 style="color: #f8fafc; margin-bottom: 8px;">No Match Data Loaded</h3>
            <p style="color: #94a3b8; max-width: 600px; margin: 0 auto 24px auto;">
                Please select <strong>Bundled Demo Data</strong> in the sidebar to immediately explore real T20 matches, or upload Cricsheet ball-by-ball JSON/ZIP files.
            </p>
            <div style="background: #0d1527; border: 1px solid #1e293b; border-radius: 8px; padding: 18px; max-width: 650px; margin: 0 auto; text-align: left;">
                <strong style="color: #00f5d4;">How to obtain Cricsheet data:</strong>
                <ul style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px; line-height: 1.6;">
                    <li>Download match JSON files from <a href="https://cricsheet.org/downloads/" target="_blank" style="color: #38bdf8;">cricsheet.org/downloads</a></li>
                    <li>Supports tournament packages (e.g., IPL, T20 Internationals, Big Bash) or individual match JSON files.</li>
                    <li>Files are licensed under <a href="https://cricsheet.org/licensing/" target="_blank" style="color: #38bdf8;">ODC-By v1.0</a>.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ==========================================
# DATASET FILTERS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("Filter Matches")

# Filter: Gender
all_genders = sorted(deliveries_df["gender"].dropna().unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", options=["All"] + all_genders, index=0)

filtered_deliv = deliveries_df.copy()
if selected_gender != "All":
    filtered_deliv = filtered_deliv[filtered_deliv["gender"] == selected_gender]

# Filter: Competition / Event
all_events = sorted(filtered_deliv["event"].dropna().unique().tolist())
selected_events = st.sidebar.multiselect(
    "Competition / Series",
    options=all_events,
    default=[],
    placeholder="All competitions included",
)
if selected_events:
    filtered_deliv = filtered_deliv[filtered_deliv["event"].isin(selected_events)]

# Filter: Season
all_seasons = sorted(filtered_deliv["season"].dropna().unique().tolist())
selected_seasons = st.sidebar.multiselect(
    "Season",
    options=all_seasons,
    default=[],
    placeholder="All seasons included",
)
if selected_seasons:
    filtered_deliv = filtered_deliv[filtered_deliv["season"].isin(selected_seasons)]

# Filter: Team
all_bowling_teams = sorted(
    list(set(filtered_deliv["bowling_team"].dropna().unique()).union(filtered_deliv["batting_team"].dropna().unique()))
)
selected_team_filter = st.sidebar.selectbox(
    "Focus Team (Optional)",
    options=["All Teams"] + all_bowling_teams,
    index=0,
    help="Filter data to matches involving this team to narrow down roster players.",
)
if selected_team_filter != "All Teams":
    filtered_deliv = filtered_deliv[
        (filtered_deliv["bowling_team"] == selected_team_filter)
        | (filtered_deliv["batting_team"] == selected_team_filter)
    ]

# Analytics Hyperparameters
st.sidebar.markdown("---")
st.sidebar.subheader("Analytics Parameters")
shrinkage_m = st.sidebar.slider(
    "Empirical Bayes Prior Weight (m in overs)",
    min_value=1.0,
    max_value=10.0,
    value=analytics.DEFAULT_SHRINKAGE_WEIGHT,
    step=0.5,
    help="Number of baseline overs mixed into each bowler's phase record. Higher values pull small samples more strongly toward league phase averages.",
)

low_sample_threshold = st.sidebar.slider(
    "Low Sample Warning Threshold (overs)",
    min_value=1.0,
    max_value=6.0,
    value=analytics.LOW_SAMPLE_THRESHOLD_OVERS,
    step=0.5,
    help="Overs bowled below this threshold will be flagged as low sample size.",
)

if filtered_deliv.empty:
    st.warning("No deliveries match the selected filters. Please broaden your filter selections.")
    st.stop()


# Compute phase metrics and competition baselines
phase_metrics_df, comp_baselines, bowler_summary_df = analytics.compute_bowler_phase_metrics(
    filtered_deliv,
    shrinkage_weight=shrinkage_m,
    low_sample_threshold=low_sample_threshold,
)


# ==========================================
# MAIN DASHBOARD NAVIGATION
# ==========================================
tab_planner, tab_comparison, tab_methodology, tab_dataset = st.tabs([
    "🎯 20-Over Allocation Planner",
    "📊 Phase Comparison & Charts",
    "🔬 Shrinkage & Methodology",
    "📄 Data Coverage & Attribution",
])


# ==========================================
# TAB 1: ALLOCATION PLANNER
# ==========================================
with tab_planner:
    st.markdown("### Suggested 20-Over Allocation")
    st.write(
        "Select your available bowling lineup. The optimizer uses **Mixed-Integer Linear Programming (MILP)** "
        "to allocate exactly 6 Powerplay, 9 Middle, and 5 Death overs while respecting the 4-over maximum per bowler "
        "and minimizing expected runs conceded based on adjusted phase economy rates."
    )

    # Roster Selection
    available_bowlers = sorted(phase_metrics_df["bowler"].unique().tolist())

    # Smart default bowler suggestion based on selected focus team or top career overs
    if selected_team_filter != "All Teams":
        team_bowlers = (
            filtered_deliv[filtered_deliv["bowling_team"] == selected_team_filter]["bowler"]
            .unique()
            .tolist()
        )
        default_bowlers = [b for b in team_bowlers if b in available_bowlers][:6]
    else:
        top_bowlers_by_ov = (
            bowler_summary_df.sort_values(by="overs", ascending=False)["bowler"]
            .head(6)
            .tolist()
        )
        default_bowlers = top_bowlers_by_ov

    col_roster_select, col_roster_actions = st.columns([3, 1])

    with col_roster_select:
        selected_roster = st.multiselect(
            "Select Bowling Squad (Minimum 5 bowlers required):",
            options=available_bowlers,
            default=default_bowlers if len(default_bowlers) >= 5 else available_bowlers[:5],
            placeholder="Search and select bowlers...",
            help="Choose the 5 to 11 bowlers available for selection in this match.",
        )

    with col_roster_actions:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset to Top Bowlers", use_container_width=True):
            st.rerun()

    # Advanced Bowler Constraints Expander
    with st.expander("⚙️ Advanced Constraints: Bowler Over Limits & Phase Restrictions", expanded=False):
        st.caption(
            "Optionally restrict part-timers (e.g. max 2 overs) or exclude bowlers from certain phases (e.g. no Death overs)."
        )
        custom_limits = {}
        phase_permissions = {}

        if selected_roster:
            cols_grid = st.columns(min(len(selected_roster), 4))
            for idx, bowler in enumerate(selected_roster):
                col = cols_grid[idx % len(cols_grid)]
                with col:
                    st.markdown(f"**{bowler}**")
                    custom_max = st.number_input(
                        f"Max overs ({bowler})",
                        min_value=0,
                        max_value=4,
                        value=4,
                        step=1,
                        key=f"limit_{bowler}",
                    )
                    custom_limits[bowler] = int(custom_max)

                    allowed = st.multiselect(
                        f"Eligible Phases",
                        options=["Powerplay", "Middle", "Death"],
                        default=["Powerplay", "Middle", "Death"],
                        key=f"phases_{bowler}",
                    )
                    phase_permissions[bowler] = set(allowed)
        else:
            custom_limits = None
            phase_permissions = None

    # Solve Allocation
    if len(selected_roster) < 5:
        st.warning(
            f"⚠️ **At least 5 bowlers are required** to bowl a standard 20-over T20 innings "
            f"(maximum 4 overs per bowler). Currently selected: {len(selected_roster)}."
        )
    else:
        alloc_res = optimizer.solve_optimal_allocation(
            roster=selected_roster,
            phase_metrics_df=phase_metrics_df,
            bowler_limits=custom_limits,
            phase_permissions=phase_permissions,
            comp_baselines=comp_baselines,
        )

        if not alloc_res.is_feasible:
            st.error(f"❌ **Allocation Infeasible**: {alloc_res.message}")
            st.markdown("#### Suggested Fixes:")
            for d in alloc_res.diagnostics:
                st.write(f"- {d}")
        else:
            # Feasible solution: Display Key Metric Cards
            st.markdown("---")
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

            with mcol1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Est. Runs Conceded</div>
                        <div class="metric-val">{alloc_res.total_estimated_runs:.1f}</div>
                        <div class="metric-sub">Across 20.0 Overs</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with mcol2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Projected Innings RPO</div>
                        <div class="metric-val-lime">{alloc_res.projected_economy:.2f}</div>
                        <div class="metric-sub">Runs Per Over</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with mcol3:
                pp_runs = alloc_res.phase_runs.get("Powerplay", 0.0)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Powerplay (Overs 1-6)</div>
                        <div class="metric-val">{pp_runs:.1f}</div>
                        <div class="metric-sub">{pp_runs / 6.0:.2f} RPO (6 ov)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with mcol4:
                mid_runs = alloc_res.phase_runs.get("Middle", 0.0)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Middle (Overs 7-15)</div>
                        <div class="metric-val">{mid_runs:.1f}</div>
                        <div class="metric-sub">{mid_runs / 9.0:.2f} RPO (9 ov)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with mcol5:
                death_runs = alloc_res.phase_runs.get("Death", 0.0)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Death (Overs 16-20)</div>
                        <div class="metric-val">{death_runs:.1f}</div>
                        <div class="metric-sub">{death_runs / 5.0:.2f} RPO (5 ov)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Allocation Table & Visual Overview
            col_chart, col_table = st.columns([1, 1])

            with col_chart:
                st.markdown("#### Suggested Overs Distribution")
                alloc_df = alloc_res.allocation_df.copy()

                # Build Stacked Bar Chart for Overs Allocation
                fig_alloc = go.Figure()
                fig_alloc.add_trace(go.Bar(
                    y=alloc_df["Bowler"],
                    x=alloc_df["PP (Overs 1-6)"],
                    name="Powerplay (1-6)",
                    orientation="h",
                    marker_color="#00f5d4",
                ))
                fig_alloc.add_trace(go.Bar(
                    y=alloc_df["Bowler"],
                    x=alloc_df["Middle (Overs 7-15)"],
                    name="Middle (7-15)",
                    orientation="h",
                    marker_color="#38bdf8",
                ))
                fig_alloc.add_trace(go.Bar(
                    y=alloc_df["Bowler"],
                    x=alloc_df["Death (Overs 16-20)"],
                    name="Death (16-20)",
                    orientation="h",
                    marker_color="#f43f5e",
                ))

                fig_alloc.update_layout(
                    barmode="stack",
                    paper_bgcolor="#0d1527",
                    plot_bgcolor="#0d1527",
                    font=dict(color="#e2e8f0"),
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(
                        title="Allocated Overs",
                        dtick=1,
                        range=[0, 4.5],
                        gridcolor="#1e293b",
                    ),
                    yaxis=dict(
                        title="",
                        autorange="reversed",
                        gridcolor="#1e293b",
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                    height=380,
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

            with col_table:
                st.markdown("#### Tactical Allocation Matrix")
                display_cols = [
                    "Bowler",
                    "PP (Overs 1-6)",
                    "Middle (Overs 7-15)",
                    "Death (Overs 16-20)",
                    "Total Overs",
                    "Est. Runs Conceded",
                    "Suggested Role",
                ]
                st.dataframe(
                    alloc_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            # Detailed Breakdown Table
            st.markdown("#### Comprehensive Bowler Breakdown")
            st.dataframe(
                alloc_df[[
                    "Bowler",
                    "Total Overs",
                    "PP (Overs 1-6)",
                    "Middle (Overs 7-15)",
                    "Death (Overs 16-20)",
                    "Est. Runs Conceded",
                    "PP Adj Econ",
                    "Middle Adj Econ",
                    "Death Adj Econ",
                    "Career Wickets",
                    "Career Overs",
                    "Sample Confidence",
                ]],
                use_container_width=True,
                hide_index=True,
            )

            # Tactical Advice and Important Limitations
            st.markdown("---")
            st.markdown("#### Tactical Notes & Key Cricket Assumptions")
            st.markdown(
                """
                - **Wicket-Taking Separation**: Wickets are displayed descriptively in the table and are not combined with runs into an artificial rating. In high-stakes matches, a captain may choose a bowler with slightly higher economy in the Death if their wicket strike rate is significantly superior.
                - **Consecutive-Over Rule**: Cricket Law 17.6 prohibits any bowler from bowling two consecutive overs. The captain must sequence these allocations alternately (e.g. Bowler A in Over 1, Bowler B in Over 2, Bowler A in Over 3).
                - **Match Situation Dynamics**: If early wickets fall in the Powerplay, captains may hold back an anchor's remaining overs for attacking middle phases.
                - **Pitch & Matchup Context**: These estimates reflect historical phase tendencies. If a pitch is gripping for spin or offering seam movement under lights, adapt the baseline accordingly.
                """
            )


# ==========================================
# TAB 2: PHASE COMPARISON & CHARTS
# ==========================================
with tab_comparison:
    st.markdown("### Bowler Phase Comparison")
    st.write(
        "Compare bowlers' performance across Powerplay, Middle, and Death overs. "
        "Raw figures and Empirical Bayes adjusted figures are displayed alongside sample sizes."
    )

    # Phase Baseline Stats Bar
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        pp_base = comp_baselines.get("Powerplay", {})
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Powerplay League Norm</div>
                <div class="metric-val">{pp_base.get('economy', 0.0):.2f} RPO</div>
                <div class="metric-sub">{pp_base.get('overs', 0.0):.1f} total overs | {pp_base.get('wickets', 0)} wickets</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with bcol2:
        mid_base = comp_baselines.get("Middle", {})
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Middle Overs League Norm</div>
                <div class="metric-val">{mid_base.get('economy', 0.0):.2f} RPO</div>
                <div class="metric-sub">{mid_base.get('overs', 0.0):.1f} total overs | {mid_base.get('wickets', 0)} wickets</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with bcol3:
        death_base = comp_baselines.get("Death", {})
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Death Overs League Norm</div>
                <div class="metric-val">{death_base.get('economy', 0.0):.2f} RPO</div>
                <div class="metric-sub">{death_base.get('overs', 0.0):.1f} total overs | {death_base.get('wickets', 0)} wickets</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Phase Economy Comparison Chart
    st.markdown("#### Adjusted Economy Rates Across Phases")

    # Filter to bowlers with at least min overs for clean visualization
    min_career_ov = st.slider(
        "Filter chart by minimum career overs bowled in dataset:",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=1.0,
        help="Filters the comparison chart to bowlers with sufficient sample size.",
    )

    qualifying_bowlers = bowler_summary_df[bowler_summary_df["overs"] >= min_career_ov]["bowler"].tolist()
    chart_df = phase_metrics_df[phase_metrics_df["bowler"].isin(qualifying_bowlers)].copy()

    if chart_df.empty:
        st.info("No bowlers match the minimum overs threshold. Lower the slider to see more bowlers.")
    else:
        # Grouped bar chart comparing PP, Middle, Death adjusted economy
        fig_comp = px.bar(
            chart_df,
            x="bowler",
            y="adjusted_economy",
            color="phase",
            barmode="group",
            color_discrete_map={
                "Powerplay": "#00f5d4",
                "Middle": "#38bdf8",
                "Death": "#f43f5e",
            },
            labels={
                "bowler": "Bowler",
                "adjusted_economy": "Adjusted Economy (RPO)",
                "phase": "Innings Phase",
            },
            hover_data={
                "raw_economy": ":.2f",
                "exact_overs": ":.1f",
                "wickets": True,
                "confidence": True,
            },
            height=480,
        )

        fig_comp.update_layout(
            paper_bgcolor="#0d1527",
            plot_bgcolor="#0d1527",
            font=dict(color="#e2e8f0"),
            xaxis=dict(
                title="",
                gridcolor="#1e293b",
                tickangle=-45,
            ),
            yaxis=dict(
                title="Adjusted Economy (Runs/Over)",
                gridcolor="#1e293b",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=20, r=20, t=30, b=80),
        )

        # Add horizontal dashed lines for league phase averages
        fig_comp.add_hline(
            y=pp_base.get("economy", 7.4),
            line_dash="dot",
            line_color="#00f5d4",
            annotation_text=f"PP Avg ({pp_base.get('economy', 7.4):.2f})",
            annotation_position="top left",
        )
        fig_comp.add_hline(
            y=death_base.get("economy", 9.6),
            line_dash="dot",
            line_color="#f43f5e",
            annotation_text=f"Death Avg ({death_base.get('economy', 9.6):.2f})",
            annotation_position="bottom right",
        )

        st.plotly_chart(fig_comp, use_container_width=True)

    # Detailed Comparison Table
    st.markdown("#### Phase Statistics Table")
    selected_phase_filter = st.selectbox(
        "Filter by Phase:",
        options=["All Phases", "Powerplay", "Middle", "Death"],
        index=0,
    )

    table_data = phase_metrics_df.copy()
    if selected_phase_filter != "All Phases":
        table_data = table_data[table_data["phase"] == selected_phase_filter]

    # Calculate difference between raw and adjusted economy
    table_data["shrinkage_shift"] = (
        table_data["adjusted_economy"] - table_data["raw_economy"]
    ).round(2)

    st.dataframe(
        table_data[[
            "bowler",
            "phase",
            "overs_formatted",
            "runs_conceded",
            "wickets",
            "raw_economy",
            "adjusted_economy",
            "shrinkage_shift",
            "dot_pct",
            "strike_rate",
            "warning",
        ]].rename(columns={
            "bowler": "Bowler",
            "phase": "Phase",
            "overs_formatted": "Overs",
            "runs_conceded": "Runs",
            "wickets": "Wkts",
            "raw_economy": "Raw Econ",
            "adjusted_economy": "Adj Econ",
            "shrinkage_shift": "Shift (Δ)",
            "dot_pct": "Dot %",
            "strike_rate": "Strike Rate",
            "warning": "Sample Status",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ==========================================
# TAB 3: METHODOLOGY & SHRINKAGE
# ==========================================
with tab_methodology:
    st.markdown("### Statistical Methodology & Mathematical Formulation")
    
    st.markdown(
        """
        OverPlan uses a two-stage analytical pipeline:
        1. **Empirical Bayes Phase Economy Shrinkage** to address sample variance and extreme flukes.
        2. **Mixed-Integer Linear Programming (MILP)** to allocate exactly 20 overs across the chosen bowling roster.
        """
    )

    st.markdown("---")
    st.markdown("#### 1. Why Raw Economy Rate Deceives")
    st.markdown(
        """
        In short-form cricket analytics, unadjusted economy rates are notoriously prone to sample size artifacts:
        - A bowler who bowls **1 over for 3 runs** has a raw economy of **3.00**.
        - A bowler who bowls **1 over for 17 runs** has a raw economy of **17.00**.
        
        Treating these raw figures as true representations of skill leads to severe decision-making errors. An optimizer using unadjusted figures would unrealistically allocate maximum overs to small-sample flukes.
        """
    )

    st.markdown("#### 2. Empirical Bayes Shrinkage Formula")
    st.latex(
        r"E_{\text{adjusted}}(b, p) = \frac{\text{Runs}_{b, p} + m \cdot \mu_p}{\text{Overs}_{b, p} + m} = \left(\frac{\text{Overs}_{b, p}}{\text{Overs}_{b, p} + m}\right) E_{\text{raw}}(b, p) + \left(\frac{m}{\text{Overs}_{b, p} + m}\right) \mu_p"
    )

    st.markdown(
        f"""
        Where:
        - $E_{{\\text{{raw}}}}(b, p)$ is the bowler's observed economy rate in phase $p$.
        - $\\mu_p$ is the competition-wide phase average economy (currently **{comp_baselines.get('Powerplay', {}).get('economy', 7.4):.2f}** for Powerplay, **{comp_baselines.get('Middle', {}).get('economy', 7.8):.2f}** for Middle, and **{comp_baselines.get('Death', {}).get('economy', 9.6):.2f}** for Death).
        - $m$ is the prior shrinkage weight in overs (configured at **{shrinkage_m:.1f} overs**).
        - When $\\text{{Overs}}_{{b, p}} = 0$ (no history in that phase), $E_{{\\text{{adjusted}}}} = \\mu_p$, safely defaulting to the league norm with a low-confidence tag.
        - As sample size $\\text{{Overs}}_{{b, p}}$ grows large ($>10$ overs), the weight $\\frac{{\\text{{Overs}}}}{{\\text{{Overs}} + m}} \\to 1$, allowing the bowler's established performance to dominate.
        """
    )

    st.markdown("---")
    st.markdown("#### 3. Constrained Optimization Formulation")
    st.markdown(
        """
        Let $x_{b, p} \\in \\{0, 1, 2, 3, 4\\}$ denote the integer number of overs allocated to bowler $b$ in phase $p \\in \\{\\text{Powerplay}, \\text{Middle}, \\text{Death}\\}$.
        """
    )

    st.latex(
        r"\min \sum_{b \in \text{Roster}} \sum_{p \in \text{Phases}} x_{b, p} \cdot E_{\text{adjusted}}(b, p)"
    )

    st.markdown(
        """
        Subject to:
        1. **Phase Totals**:
           - $\\sum_{b} x_{b, \\text{PP}} = 6$ (Overs 1–6)
           - $\\sum_{b} x_{b, \\text{MID}} = 9$ (Overs 7–15)
           - $\\sum_{b} x_{b, \\text{DEATH}} = 5$ (Overs 16–20)
        2. **Bowler Maximum Limit**:
           - $\\sum_{p} x_{b, p} \\le 4$ for all $b \\in \\text{Roster}$
        3. **Integrality & Non-negativity**:
           - $x_{b, p} \\in \\mathbb{Z}_{\\ge 0}$
        """
    )

    st.markdown("---")
    st.markdown("#### 4. Cricket Rules & Accounting Integrity")
    st.markdown(
        """
        - **Legal Deliveries**: Only legal balls are counted toward completed overs. Deliveries featuring **wides** or **no-balls** require re-bowling under MCC Law 21 & 22 and are excluded from the legal ball count.
        - **Bowler-Conceded Runs**: Calculated strictly as $\\text{Batter Runs} + \\text{Wides} + \\text{No-balls}$. Byes, leg-byes, and penalty runs are fielding/extras extras and are **not** debited to the bowler's economy.
        - **Bowler-Credited Wickets**: Only dismissals attributed to the bowler (bowled, caught, caught and bowled, lbw, stumped, hit wicket) are included. Run-outs and retirements are excluded.
        """
    )


# ==========================================
# TAB 4: DATASET & ATTRIBUTION
# ==========================================
with tab_dataset:
    st.markdown("### Data Coverage & Cricsheet Attribution")

    st.markdown(
        """
        <div style="background: #111c33; border: 1px solid #1e293b; border-radius: 8px; padding: 18px; margin-bottom: 20px;">
            <strong style="color: #00f5d4; font-size: 1.1rem;">Cricsheet Open Data License Attribution</strong>
            <p style="color: #cbd5e1; margin-top: 8px; line-height: 1.6;">
                Match data used in OverPlan is sourced from <strong>Cricsheet</strong> (<a href="https://cricsheet.org" target="_blank" style="color: #38bdf8;">cricsheet.org</a>).
                The data is provided under the terms of the <strong>Open Data Commons Attribution License (ODC-By v1.0)</strong>.
            </p>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                License details: <a href="https://opendatacommons.org/licenses/by/1-0/" target="_blank" style="color: #38bdf8;">opendatacommons.org/licenses/by/1-0/</a>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Loaded Matches Overview")
    st.write(f"Total Matches Loaded: **{len(matches_df)}** | Total Deliveries: **{len(deliveries_df):,}**")

    st.dataframe(
        matches_df[[
            "date",
            "teams",
            "event",
            "season",
            "gender",
            "venue",
            "winner",
            "result_detail",
            "source_name",
        ]].rename(columns={
            "date": "Date",
            "teams": "Teams",
            "event": "Competition",
            "season": "Season",
            "gender": "Gender",
            "venue": "Venue",
            "winner": "Winner",
            "result_detail": "Margin",
            "source_name": "Source File",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Known Limitations")
    st.markdown(
        """
        1. **Shortened Rain Matches**: Matches shortened by rain (DLS method) may feature modified phase lengths in reality. OverPlan's phase mappings apply standard 20-over definitions (Overs 1-6, 7-15, 16-20).
        2. **Historical vs. Current Form**: Historical career figures reflect past performance across available matches. Squad changes, injuries, and recent technical modifications are not captured in pure historical data.
        3. **Tactical Flexibility**: A pre-game allocation serves as a baseline plan; real match situations (such as a flurry of wickets or extreme wind favoring one end) require live adjustments by captain and coach.
        """
    )
