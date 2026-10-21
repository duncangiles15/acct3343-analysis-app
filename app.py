
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Manufacturing Stock Screener",
    page_icon="📊",
    layout="wide",
)

st.title("Manufacturing Stock Screening Dashboard")

st.markdown(
    """
    This dashboard applies a **sequential screening process** to a universe
    of manufacturing companies. Companies must pass each stage to remain
    in the candidate pool.

    The screen examines financial distress, earnings quality, operating
    asset quality, and gross profitability. Passing the screen does not
    guarantee that a stock is undervalued or suitable for investment.
    """
)


# ============================================================
# 2. LOAD AND PREPARE THE DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")

    required_columns = [
        "Name",
        "Market Cap",
        "Ticker",
        "Ticker_guess",
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profitability",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    numeric_columns = [
        "Rank",
        "Price",
        "Today",
        "Price (30 days)",
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profitability",
        "Gross_Profit",
        "Total_Assets",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def parse_market_cap(value):
        """Convert values such as $2.4T, $850B, or $45.2M to dollars."""
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        text = str(value).strip().upper()
        text = text.replace(",", "").replace("$", "").replace(" ", "")

        match = re.fullmatch(
            r"([-+]?\d*\.?\d+)([TBMK]?)",
            text,
        )

        if not match:
            return np.nan

        number = float(match.group(1))
        suffix = match.group(2)

        multipliers = {
            "": 1,
            "K": 1_000,
            "M": 1_000_000,
            "B": 1_000_000_000,
            "T": 1_000_000_000_000,
        }

        return number * multipliers[suffix]

    df["Market_Cap_Value"] = df["Market Cap"].apply(parse_market_cap)

    # Use the primary ticker, falling back to the guessed ticker if needed.
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip()
    df["Ticker_guess"] = (
        df["Ticker_guess"].fillna("").astype(str).str.strip()
    )

    df["Display_Ticker"] = df["Ticker"]
    use_guess = df["Display_Ticker"].eq("")
    df.loc[use_guess, "Display_Ticker"] = df.loc[use_guess, "Ticker_guess"]

    df["Name"] = df["Name"].fillna("Unknown company")

    return df


try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "Could not find universe.csv. Make sure it is in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()


if data.empty:
    st.error("universe.csv contains no companies.")
    st.stop()


# ============================================================
# 3. SIDEBAR: SCREENING CONTROLS
# ============================================================

st.sidebar.title("Screen Controls")
st.sidebar.caption(
    "Adjust the rules below. Each stage filters the companies "
    "that survived the previous stage."
)

max_market_cap_b = data["Market_Cap_Value"].max() / 1_000_000_000

if pd.isna(max_market_cap_b) or max_market_cap_b <= 0:
    max_market_cap_b = 100.0

slider_max = max(100.0, float(np.ceil(max_market_cap_b / 100) * 100))

min_market_cap = st.sidebar.number_input(
    "Minimum market cap ($ billions)",
    min_value=0.0,
    max_value=slider_max,
    value=0.0,
    step=1.0,
    help="Companies below this market capitalization are excluded first.",
)

st.sidebar.subheader("1. Financial Health")

min_altman_z = st.sidebar.number_input(
    "Minimum Altman Z-Score",
    min_value=-10.0,
    max_value=20.0,
    value=1.81,
    step=0.1,
    help=(
        "The manufacturing-model distress boundary is commonly cited "
        "as 1.81. This screen excludes companies below the selected value."
    ),
)

st.sidebar.subheader("2. Earnings Quality")

keep_lowest_sta = st.sidebar.slider(
    "Keep lowest STA percentile (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help=(
        "Lower STA is generally preferred. At 50%, the screen keeps "
        "companies at or below the median STA cutoff."
    ),
)

st.sidebar.subheader("3. Operating Asset Quality")

keep_lowest_snoa = st.sidebar.slider(
    "Keep lowest SNOA percentile (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help=(
        "Lower SNOA is preferred. At 50%, the screen keeps companies "
        "at or below the median SNOA cutoff."
    ),
)

st.sidebar.subheader("4. Gross Profitability")

keep_highest_gp = st.sidebar.slider(
    "Keep highest gross profitability percentile (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help=(
        "Higher gross profitability is preferred. At 50%, the screen "
        "keeps companies at or above the median gross profitability cutoff."
    ),
)

if st.sidebar.button("Reset screening controls"):
    st.rerun()


# ============================================================
# 4. DEFINE THE INITIAL ELIGIBLE UNIVERSE
# ============================================================

# Market cap is converted from dollars to billions for this comparison.
eligible = data[
    data["Market_Cap_Value"].notna()
    & (data["Market_Cap_Value"] >= min_market_cap * 1_000_000_000)
].copy()

if eligible.empty:
    st.warning(
        "No companies meet the selected minimum market capitalization. "
        "Lower the market-cap filter to continue."
    )
    st.stop()


# ============================================================
# 5. CALCULATE FIXED PERCENTILE CUT-OFFS
# ============================================================

# Important methodology choice:
# Calculate percentile cutoffs from the initial eligible universe,
# not from the shrinking pool at each stage. This keeps the standard
# for each metric consistent and makes the funnel easier to interpret.

sta_cutoff = eligible["STA"].dropna().quantile(
    keep_lowest_sta / 100
)

snoa_cutoff = eligible["SNOA"].dropna().quantile(
    keep_lowest_snoa / 100
)

gp_cutoff = eligible["Gross_Profitability"].dropna().quantile(
    1 - keep_highest_gp / 100
)


# ============================================================
# 6. RUN THE SEQUENTIAL SCREENING FUNNEL
# ============================================================

current = eligible.copy()
stage_records = []
stage_snapshots = {}


def apply_stage(stage_name, metric_name, rule_description, condition):
    """
    Apply one filter to the surviving companies, record how many
    pass, and preserve the survivors for later inspection.
    """
    global current

    before = len(current)

    # Missing metric values cannot pass a metric-based filter.
    pass_mask = current[metric_name].notna() & condition(current[metric_name])

    missing_count = int(current[metric_name].isna().sum())
    failed_value_count = int(
        (current[metric_name].notna() & ~pass_mask).sum()
    )

    current = current.loc[pass_mask].copy()

    after = len(current)

    stage_records.append(
        {
            "Stage": stage_name,
            "Rule": rule_description,
            "Before": before,
            "Removed": before - after,
            "Missing data": missing_count,
            "Failed rule": failed_value_count,
            "Remaining": after,
        }
    )

    stage_snapshots[stage_name] = current.copy()


# Stage 1: Altman Z-Score financial-health filter.
apply_stage(
    stage_name="Financial Health — Altman Z",
    metric_name="Altman_Z",
    rule_description=f"Altman Z ≥ {min_altman_z:.2f}",
    condition=lambda s: s >= min_altman_z,
)

# Stage 2: STA earnings-quality filter.
apply_stage(
    stage_name="Earnings Quality — STA",
    metric_name="STA",
    rule_description=(
        f"STA ≤ {sta_cutoff:.4f} "
        f"(lowest {keep_lowest_sta}% of initial eligible universe)"
    ),
    condition=lambda s: s <= sta_cutoff,
)

# Stage 3: SNOA operating-asset-quality filter.
apply_stage(
    stage_name="Operating Asset Quality — SNOA",
    metric_name="SNOA",
    rule_description=(
        f"SNOA ≤ {snoa_cutoff:.4f} "
        f"(lowest {keep_lowest_snoa}% of initial eligible universe)"
    ),
    condition=lambda s: s <= snoa_cutoff,
)

# Stage 4: Gross profitability filter.
apply_stage(
    stage_name="Gross Profitability",
    metric_name="Gross_Profitability",
    rule_description=(
        f"Gross profitability ≥ {gp_cutoff:.4f} "
        f"(highest {keep_highest_gp}% of initial eligible universe)"
    ),
    condition=lambda s: s >= gp_cutoff,
)

final_candidates = current.copy()


# ============================================================
# 7. DISPLAY THE LIVE FUNNEL
# ============================================================

st.header("Live Screening Funnel")

st.caption(
    "The starting count is the number of companies meeting the market-cap "
    "filter. Each subsequent stage is applied only to the companies "
    "remaining from the previous stage."
)

funnel_rows = [
    {
        "Stage": "Starting eligible universe",
        "Before": len(eligible),
        "Removed": 0,
        "Remaining": len(eligible),
    }
]

for record in stage_records:
    funnel_rows.append(
        {
            "Stage": record["Stage"],
            "Before": record["Before"],
            "Removed": record["Removed"],
            "Remaining": record["Remaining"],
        }
    )

funnel_df = pd.DataFrame(funnel_rows)

col1, col2, col3 = st.columns(3)

col1.metric("Eligible companies", f"{len(eligible):,}")
col2.metric("Final candidates", f"{len(final_candidates):,}")

if len(eligible) > 0:
    survival_rate = len(final_candidates) / len(eligible) * 100
else:
    survival_rate = 0

col3.metric("Survival rate", f"{survival_rate:.1f}%")

st.dataframe(
    funnel_df,
    use_container_width=True,
    hide_index=True,
)

chart_df = funnel_df.copy()

fig = px.bar(
    chart_df,
    x="Stage",
    y="Remaining",
    title="Companies Remaining After Each Screening Stage",
    text="Remaining",
)

fig.update_layout(
    xaxis_title="Screening stage",
    yaxis_title="Companies remaining",
    xaxis_tickangle=-25,
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 8. EXPLAIN EACH STAGE AND SHOW ITS COUNTS
# ============================================================

st.header("Screening Stage Details")

with st.expander("Stage 1 — Financial Health: Altman Z-Score", expanded=True):
    st.markdown(
        """
        **Purpose:** remove companies that fall below the selected
        financial-distress boundary before evaluating other qualities.

        The manufacturing Altman Z-Score combines working capital,
        retained earnings, EBIT, market equity relative to liabilities,
        and sales relative to assets. A higher score generally indicates
        a stronger financial position under this model.

        The commonly cited manufacturing distress boundary is 1.81,
        while 2.99 is commonly used as the safe-zone boundary. This app
        uses the selected minimum as an initial exclusion rule; passing
        it does not guarantee that a company is financially safe.
        """
    )

with st.expander("Stage 2 — Earnings Quality: STA"):
    st.markdown(
        """
        **Formula:**

        STA = (Net Income − Operating Cash Flow) / Total Assets

        Lower STA is generally preferred in this screen because a lower
        accrual measure can indicate that reported earnings are more
        strongly supported by cash flow. Negative values can occur when
        operating cash flow exceeds net income.

        The selected percentile cutoff is calculated from the initial
        eligible universe. Companies with missing STA are excluded at
        this stage because their earnings quality cannot be evaluated
        using this measure.
        """
    )

with st.expander("Stage 3 — Operating Asset Quality: SNOA"):
    st.markdown(
        """
        **Purpose:** favour companies with lower net operating assets
        relative to total assets.

        SNOA is intended to measure the extent of net operating assets
        relative to the company's asset base. Lower values are preferred
        by this particular screening rule, but the metric should be
        interpreted in the context of each company's business model.

        The cutoff is based on the initial eligible universe, not just
        the survivors of the previous stage.
        """
    )

with st.expander("Stage 4 — Gross Profitability"):
    st.markdown(
        """
        **Formula:**

        Gross Profitability = Gross Profit / Total Assets

        Higher gross profitability means the company generates more
        gross profit relative to its asset base. The screen therefore
        retains companies at or above the selected upper-tail cutoff.

        Gross profitability is a business-quality measure, not a direct
        measure of valuation. A highly profitable company can still be
        overpriced.
        """
    )


# ============================================================
# 9. SHOW THE FINAL CANDIDATES
# ============================================================

st.header("Final Candidates")

st.markdown(
    """
    These companies passed every screening stage using the current
    settings. They are candidates for further research, not automatic
    buy recommendations.
    """
)

if final_candidates.empty:
    st.warning(
        "No companies passed every stage. Consider whether your cutoffs "
        "are too restrictive, and review how much data is missing."
    )
else:
    sort_options = {
        "Highest gross profitability": (
            "Gross_Profitability",
            False,
        ),
        "Lowest STA": ("STA", True),
        "Lowest SNOA": ("SNOA", True),
        "Highest Altman Z-Score": ("Altman_Z", False),
        "Largest market capitalization": ("Market_Cap_Value", False),
    }

    sort_choice = st.selectbox(
        "Sort final candidates by",
        list(sort_options.keys()),
    )

    sort_column, ascending = sort_options[sort_choice]

    sorted_candidates = final_candidates.sort_values(
        by=sort_column,
        ascending=ascending,
        na_position="last",
    )

    display_columns = [
        col
        for col in [
            "Display_Ticker",
            "Name",
            "Market Cap",
            "Price",
            "Country",
            "Altman_Z",
            "STA",
            "SNOA",
            "Gross_Profitability",
        ]
        if col in sorted_candidates.columns
    ]

    display_df = sorted_candidates[display_columns].copy()

    # Convert ratio columns to percentage points for easier reading.
    percentage_columns = {
        "STA": "STA (%)",
        "SNOA": "SNOA (%)",
        "Gross_Profitability": "Gross Profitability (%)",
    }

    for original, display_name in percentage_columns.items():
        if original in display_df.columns:
            display_df[display_name] = display_df[original] * 100
            display_df.drop(columns=[original], inplace=True)

    if "Display_Ticker" in display_df.columns:
        display_df.rename(
            columns={"Display_Ticker": "Ticker"},
            inplace=True,
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv = sorted_candidates.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download final candidates (CSV)",
        data=csv,
        file_name="manufacturing_screen_final_candidates.csv",
        mime="text/csv",
    )


# ============================================================
# 10. REVIEW THE COMPANIES AT EACH STAGE
# ============================================================

st.header("Review Each Stage")

stage_names = list(stage_snapshots.keys())

selected_stage = st.selectbox(
    "Choose a stage to inspect",
    stage_names,
)

stage_data = stage_snapshots[selected_stage]

st.write(
    f"**{len(stage_data):,} companies** remained after "
    f"{selected_stage.lower()}."
)

review_columns = [
    col
    for col in [
        "Display_Ticker",
        "Name",
        "Market Cap",
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profitability",
    ]
    if col in stage_data.columns
]

st.dataframe(
    stage_data[review_columns].sort_values(
        by="Name",
        na_position="last",
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    label=f"Download survivors after {selected_stage}",
    data=stage_data.to_csv(index=False).encode("utf-8"),
    file_name="screen_stage_survivors.csv",
    mime="text/csv",
    key=f"download_{selected_stage}",
)


# ============================================================
# 11. METHODOLOGY, COVERAGE, AND LIMITATIONS
# ============================================================

st.header("Methodology and Data Coverage")

with st.expander("How the sequential process works", expanded=True):
    st.markdown(
        """
        1. **Market capitalization:** define the initial eligible universe.
        2. **Altman Z-Score:** apply a financial-distress exclusion rule.
        3. **STA:** retain the selected lower portion of STA values.
        4. **SNOA:** retain the selected lower portion of SNOA values.
        5. **Gross profitability:** retain the selected upper portion of
           gross profitability values.

        **Percentile methodology:** STA, SNOA, and gross profitability
        cutoffs are calculated from the eligible universe after the
        market-cap filter. The same cutoffs are then used at their
        respective stages. This prevents the percentile standard from
        changing as the funnel gets smaller.

        **Missing data:** a company missing the metric required at a
        particular stage cannot pass that stage. Missing observations
        are counted separately from companies that fail the numeric rule.

        **Interpretation:** the order matters because later metrics are
        evaluated only among companies that passed earlier stages.
        A company excluded at an early stage does not get reinstated
        by a strong result on a later metric.
        """
    )

with st.expander("Coverage and missing values"):
    coverage_metrics = [
        ("Altman Z-Score", "Altman_Z"),
        ("STA", "STA"),
        ("SNOA", "SNOA"),
        ("Gross Profitability", "Gross_Profitability"),
    ]

    coverage_rows = []

    for label, column in coverage_metrics:
        available = int(eligible[column].notna().sum())
        total = len(eligible)

        coverage_rows.append(
            {
                "Metric": label,
                "Available observations": available,
                "Missing observations": total - available,
                "Coverage (%)": (
                    available / total * 100 if total else 0
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(coverage_rows),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Important limitations"):
    st.markdown(
        """
        - This screen uses accounting ratios and a financial-distress model;
          it does not estimate intrinsic value or determine whether a stock
          is cheap relative to its future cash flows.
        - The Altman Z-Score is a model, not a guarantee of solvency.
          Its interpretation can be affected by data quality and company
          characteristics.
        - Lower STA and SNOA and higher gross profitability are screening
          preferences, not universal rules that apply identically to every
          manufacturer.
        - The percentile thresholds are configurable choices, not proven
          optimal parameters. Changing them changes the final candidate pool.
        - The order of the stages affects which companies survive.
        - Missing or stale financial data can materially affect results.
        - The app relies on the data in universe.csv and does not independently
          verify every company's financial statements.
        - Passing all stages means a company meets the chosen rules. It does
          not mean its stock is undervalued or that it should be purchased.
        """
    )

st.caption(
    "Results are calculated from universe.csv using the screening settings "
    "selected in the sidebar."
)
