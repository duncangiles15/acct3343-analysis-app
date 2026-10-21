
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Manufacturing Stock Screener",
    page_icon="📊",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "universe.csv"


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

@st.cache_data
def load_data(file_path: str, modified_time: float) -> pd.DataFrame:
    """Load the local universe CSV and normalize its columns."""

    df = pd.read_csv(file_path)

    # Prevent duplicate source column names from creating
    # duplicate-column errors later in Streamlit.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required_columns = [
        "Rank",
        "Name",
        "Market Cap",
        "Price",
        "Today",
        "Price (30 days)",
        "Country",
        "Ticker_guess",
        "Ticker",
        "Altman_Fiscal_Date",
        "X1_Working_Capital_Pct",
        "X2_Retained_Earnings_Pct",
        "X3_EBIT_Pct",
        "X4_Market_Equity_to_Liabilities_Pct",
        "X5_Sales_to_Assets",
        "Altman_Z",
        "Altman_Status",
        "Fiscal_Date",
        "STA",
        "SNOA",
        "STA_Status",
        "SNOA_Status",
        "GP_Fiscal_Date",
        "Gross_Profit",
        "Total_Assets",
        "Gross_Profitability",
        "GP_Status",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = np.nan

    # Convert numeric columns while preserving genuinely missing values.
    numeric_columns = [
        "Rank",
        "Price",
        "Today",
        "Price (30 days)",
        "X1_Working_Capital_Pct",
        "X2_Retained_Earnings_Pct",
        "X3_EBIT_Pct",
        "X4_Market_Equity_to_Liabilities_Pct",
        "X5_Sales_to_Assets",
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profit",
        "Total_Assets",
        "Gross_Profitability",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Normalize ticker fields.
    df["Ticker"] = df["Ticker"].astype("string").str.strip()
    df["Ticker_guess"] = df["Ticker_guess"].astype("string").str.strip()

    df["Display_Ticker"] = df["Ticker"].where(
        df["Ticker"].notna() & df["Ticker"].ne(""),
        df["Ticker_guess"],
    )

    df["Display_Ticker"] = df["Display_Ticker"].fillna("").astype(str)

    # If the preferred ticker fields are empty, use the company name
    # as a final identifier rather than inventing a ticker.
    df["Display_Ticker"] = df["Display_Ticker"].where(
        df["Display_Ticker"].str.strip().ne(""),
        df["Name"].fillna("Unknown company").astype(str),
    )

    df["Name"] = df["Name"].fillna("Unknown company").astype(str)
    df["Market Cap"] = df["Market Cap"].fillna("").astype(str)

    # Parse market capitalization strings such as:
    # $2.4T, $850B, $45.2M, or numeric values.
    def parse_market_cap(value):
        if pd.isna(value):
            return np.nan

        text = str(value).strip().upper()

        if text in {"", "-", "N/A", "NA", "NAN", "NONE"}:
            return np.nan

        text = (
            text.replace("$", "")
            .replace(",", "")
            .replace(" ", "")
        )

        multiplier = 1.0

        if text.endswith("T"):
            multiplier = 1_000_000_000_000
            text = text[:-1]
        elif text.endswith("B"):
            multiplier = 1_000_000_000
            text = text[:-1]
        elif text.endswith("M"):
            multiplier = 1_000_000
            text = text[:-1]
        elif text.endswith("K"):
            multiplier = 1_000
            text = text[:-1]

        try:
            return float(text) * multiplier
        except (ValueError, TypeError):
            return np.nan

    df["Market_Cap_Value"] = df["Market Cap"].apply(parse_market_cap)

    # Use the ticker as a stable sorting key.
    df["Sort_Ticker"] = df["Display_Ticker"].astype(str).str.upper()

    return df


if not DATA_FILE.exists():
    st.error(
        "The file `universe.csv` was not found. "
        "Place it in the same GitHub repository directory as `app.py`."
    )
    st.stop()

try:
    data = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as error:
    st.error(f"Unable to load universe.csv: {error}")
    st.stop()

if data.empty:
    st.error("The universe.csv file contains no observations.")
    st.stop()


# ============================================================
# FINANCIAL METRIC AND DISPLAY HELPERS
# ============================================================

def add_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate favourable-direction percentile scores.

    Higher Altman Z is preferred.
    Lower STA is preferred.
    Lower SNOA is preferred.
    Higher gross profitability is preferred.

    The composite is calculated only when all four raw metrics
    are available for a company.
    """

    result = df.copy()

    metric_columns = [
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profitability",
    ]

    # Each percentile score runs from 0 to 100.
    # A higher score means a more favourable relative position.
    result["Altman_Score"] = (
        result["Altman_Z"].rank(method="average", pct=True) * 100
    )

    result["STA_Score"] = (
        result["STA"].rank(method="average", pct=True, ascending=False) * 100
    )

    result["SNOA_Score"] = (
        result["SNOA"].rank(method="average", pct=True, ascending=False) * 100
    )

    result["GP_Score"] = (
        result["Gross_Profitability"].rank(method="average", pct=True) * 100
    )

    score_columns = [
        "Altman_Score",
        "STA_Score",
        "SNOA_Score",
        "GP_Score",
    ]

    # Do not treat missing data as a zero score.
    complete_metrics = result[metric_columns].notna().all(axis=1)

    result["Quality_Score"] = np.nan
    result.loc[complete_metrics, "Quality_Score"] = (
        result.loc[complete_metrics, score_columns].mean(axis=1)
    )

    return result


def stable_sort(
    df: pd.DataFrame,
    metric: str,
    ascending: bool,
) -> pd.DataFrame:
    """
    Sort by the requested metric and then ticker.

    The ticker is a deterministic secondary key when equal metric
    values fall across a retention boundary.
    """

    return df.sort_values(
        by=[metric, "Sort_Ticker"],
        ascending=[ascending, True],
        kind="mergesort",
        na_position="last",
    ).copy()


def format_market_cap(value):
    if pd.isna(value):
        return "—"

    value = float(value)

    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:,.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"

    return f"${value:,.0f}"


def format_ratio_as_percent(series: pd.Series) -> pd.Series:
    """Convert stored decimal ratios to percentage display values."""

    return series * 100


def prepare_display_table(
    df: pd.DataFrame,
    include_scores: bool = True,
) -> pd.DataFrame:
    """Build display tables from explicit columns to avoid duplicates."""

    columns = [
        "Display_Ticker",
        "Name",
        "Country",
        "Market Cap",
        "Price",
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profitability",
    ]

    if include_scores:
        columns.extend(
            [
                "Altman_Score",
                "STA_Score",
                "SNOA_Score",
                "GP_Score",
                "Quality_Score",
            ]
        )

    available_columns = [column for column in columns if column in df.columns]
    table = df[available_columns].copy()

    rename_map = {
        "Display_Ticker": "Ticker",
        "Name": "Company",
        "Country": "Country",
        "Market Cap": "Market Cap",
        "Price": "Price",
        "Altman_Z": "Altman Z",
        "STA": "STA (%)",
        "SNOA": "SNOA (%)",
        "Gross_Profitability": "Gross Profitability (%)",
        "Altman_Score": "Altman Z Percentile",
        "STA_Score": "STA Percentile",
        "SNOA_Score": "SNOA Percentile",
        "GP_Score": "Gross Profitability Percentile",
        "Quality_Score": "Composite Score",
    }

    table = table.rename(columns=rename_map)

    # Display ratios as percentages, but retain raw decimal values
    # in the underlying dataset and calculations.
    for column in ["STA (%)", "SNOA (%)", "Gross Profitability (%)"]:
        if column in table.columns:
            table[column] = format_ratio_as_percent(table[column])

    numeric_display_columns = [
        "Price",
        "Altman Z",
        "STA (%)",
        "SNOA (%)",
        "Gross Profitability (%)",
        "Altman Z Percentile",
        "STA Percentile",
        "SNOA Percentile",
        "Gross Profitability Percentile",
        "Composite Score",
    ]

    for column in numeric_display_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(2)

    # Final protection against duplicate display names.
    table = table.loc[:, ~table.columns.duplicated()]

    return table


def stage_summary(
    stage: str,
    entering: int,
    missing: int,
    excluded: int,
    remaining: int,
) -> dict:
    return {
        "Stage": stage,
        "Entering": entering,
        "Missing required metric": missing,
        "Excluded by filter": excluded,
        "Remaining": remaining,
    }


def retention_filter(
    entering_df: pd.DataFrame,
    metric: str,
    retention_percent: int,
    ascending: bool,
):
    """
    Apply a rank-based retention filter.

    Missing observations are removed first and counted separately.
    The number retained is floor(valid observations × retention rate).
    """

    entering_count = len(entering_df)

    valid = entering_df[entering_df[metric].notna()].copy()
    missing_count = entering_count - len(valid)

    ordered = stable_sort(valid, metric, ascending=ascending)

    keep_count = math.floor(len(ordered) * retention_percent / 100)
    keep_count = max(0, min(keep_count, len(ordered)))

    survivors = ordered.head(keep_count).copy()
    excluded = ordered.iloc[keep_count:].copy()

    summary = stage_summary(
        stage=metric,
        entering=entering_count,
        missing=missing_count,
        excluded=len(excluded),
        remaining=len(survivors),
    )

    cutoff = {
        "metric": metric,
        "retention_percent": retention_percent,
        "valid_count": len(ordered),
        "keep_count": keep_count,
        "excluded_count": len(excluded),
        "excluded": excluded,
        "survivors": survivors,
        "ascending": ascending,
    }

    return survivors, summary, cutoff


def show_cutoff_details(
    title: str,
    cutoff: dict,
    display_metric: str,
    unit: str = "",
):
    """Show the observations nearest the rank-retention boundary."""

    with st.expander(title):
        valid_count = cutoff["valid_count"]
        keep_count = cutoff["keep_count"]
        excluded_count = cutoff["excluded_count"]

        st.write(
            f"**Valid observations:** {valid_count:,}  \n"
            f"**Target retained:** {keep_count:,}  \n"
            f"**Excluded after ranking:** {excluded_count:,}"
        )

        if valid_count == 0:
            st.info("No valid observations were available at this stage.")
            return

        survivors = cutoff["survivors"]
        excluded = cutoff["excluded"]

        if survivors.empty:
            st.warning(
                "No observations were retained. The current retention "
                "percentage and number of valid observations result in "
                "a floor-based target of zero."
            )
            return

        if excluded.empty:
            st.info("All valid observations were retained.")
            return

        # The last excluded and first retained observations are the
        # companies immediately on either side of the cutoff.
        last_excluded = excluded.tail(1)
        first_retained = survivors.tail(1)

        excluded_value = last_excluded.iloc[0][cutoff["metric"]]
        retained_value = first_retained.iloc[0][cutoff["metric"]]

        excluded_ticker = last_excluded.iloc[0]["Display_Ticker"]
        retained_ticker = first_retained.iloc[0]["Display_Ticker"]

        if unit == "%":
            excluded_value *= 100
            retained_value *= 100

        left, right = st.columns(2)

        with left:
            st.markdown("**Last company excluded**")
            st.write(f"**{excluded_ticker}**")
            st.write(
                f"{display_metric}: {excluded_value:,.2f}{unit}"
            )

        with right:
            st.markdown("**Last company retained**")
            st.write(f"**{retained_ticker}**")
            st.write(
                f"{display_metric}: {retained_value:,.2f}{unit}"
            )

        st.caption(
            "Companies are ordered by the screening metric. "
            "Ticker is used as the secondary sort key, so the result "
            "is reproducible when values tie at the cutoff."
        )


# ============================================================
# CALCULATE PERCENTILE SCORES
# ============================================================

data = add_quality_scores(data)


# ============================================================
# HEADER AND METHODOLOGY NAVIGATION
# ============================================================

st.title("Manufacturing Stock Screener")

st.markdown(
    """
    A quantitative screening dashboard for manufacturing companies.
    The funnel applies financial filters in sequence, while the
    composite score compares companies on four financial measures.
    """
)

st.markdown(
    """
    **Quick navigation:** [View methodology, formulas and screening rules ↓](#methodology)
    """
)

st.caption(
    "Data source: the repository's `universe.csv` file. "
    "The screen is a quantitative research tool, not investment advice."
)


# ============================================================
# SIDEBAR: FUNNEL CONTROL
# ============================================================

st.sidebar.title("Funnel Control")

st.sidebar.markdown(
    """
    Adjust the settings below to change which companies pass through
    the screening funnel. Each stage starts with the companies that
    survived the preceding stage.
    """
)

market_cap_min_billions = st.sidebar.number_input(
    "Minimum market capitalization ($B)",
    min_value=0.0,
    max_value=10_000.0,
    value=0.5,
    step=0.5,
    help=(
        "Companies below this market capitalization are removed before "
        "the financial filters begin. For example, 0.5 means $500 million."
    ),
)

altman_threshold = st.sidebar.number_input(
    "Minimum Altman Z-score",
    min_value=-20.0,
    max_value=50.0,
    value=1.81,
    step=0.01,
    help=(
        "Companies with an Altman Z-score below this value are excluded. "
        "For manufacturing firms, 1.81 is a commonly cited distress-zone "
        "boundary."
    ),
)

st.sidebar.divider()
st.sidebar.subheader("Rank-Based Filters")

sta_retention = st.sidebar.slider(
    "% kept after STA filter",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help=(
        "The percentage of companies with valid STA values kept at this "
        "stage. Lower STA values are preferred because they indicate "
        "lower accruals relative to assets under this screening approach."
    ),
)

snoa_retention = st.sidebar.slider(
    "% kept after SNOA filter",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help=(
        "The percentage of companies with valid SNOA values kept at this "
        "stage. Lower SNOA values are preferred."
    ),
)

gp_retention = st.sidebar.slider(
    "% kept after gross profitability filter",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help=(
        "The percentage of companies with valid gross profitability values "
        "kept at this stage. Higher gross profitability is preferred."
    ),
)

st.sidebar.divider()
st.sidebar.caption(
    "Retention percentages apply to the valid observations entering each "
    "stage, after missing values have been removed. The retained count "
    "uses floor(valid observations × retention percentage)."
)


# ============================================================
# MARKET CAP ELIGIBILITY
# ============================================================

minimum_market_cap = market_cap_min_billions * 1_000_000_000

eligible = data[
    data["Market_Cap_Value"].notna()
    & (data["Market_Cap_Value"] >= minimum_market_cap)
].copy()

eligible_count = len(eligible)


# ============================================================
# SCREENING FUNNEL
# ============================================================

st.header("Screening Funnel")

st.markdown(
    """
    The funnel applies the filters in order. The results at each stage
    depend on the companies that passed all earlier stages.
    """
)

funnel_rows = []
cutoff_details = {}


# ------------------------------
# Stage 1: Altman Z
# ------------------------------

stage_entering = len(eligible)

altman_valid = eligible[eligible["Altman_Z"].notna()].copy()
altman_missing = stage_entering - len(altman_valid)

altman_passed = altman_valid[
    altman_valid["Altman_Z"] >= altman_threshold
].copy()

altman_excluded = altman_valid[
    altman_valid["Altman_Z"] < altman_threshold
].copy()

# A higher Altman Z is preferred, so list higher values first.
altman_excluded = stable_sort(
    altman_excluded,
    "Altman_Z",
    ascending=False,
)

altman_passed = stable_sort(
    altman_passed,
    "Altman_Z",
    ascending=False,
)

funnel_rows.append(
    stage_summary(
        "1. Altman Z threshold",
        stage_entering,
        altman_missing,
        len(altman_excluded),
        len(altman_passed),
    )
)

cutoff_details["Altman Z"] = {
    "metric": "Altman_Z",
    "threshold": altman_threshold,
    "missing_count": altman_missing,
    "excluded": altman_excluded,
    "survivors": altman_passed,
}

current = altman_passed.copy()


# ------------------------------
# Stage 2: STA
# ------------------------------

sta_survivors, sta_summary, sta_cutoff = retention_filter(
    current,
    metric="STA",
    retention_percent=sta_retention,
    ascending=True,
)

sta_summary["Stage"] = "2. STA"
funnel_rows.append(sta_summary)
cutoff_details["STA"] = sta_cutoff

current = sta_survivors.copy()


# ------------------------------
# Stage 3: SNOA
# ------------------------------

snoa_survivors, snoa_summary, snoa_cutoff = retention_filter(
    current,
    metric="SNOA",
    retention_percent=snoa_retention,
    ascending=True,
)

snoa_summary["Stage"] = "3. SNOA"
funnel_rows.append(snoa_summary)
cutoff_details["SNOA"] = snoa_cutoff

current = snoa_survivors.copy()


# ------------------------------
# Stage 4: Gross Profitability
# ------------------------------

gp_survivors, gp_summary, gp_cutoff = retention_filter(
    current,
    metric="Gross_Profitability",
    retention_percent=gp_retention,
    ascending=False,
)

gp_summary["Stage"] = "4. Gross Profitability"
funnel_rows.append(gp_summary)
cutoff_details["Gross Profitability"] = gp_cutoff

final_survivors = gp_survivors.copy()

# Add composite scores already calculated across the full dataset.
# Re-sort the final funnel by composite score for the candidate list.
final_survivors = final_survivors.sort_values(
    by=["Quality_Score", "Sort_Ticker"],
    ascending=[False, True],
    kind="mergesort",
)


# ============================================================
# FUNNEL SUMMARY TABLE AND CHART
# ============================================================

funnel_table = pd.DataFrame(funnel_rows)

st.dataframe(
    funnel_table,
    hide_index=True,
    use_container_width=True,
)

chart_data = funnel_table[["Stage", "Remaining"]].copy()

fig = px.bar(
    chart_data,
    x="Stage",
    y="Remaining",
    title="Companies Remaining After Each Stage",
    text="Remaining",
)

fig.update_traces(textposition="outside")
fig.update_layout(
    xaxis_title="Screening stage",
    yaxis_title="Companies remaining",
    showlegend=False,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig, use_container_width=True)

if not final_survivors.empty:
    st.success(
        f"**{len(final_survivors):,} companies passed all four financial "
        f"stages**, from {eligible_count:,} companies meeting the market "
        f"capitalization requirement."
    )
else:
    st.warning(
        "No companies passed all four stages with the current settings. "
        "Consider reviewing the retention percentages or Altman Z threshold."
    )


# ============================================================
# CUTOFF INSPECTION
# ============================================================

st.header("Cutoff Inspection")

st.markdown(
    """
    Review the companies immediately around each filter's boundary.
    This helps show which observations were retained or excluded and why.
    """
)

show_cutoff_details(
    title="Altman Z threshold details",
    cutoff=cutoff_details["Altman Z"],
    display_metric="Altman Z",
)

show_cutoff_details(
    title="STA retention cutoff",
    cutoff=cutoff_details["STA"],
    display_metric="STA",
    unit="%",
)

show_cutoff_details(
    title="SNOA retention cutoff",
    cutoff=cutoff_details["SNOA"],
    display_metric="SNOA",
    unit="%",
)

show_cutoff_details(
    title="Gross profitability retention cutoff",
    cutoff=cutoff_details["Gross Profitability"],
    display_metric="Gross profitability",
    unit="%",
)


# ============================================================
# TOP BUY CANDIDATES
# ============================================================

st.header("Top 10 BUY Candidates")

st.markdown(
    """
    These are the highest composite-scoring companies among those that
    passed the full funnel. The composite score summarizes relative
    performance across the four financial metrics; it is not a price
    target or a guarantee of future returns.
    """
)

buy_pool = final_survivors[
    final_survivors["Quality_Score"].notna()
].copy()

buy_pool = buy_pool.sort_values(
    by=["Quality_Score", "Sort_Ticker"],
    ascending=[False, True],
    kind="mergesort",
)

buy_candidates = buy_pool.head(10).copy()

if buy_candidates.empty:
    st.info(
        "No BUY candidates are available because no companies passed the "
        "funnel with a complete composite score."
    )
else:
    buy_table = prepare_display_table(buy_candidates)

    st.dataframe(
        buy_table,
        hide_index=True,
        use_container_width=True,
    )

    buy_csv = buy_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download BUY candidates as CSV",
        data=buy_csv,
        file_name="top_10_buy_candidates.csv",
        mime="text/csv",
        key="download_buy_candidates",
    )


# ============================================================
# TOP SELL CANDIDATES
# ============================================================

st.header("Top 10 SELL Candidates")

st.markdown(
    """
    These are the lowest composite-scoring companies among the market-cap
    eligible universe with all four metrics available. Unlike the BUY
    list, this list is not restricted to companies that passed the funnel.
    It is a relative comparison group, not a recommendation to short a stock.
    """
)

sell_pool = eligible[
    eligible["Quality_Score"].notna()
].copy()

sell_pool = sell_pool.sort_values(
    by=["Quality_Score", "Sort_Ticker"],
    ascending=[True, True],
    kind="mergesort",
)

sell_candidates = sell_pool.head(10).copy()

if sell_candidates.empty:
    st.info(
        "No SELL candidates are available because there are no eligible "
        "companies with all four financial metrics."
    )
else:
    sell_table = prepare_display_table(sell_candidates)

    st.dataframe(
        sell_table,
        hide_index=True,
        use_container_width=True,
    )

    sell_csv = sell_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download SELL candidates as CSV",
        data=sell_csv,
        file_name="top_10_sell_candidates.csv",
        mime="text/csv",
        key="download_sell_candidates",
    )


# ============================================================
# COMPOSITE SCORE EXPLANATION
# ============================================================

st.header("Composite Quality Score")

st.markdown(
    """
    The composite score is an equal-weight average of four percentile
    scores. Each percentile score is oriented so that a higher score
    represents a more favourable relative position.
    """
)

score_columns = [
    "Altman_Score",
    "STA_Score",
    "SNOA_Score",
    "GP_Score",
]

composite_formula = (
    "Composite\\ Score = "
    "\\frac{Altman\\ Z\\ Percentile + STA\\ Percentile + "
    "SNOA\\ Percentile + GP\\ Percentile}{4}"
)

st.latex(composite_formula)

st.markdown(
    """
    **Score interpretation**

    - **Altman Z percentile:** higher Altman Z receives a higher score.
    - **STA percentile:** lower STA receives a higher score.
    - **SNOA percentile:** lower SNOA receives a higher score.
    - **Gross profitability percentile:** higher gross profitability
      receives a higher score.
    - The composite score is available only when all four underlying
      financial metrics are present.
    - Percentile scores are relative to the dataset used for the
      calculation. They are not probabilities, expected returns, or
      absolute measures of investment quality.
    """
)


# ============================================================
# STAGE SURVIVORS AND EXPORT
# ============================================================

st.header("Inspect Companies at Each Stage")

stage_options = {
    "Market-cap eligible universe": eligible,
    "Passed Altman Z threshold": altman_passed,
    "Passed STA filter": sta_survivors,
    "Passed SNOA filter": snoa_survivors,
    "Passed gross profitability filter": final_survivors,
}

selected_stage = st.selectbox(
    "Select a group of companies to inspect",
    list(stage_options.keys()),
)

stage_df = stage_options[selected_stage].copy()

if stage_df.empty:
    st.info("There are no companies in this group with the current settings.")
else:
    stage_display = prepare_display_table(stage_df)

    st.caption(f"{len(stage_df):,} companies in this group.")

    st.dataframe(
        stage_display,
        hide_index=True,
        use_container_width=True,
    )

    stage_csv = stage_display.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download this group as CSV",
        data=stage_csv,
        file_name="screening_stage_companies.csv",
        mime="text/csv",
        key="download_stage_companies",
    )


# ============================================================
# DATA COVERAGE
# ============================================================

st.header("Data Coverage")

st.markdown(
    """
    Coverage indicates how many companies have each metric available.
    A company missing a required metric is excluded at the relevant
    stage rather than being assigned an artificial zero.
    """
)

coverage_metrics = {
    "Altman Z": "Altman_Z",
    "STA": "STA",
    "SNOA": "SNOA",
    "Gross Profitability": "Gross_Profitability",
}

coverage_rows = []

for label, column in coverage_metrics.items():
    available_count = int(data[column].notna().sum())
    missing_count = int(data[column].isna().sum())
    total_count = len(data)

    coverage_rows.append(
        {
            "Metric": label,
            "Available": available_count,
            "Missing": missing_count,
            "Coverage (%)": (
                available_count / total_count * 100 if total_count else np.nan
            ),
        }
    )

coverage_table = pd.DataFrame(coverage_rows)
coverage_table["Coverage (%)"] = coverage_table["Coverage (%)"].round(1)

st.dataframe(
    coverage_table,
    hide_index=True,
    use_container_width=True,
)

complete_count = int(
    data[
        data[
            [
                "Altman_Z",
                "STA",
                "SNOA",
                "Gross_Profitability",
            ]
        ].notna().all(axis=1)
    ].shape[0]
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Companies in source file", f"{len(data):,}")

with metric_col2:
    st.metric("Market-cap eligible", f"{eligible_count:,}")

with metric_col3:
    st.metric("Complete on all four metrics", f"{complete_count:,}")


# ============================================================
# METHODOLOGY
# ============================================================

st.markdown('<div id="methodology"></div>', unsafe_allow_html=True)

st.header("Methodology")

st.markdown(
    """
    This section documents the formulas, screening sequence, ranking
    rules, and limitations used by the dashboard.
    """
)

with st.expander("1. Universe and market capitalization", expanded=True):
    st.markdown(
        """
        The screen starts with the companies contained in `universe.csv`.
        The market-cap control removes companies whose parsed market
        capitalization is below the selected minimum.

        Market capitalization is expressed in dollars internally.
        Values such as `$850B` and `$2.4T` are converted to numeric values
        for filtering. Companies with an unavailable or unparseable market
        capitalization cannot pass this initial eligibility step.
        """
    )

with st.expander("2. Altman Z-score", expanded=True):
    st.markdown(
        """
        The manufacturing version of the Altman Z-score combines five
        accounting and market-based ratios. The dashboard uses the
        `Altman_Z` value supplied in the dataset for screening.
        """
    )

    st.latex(
        r"Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5"
    )

    st.markdown("The five components are:")

    st.latex(
        r"X_1 = \frac{\text{Working Capital}}{\text{Total Assets}}"
    )
    st.latex(
        r"X_2 = \frac{\text{Retained Earnings}}{\text{Total Assets}}"
    )
    st.latex(
        r"X_3 = \frac{\text{EBIT}}{\text{Total Assets}}"
    )
    st.latex(
        r"X_4 = \frac{\text{Market Value of Equity}}{\text{Total Liabilities}}"
    )
    st.latex(
        r"X_5 = \frac{\text{Sales}}{\text{Total Assets}}"
    )

    st.markdown(
        """
        **Commonly cited manufacturing-company reference zones**

        - **Z below 1.81:** distress zone.
        - **Z from 1.81 to 2.99:** grey zone.
        - **Z above 2.99:** safe zone.

        These are reference boundaries, not guarantees of bankruptcy or
        financial safety. The dashboard's selected minimum is a screening
        parameter. At the default of 1.81, companies with Z-scores below
        1.81 are excluded.

        The ratios are used as inputs to the Z-score formula; the existing
        Z-score is not recalculated by this app. This avoids silently
        changing the source dataset's calculations.
        """
    )

with st.expander("3. STA: total accruals scaled by assets", expanded=True):
    st.markdown(
        """
        STA is used here as an accrual-quality measure. Lower values are
        preferred by this screen. The exact implementation follows the
        variables supplied in the dataset.
        """
    )

    st.latex(
        r"STA = \frac{\text{Net Income} - \text{Operating Cash Flow}}"
        r"{\text{Total Assets}}"
    )

    st.markdown(
        """
        **Screening direction:** ascending STA, so the lowest valid values
        are retained first.

        STA is not interpreted in isolation. Its meaning can depend on the
        company's business model, accounting choices, and reporting period.
        """
    )

with st.expander("4. SNOA: scaled net operating assets", expanded=True):
    st.markdown(
        """
        SNOA measures net operating assets relative to total assets.
        Lower values are preferred by this screen.
        """
    )

    st.latex(
        r"SNOA = \frac{\text{Operating Assets} - \text{Operating Liabilities}}"
        r"{\text{Total Assets}}"
    )

    st.markdown(
        """
        **Screening direction:** ascending SNOA, so the lowest valid values
        are retained first.

        Operating-asset and operating-liability definitions can vary across
        data providers. The dashboard uses the SNOA values already stored
        in the dataset.
        """
    )

with st.expander("5. Gross profitability", expanded=True):
    st.markdown(
        """
        Gross profitability measures gross profit relative to the asset base.
        Higher values are preferred by this screen.
        """
    )

    st.latex(
        r"\text{Gross Profitability} = "
        r"\frac{\text{Gross Profit}}{\text{Total Assets}}"
    )

    st.markdown(
        """
        **Screening direction:** descending gross profitability, so the
        highest valid values are retained first.

        The app uses the `Gross_Profitability` field in the dataset rather
        than recalculating it from raw financial statements.
        """
    )

with st.expander("6. Sequential funnel and retention percentages", expanded=True):
    st.markdown(
        """
        The screening sequence is:

        1. Minimum market capitalization.
        2. Altman Z-score threshold.
        3. STA rank-retention filter.
        4. SNOA rank-retention filter.
        5. Gross profitability rank-retention filter.

        Each stage operates only on the companies that survived the previous
        stage. Therefore, changing an earlier filter can change the
        companies and counts entering every later filter.
        """
    )

    st.markdown("For a rank-based stage, the target number retained is:")

    st.latex(
        r"N_{\text{retained}} = "
        r"\left\lfloor N_{\text{valid}} \times "
        r"\frac{p}{100} \right\rfloor"
    )

    st.markdown(
        """
        Where:

        - \(N_{\\text{valid}}\) is the number of companies entering that
          stage with a non-missing value for its required metric.
        - \(p\) is the selected retention percentage.
        - The floor operation rounds down to the nearest whole company.

        Missing observations are removed and counted separately from
        companies excluded by the ranking cutoff. The retention percentage
        is applied to valid observations, not to the original universe size.

        When metric values tie across the retention boundary, ticker is
        used as a deterministic secondary sort key. This makes the result
        reproducible, although it means companies with identical metric
        values can fall on opposite sides of the cutoff.
        """
    )

with st.expander("7. Percentile scores and composite quality score", expanded=True):
    st.markdown(
        """
        The composite score compares eligible companies on four metrics.
        Each score is a percentile rank within the dataset and is oriented
        so that a higher score indicates a more favourable relative position.
        """
    )

    st.latex(
        r"\text{Composite Score} = "
        r"\frac{S_Z + S_{STA} + S_{SNOA} + S_{GP}}{4}"
    )

    st.markdown(
        """
        Where:

        - \(S_Z\) is the Altman Z percentile score; higher Z is preferred.
        - \(S_{STA}\) is the STA percentile score; lower STA is preferred.
        - \(S_{SNOA}\) is the SNOA percentile score; lower SNOA is preferred.
        - \(S_{GP}\) is the gross profitability percentile score; higher
          gross profitability is preferred.

        Each component is scaled from 0 to 100. The four components receive
        equal weight. The composite score is calculated only when all four
        underlying metrics are present.

        Percentile scores depend on the companies included in the ranking.
        They are relative measures, not probabilities, expected returns,
        valuation targets, or absolute ratings.
        """
    )

with st.expander("8. BUY and SELL candidate lists", expanded=True):
    st.markdown(
        """
        **Top 10 BUY candidates:** the ten highest composite-scoring
        companies among the final funnel survivors with a complete score.

        **Top 10 SELL candidates:** the ten lowest composite-scoring
        companies among the market-cap eligible universe with a complete
        score. The SELL list is not restricted to companies that passed
        the funnel.

        These labels describe the output of this particular quantitative
        screen. They do not constitute individualized investment advice,
        and they do not account for every factor relevant to an investment
        decision.
        """
    )

with st.expander("9. Data coverage and limitations", expanded=True):
    st.markdown(
        """
        - The app uses the local `universe.csv` file; it does not retrieve
          live market or financial-statement data.
        - The financial metrics are used as provided in the source file.
          The app does not independently audit company filings or provider
          calculations.
        - Missing values remain missing and are not replaced with zero.
        - Different financial data providers can use different definitions,
          fiscal dates, and calculation conventions.
        - The manufacturing Altman Z formulation is not necessarily
          appropriate for every company, capital structure, or industry.
        - The funnel and composite score are screening tools, not substitutes
          for reviewing financial statements, valuation, liquidity, debt,
          competitive position, and company-specific risks.
        """
    )

st.caption(
    "Manufacturing Stock Screener • Quantitative screening only • "
    "Review the underlying data and company filings before making "
    "investment decisions."
)
