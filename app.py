
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
    page_title="Equity Screening Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("Equity Screening & Investment Analysis")
st.markdown(
    """
    A systematic stock-screening model that evaluates companies using four
    financial metrics: **STA, SNOA, Altman Z-Score, and Gross Profitability
    (GPA)**. Each metric contributes a BUY, SELL, or neutral signal to the
    final classification.
    """
)


# ============================================================
# DATA LOADING
# ============================================================

DATA_FILE = Path(__file__).parent / "universe.csv"


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Normalize column names without changing the underlying dataset.
    data.columns = data.columns.str.strip()

    required_columns = [
        "Name",
        "Ticker",
        "Market Cap",
        "Price",
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profitability",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Convert financial columns to numeric while preserving missing values.
    numeric_columns = [
        "Market Cap",
        "Price",
        "Today",
        "Price (30 days)",
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profitability",
        "Gross_Profit",
        "Total_Assets",
        "X1_Working_Capital_Pct",
        "X2_Retained_Earnings_Pct",
        "X3_EBIT_Pct",
        "X4_Market_Equity_to_Liabilities_Pct",
        "X5_Sales_to_Assets",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    data["Ticker"] = data["Ticker"].fillna("").astype(str).str.strip()
    data["Name"] = data["Name"].fillna("Unknown company").astype(str)

    # Use a guessed ticker if the primary ticker is unavailable.
    if "Ticker_guess" in data.columns:
        guessed = data["Ticker_guess"].fillna("").astype(str).str.strip()
        data["Ticker"] = data["Ticker"].where(
            data["Ticker"].ne(""), guessed
        )

    return data


try:
    df = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def percentile_flag(series, percentile, direction):
    """
    Flag companies based on the distribution of a financial metric.

    direction='high':
        Flag values at or above the selected upper percentile threshold.

    direction='low':
        Flag values at or below the selected lower percentile threshold.

    Missing values never receive a signal.
    """
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()

    result = pd.Series(False, index=series.index, dtype=bool)

    if valid.empty:
        return result

    if direction == "high":
        threshold = valid.quantile(1 - percentile / 100)
        result.loc[valid.index] = valid >= threshold
    else:
        threshold = valid.quantile(percentile / 100)
        result.loc[valid.index] = valid <= threshold

    return result


def format_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_market_cap(value):
    if pd.isna(value):
        return "N/A"

    absolute_value = abs(value)

    if absolute_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if absolute_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if absolute_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    return f"${value:,.0f}"


def display_table(data, columns, rename=None):
    available = [column for column in columns if column in data.columns]
    result = data[available].copy()

    if rename:
        result = result.rename(columns=rename)

    st.dataframe(result, use_container_width=True, hide_index=True)


def metric_signal_summary(data, buy_column, sell_column):
    return pd.DataFrame(
        {
            "Signal": ["BUY", "SELL", "Neutral"],
            "Companies": [
                int(data[buy_column].sum()),
                int(data[sell_column].sum()),
                int((~data[buy_column] & ~data[sell_column]).sum()),
            ],
        }
    )


# ============================================================
# SIDEBAR: MODEL SETTINGS
# ============================================================

st.sidebar.header("Screening Settings")
st.sidebar.caption(
    "Adjust the screening rules. The results and rankings update automatically."
)

market_cap_values = df["Market Cap"].dropna()

if not market_cap_values.empty:
    minimum_market_cap = st.sidebar.number_input(
        "Minimum market capitalization ($)",
        min_value=0,
        max_value=int(max(market_cap_values.max(), 1)),
        value=0,
        step=10_000_000,
        help="Exclude companies below this market capitalization.",
    )
else:
    minimum_market_cap = 0

st.sidebar.subheader("Percentile-Based Screens")

gpa_percentile = st.sidebar.slider(
    "GPA BUY / SELL cutoff (%)",
    min_value=5,
    max_value=40,
    value=20,
    step=5,
    help="BUY: top percentile. SELL: bottom percentile.",
)

sta_percentile = st.sidebar.slider(
    "STA BUY / SELL cutoff (%)",
    min_value=5,
    max_value=40,
    value=20,
    step=5,
    help="BUY: lowest STA values. SELL: highest STA values.",
)

snoa_percentile = st.sidebar.slider(
    "SNOA BUY / SELL cutoff (%)",
    min_value=5,
    max_value=40,
    value=20,
    step=5,
    help="BUY: lowest SNOA values. SELL: highest SNOA values.",
)

st.sidebar.subheader("Altman Z-Score")

distress_threshold = st.sidebar.number_input(
    "Distress threshold",
    min_value=0.0,
    max_value=5.0,
    value=1.81,
    step=0.01,
    format="%.2f",
)

safe_threshold = st.sidebar.number_input(
    "Safe-zone threshold",
    min_value=0.0,
    max_value=10.0,
    value=2.99,
    step=0.01,
    format="%.2f",
)

if safe_threshold <= distress_threshold:
    st.sidebar.error(
        "The safe-zone threshold must exceed the distress threshold."
    )
    st.stop()

st.sidebar.subheader("Final Classification")

minimum_signals = st.sidebar.slider(
    "Minimum matching signals required",
    min_value=1,
    max_value=4,
    value=2,
    help=(
        "A company needs at least this many BUY signals or SELL signals, "
        "and more signals in that direction than in the opposite direction."
    ),
)

st.sidebar.divider()
st.sidebar.caption(
    "Percentile cutoffs are calculated from the companies remaining "
    "after the market-cap filter."
)


# ============================================================
# STEP 1: DEFINE THE INVESTMENT UNIVERSE
# ============================================================

st.header("Step 1 — Define the Investment Universe")

st.write(
    """
    The screening process begins with the companies in the supplied dataset.
    A minimum market-cap filter can reduce the universe before the four
    financial screens are applied. Companies with missing metric values
    remain in the dataset but do not receive a signal for that metric.
    """
)

universe = df.copy()

if minimum_market_cap > 0:
    universe = universe[
        universe["Market Cap"].ge(minimum_market_cap)
        & universe["Market Cap"].notna()
    ].copy()

if universe.empty:
    st.warning("No companies meet the current universe filter.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Companies in Universe", f"{len(universe):,}")

with col2:
    st.metric(
        "With Altman Z-Score",
        f"{universe['Altman_Z'].notna().sum():,}",
    )

with col3:
    st.metric("With STA", f"{universe['STA'].notna().sum():,}")

with col4:
    st.metric(
        "With SNOA",
        f"{universe['SNOA'].notna().sum():,}",
    )

st.caption(
    f"Companies with GPA data: "
    f"{universe['Gross_Profitability'].notna().sum():,}"
)


# ============================================================
# CALCULATE THE FOUR SIGNALS
# The signals are calculated before the sections are displayed,
# so the final classification always uses all four metrics.
# ============================================================

screened = universe.copy()

# ------------------------------------------------------------
# STA: lower values are favorable
# ------------------------------------------------------------

screened["STA_BUY"] = percentile_flag(
    screened["STA"], sta_percentile, "low"
)
screened["STA_SELL"] = percentile_flag(
    screened["STA"], sta_percentile, "high"
)

# ------------------------------------------------------------
# SNOA: lower values are favorable
# ------------------------------------------------------------

screened["SNOA_BUY"] = percentile_flag(
    screened["SNOA"], snoa_percentile, "low"
)
screened["SNOA_SELL"] = percentile_flag(
    screened["SNOA"], snoa_percentile, "high"
)

# ------------------------------------------------------------
# ALTMAN Z: explicit distress / grey-zone / safe-zone rules
# ------------------------------------------------------------

screened["Altman_BUY"] = (
    screened["Altman_Z"].notna()
    & (screened["Altman_Z"] > safe_threshold)
)

screened["Altman_SELL"] = (
    screened["Altman_Z"].notna()
    & (screened["Altman_Z"] < distress_threshold)
)

screened["Altman_Neutral"] = (
    screened["Altman_Z"].notna()
    & (screened["Altman_Z"] >= distress_threshold)
    & (screened["Altman_Z"] <= safe_threshold)
)

# ------------------------------------------------------------
# GPA: higher values are favorable
# ------------------------------------------------------------

screened["GPA_BUY"] = percentile_flag(
    screened["Gross_Profitability"], gpa_percentile, "high"
)
screened["GPA_SELL"] = percentile_flag(
    screened["Gross_Profitability"], gpa_percentile, "low"
)

# ------------------------------------------------------------
# COMBINE ALL FOUR METRICS
# ------------------------------------------------------------

buy_signal_columns = [
    "STA_BUY",
    "SNOA_BUY",
    "Altman_BUY",
    "GPA_BUY",
]

sell_signal_columns = [
    "STA_SELL",
    "SNOA_SELL",
    "Altman_SELL",
    "GPA_SELL",
]

screened["BUY_Signals"] = screened[buy_signal_columns].sum(axis=1)
screened["SELL_Signals"] = screened[sell_signal_columns].sum(axis=1)

screened["Available_Metrics"] = (
    screened[
        [
            "STA",
            "SNOA",
            "Altman_Z",
            "Gross_Profitability",
        ]
    ]
    .notna()
    .sum(axis=1)
)

screened["Net_Signals"] = (
    screened["BUY_Signals"] - screened["SELL_Signals"]
)

screened["Classification"] = "REVIEW"

buy_condition = (
    (screened["BUY_Signals"] >= minimum_signals)
    & (screened["BUY_Signals"] > screened["SELL_Signals"])
)

sell_condition = (
    (screened["SELL_Signals"] >= minimum_signals)
    & (screened["SELL_Signals"] > screened["BUY_Signals"])
)

screened.loc[buy_condition, "Classification"] = "BUY"
screened.loc[sell_condition, "Classification"] = "SELL"


# ============================================================
# STEP 2: STA
# ============================================================

st.divider()
st.header("Step 2 — Screen Earnings Quality Using STA")

st.markdown(
    """
    **What it measures**

    STA is calculated as:

    `STA = (Net Income − Operating Cash Flow) / Total Assets`

    It helps identify differences between reported accounting earnings and
    operating cash flow. A lower STA value is treated as a more favorable
    signal in this screening model, while a higher value is treated as less
    favorable.

    **Active rule**

    - BUY: STA is in the lowest selected percentile.
    - SELL: STA is in the highest selected percentile.
    - Neutral: STA falls between those cutoffs, or the value is missing.

    These are relative screening rules, not proof of fraud or a definitive
    measure of earnings quality.
    """
)

sta_buy_count = int(screened["STA_BUY"].sum())
sta_sell_count = int(screened["STA_SELL"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Companies with STA", f"{screened['STA'].notna().sum():,}")
c2.metric("STA BUY Signals", f"{sta_buy_count:,}")
c3.metric("STA SELL Signals", f"{sta_sell_count:,}")

sta_tab1, sta_tab2 = st.tabs(["BUY Signals", "SELL Signals"])

with sta_tab1:
    st.caption("Companies with the lowest STA values.")
    sta_buy_results = screened[screened["STA_BUY"]].sort_values(
        "STA", ascending=True
    )
    display_table(
        sta_buy_results,
        ["Name", "Ticker", "Market Cap", "STA", "Altman_Z"],
        {
            "Name": "Company",
            "Ticker": "Ticker",
            "Market Cap": "Market Cap ($)",
            "STA": "STA",
            "Altman_Z": "Altman Z-Score",
        },
    )

with sta_tab2:
    st.caption("Companies with the highest STA values.")
    sta_sell_results = screened[screened["STA_SELL"]].sort_values(
        "STA", ascending=False
    )
    display_table(
        sta_sell_results,
        ["Name", "Ticker", "Market Cap", "STA", "Altman_Z"],
        {
            "Name": "Company",
            "Market Cap": "Market Cap ($)",
            "Altman_Z": "Altman Z-Score",
        },
    )


# ============================================================
# STEP 3: SNOA
# ============================================================

st.divider()
st.header("Step 3 — Screen Operating Assets Using SNOA")

st.markdown(
    """
    **What it measures**

    SNOA represents scaled net operating assets in the supplied dataset.
    The existing dataset calculation is retained:

    `SNOA = (Operating Assets − Operating Liabilities) / Total Assets`

    Lower SNOA values are treated as more favorable for this screen, while
    higher values are treated as less favorable.

    **Active rule**

    - BUY: SNOA is in the lowest selected percentile.
    - SELL: SNOA is in the highest selected percentile.
    - Neutral: SNOA falls between those cutoffs, or the value is missing.

    SNOA is a screening signal, not a standalone valuation or a guarantee
    of future investment performance.
    """
)

c1, c2, c3 = st.columns(3)
c1.metric("Companies with SNOA", f"{screened['SNOA'].notna().sum():,}")
c2.metric("SNOA BUY Signals", f"{int(screened['SNOA_BUY'].sum()):,}")
c3.metric("SNOA SELL Signals", f"{int(screened['SNOA_SELL'].sum()):,}")

snoa_tab1, snoa_tab2 = st.tabs(["BUY Signals", "SELL Signals"])

with snoa_tab1:
    snoa_buy_results = screened[screened["SNOA_BUY"]].sort_values(
        "SNOA", ascending=True
    )
    display_table(
        snoa_buy_results,
        ["Name", "Ticker", "Market Cap", "SNOA", "STA"],
        {
            "Name": "Company",
            "Market Cap": "Market Cap ($)",
        },
    )

with snoa_tab2:
    snoa_sell_results = screened[screened["SNOA_SELL"]].sort_values(
        "SNOA", ascending=False
    )
    display_table(
        snoa_sell_results,
        ["Name", "Ticker", "Market Cap", "SNOA", "STA"],
        {
            "Name": "Company",
            "Market Cap": "Market Cap ($)",
        },
    )


# ============================================================
# STEP 4: ALTMAN Z-SCORE
# ============================================================

st.divider()
st.header("Step 4 — Assess Financial Distress Using Altman Z-Score")

st.markdown(
    """
    **What it measures**

    The Altman Z-Score combines financial ratios to estimate a company's
    financial-distress risk. The supplied dataset uses the manufacturing
    version of the model:

    `Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5`

    where the component ratios are supplied in the dataset.

    **Active rule**

    - **BUY signal:** Z-Score is strictly above the safe-zone threshold.
    - **SELL signal:** Z-Score is strictly below the distress threshold.
    - **Neutral:** Z-Score is between the thresholds, including both
      boundaries.
    - **Missing:** no signal.

    The traditional reference points are 1.81 and 2.99. The grey zone
    between them is deliberately neutral: it contributes neither a BUY nor
    a SELL signal. A high score contributes to BUY, and a distress score
    contributes to SELL.
    """
)

altman_valid = screened["Altman_Z"].dropna()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies with Z-Score", f"{len(altman_valid):,}")
c2.metric("Altman BUY Signals", f"{int(screened['Altman_BUY'].sum()):,}")
c3.metric("Altman SELL Signals", f"{int(screened['Altman_SELL'].sum()):,}")
c4.metric(
    "Grey-Zone Companies",
    f"{int(screened['Altman_Neutral'].sum()):,}",
)

if not altman_valid.empty:
    altman_fig = px.histogram(
        screened[screened["Altman_Z"].notna()],
        x="Altman_Z",
        nbins=40,
        title="Distribution of Altman Z-Scores",
        labels={"Altman_Z": "Altman Z-Score"},
    )

    altman_fig.add_vline(
        x=distress_threshold,
        line_dash="dash",
        annotation_text="Distress threshold",
    )

    altman_fig.add_vline(
        x=safe_threshold,
        line_dash="dash",
        annotation_text="Safe-zone threshold",
    )

    st.plotly_chart(altman_fig, use_container_width=True)

    altman_buy_tab, altman_sell_tab, altman_neutral_tab = st.tabs(
        ["BUY Signals", "SELL Signals", "Grey Zone"]
    )

    with altman_buy_tab:
        st.caption(
            f"Z-Score > {safe_threshold:.2f}. "
            "These companies receive one BUY signal."
        )
        results = screened[screened["Altman_BUY"]].sort_values(
            "Altman_Z", ascending=False
        )
        display_table(
            results,
            ["Name", "Ticker", "Market Cap", "Altman_Z", "STA", "SNOA"],
            {
                "Name": "Company",
                "Market Cap": "Market Cap ($)",
                "Altman_Z": "Altman Z-Score",
            },
        )

    with altman_sell_tab:
        st.caption(
            f"Z-Score < {distress_threshold:.2f}. "
            "These companies receive one SELL signal."
        )
        results = screened[screened["Altman_SELL"]].sort_values(
            "Altman_Z", ascending=True
        )
        display_table(
            results,
            ["Name", "Ticker", "Market Cap", "Altman_Z", "STA", "SNOA"],
            {
                "Name": "Company",
                "Market Cap": "Market Cap ($)",
                "Altman_Z": "Altman Z-Score",
            },
        )

    with altman_neutral_tab:
        st.caption(
            f"{distress_threshold:.2f} ≤ Z-Score ≤ {safe_threshold:.2f}. "
            "These companies receive neither an Altman BUY nor SELL signal."
        )
        results = screened[screened["Altman_Neutral"]].sort_values(
            "Altman_Z", ascending=True
        )
        display_table(
            results,
            ["Name", "Ticker", "Market Cap", "Altman_Z"],
            {
                "Name": "Company",
                "Market Cap": "Market Cap ($)",
                "Altman_Z": "Altman Z-Score",
            },
        )
else:
    st.info("No companies have an available Altman Z-Score.")


# ============================================================
# STEP 5: GROSS PROFITABILITY
# ============================================================

st.divider()
st.header("Step 5 — Screen for Gross Profitability (GPA)")

st.markdown(
    """
    **What it measures**

    Gross profitability is calculated as:

    `GPA = Gross Profit / Total Assets`

    It measures how much gross profit a company generates relative to its
    asset base. Higher values are treated as more favorable.

    **Active rule**

    - BUY: GPA is in the highest selected percentile.
    - SELL: GPA is in the lowest selected percentile.
    - Neutral: GPA falls between those cutoffs, or the value is missing.

    This relative screen helps compare profitability across the available
    universe, but it does not replace a full valuation analysis.
    """
)

c1, c2, c3 = st.columns(3)
c1.metric(
    "Companies with GPA",
    f"{screened['Gross_Profitability'].notna().sum():,}",
)
c2.metric("GPA BUY Signals", f"{int(screened['GPA_BUY'].sum()):,}")
c3.metric("GPA SELL Signals", f"{int(screened['GPA_SELL'].sum()):,}")

gpa_tab1, gpa_tab2 = st.tabs(["BUY Signals", "SELL Signals"])

with gpa_tab1:
    gpa_buy_results = screened[screened["GPA_BUY"]].sort_values(
        "Gross_Profitability", ascending=False
    )
    display_table(
        gpa_buy_results,
        [
            "Name",
            "Ticker",
            "Market Cap",
            "Gross_Profitability",
            "Altman_Z",
        ],
        {
            "Name": "Company",
            "Market Cap": "Market Cap ($)",
            "Gross_Profitability": "GPA",
            "Altman_Z": "Altman Z-Score",
        },
    )

with gpa_tab2:
    gpa_sell_results = screened[screened["GPA_SELL"]].sort_values(
        "Gross_Profitability", ascending=True
    )
    display_table(
        gpa_sell_results,
        [
            "Name",
            "Ticker",
            "Market Cap",
            "Gross_Profitability",
            "Altman_Z",
        ],
        {
            "Name": "Company",
            "Market Cap": "Market Cap ($)",
            "Gross_Profitability": "GPA",
            "Altman_Z": "Altman Z-Score",
        },
    )


# ============================================================
# STEP 6: COMBINE ALL FOUR SIGNALS
# ============================================================

st.divider()
st.header("Step 6 — Combine the Four Screening Signals")

st.markdown(
    """
    Each company can receive up to four BUY signals and four SELL signals:
    one from STA, one from SNOA, one from Altman Z-Score, and one from GPA.

    The final classification compares the number of signals in each
    direction. A company must meet the minimum matching-signal setting
    and have more signals in that direction than in the opposite direction.

    **Important:** a grey-zone Altman Z-Score contributes zero signals.
    Missing data also contributes zero signals; it is not automatically
    treated as favorable or unfavorable.
    """
)

methodology = pd.DataFrame(
    [
        {
            "Metric": "STA",
            "BUY rule": f"Lowest {sta_percentile}% of values",
            "SELL rule": f"Highest {sta_percentile}% of values",
            "Neutral rule": "Between cutoffs or missing",
        },
        {
            "Metric": "SNOA",
            "BUY rule": f"Lowest {snoa_percentile}% of values",
            "SELL rule": f"Highest {snoa_percentile}% of values",
            "Neutral rule": "Between cutoffs or missing",
        },
        {
            "Metric": "Altman Z-Score",
            "BUY rule": f"Z > {safe_threshold:.2f}",
            "SELL rule": f"Z < {distress_threshold:.2f}",
            "Neutral rule": (
                f"{distress_threshold:.2f} ≤ Z ≤ {safe_threshold:.2f}"
            ),
        },
        {
            "Metric": "GPA",
            "BUY rule": f"Highest {gpa_percentile}% of values",
            "SELL rule": f"Lowest {gpa_percentile}% of values",
            "Neutral rule": "Between cutoffs or missing",
        },
    ]
)

st.subheader("Current Screening Rules")
st.dataframe(methodology, use_container_width=True, hide_index=True)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric(
        "BUY Classified",
        f"{(screened['Classification'] == 'BUY').sum():,}",
    )

with summary_col2:
    st.metric(
        "SELL Classified",
        f"{(screened['Classification'] == 'SELL').sum():,}",
    )

with summary_col3:
    st.metric(
        "Review",
        f"{(screened['Classification'] == 'REVIEW').sum():,}",
    )

signal_counts = pd.DataFrame(
    {
        "Metric": ["STA", "SNOA", "Altman Z-Score", "GPA"],
        "BUY": [
            screened["STA_BUY"].sum(),
            screened["SNOA_BUY"].sum(),
            screened["Altman_BUY"].sum(),
            screened["GPA_BUY"].sum(),
        ],
        "SELL": [
            screened["STA_SELL"].sum(),
            screened["SNOA_SELL"].sum(),
            screened["Altman_SELL"].sum(),
            screened["GPA_SELL"].sum(),
        ],
    }
)

signal_long = signal_counts.melt(
    id_vars="Metric",
    var_name="Signal",
    value_name="Companies",
)

signal_fig = px.bar(
    signal_long,
    x="Metric",
    y="Companies",
    color="Signal",
    barmode="group",
    title="BUY and SELL Signals by Metric",
    text="Companies",
)

st.plotly_chart(signal_fig, use_container_width=True)

st.subheader("Overall Classification")

classification_counts = (
    screened["Classification"]
    .value_counts()
    .reindex(["BUY", "SELL", "REVIEW"], fill_value=0)
    .rename_axis("Classification")
    .reset_index(name="Companies")
)

classification_fig = px.bar(
    classification_counts,
    x="Classification",
    y="Companies",
    color="Classification",
    title="Companies by Final Classification",
    text="Companies",
)

st.plotly_chart(classification_fig, use_container_width=True)

st.subheader("All Companies — Combined Signals")

combined_columns = [
    "Name",
    "Ticker",
    "Market Cap",
    "Price",
    "STA",
    "STA_BUY",
    "STA_SELL",
    "SNOA",
    "SNOA_BUY",
    "SNOA_SELL",
    "Altman_Z",
    "Altman_BUY",
    "Altman_SELL",
    "Altman_Neutral",
    "Gross_Profitability",
    "GPA_BUY",
    "GPA_SELL",
    "BUY_Signals",
    "SELL_Signals",
    "Available_Metrics",
    "Classification",
]

combined_rename = {
    "Name": "Company",
    "Market Cap": "Market Cap ($)",
    "STA_BUY": "STA BUY",
    "STA_SELL": "STA SELL",
    "SNOA_BUY": "SNOA BUY",
    "SNOA_SELL": "SNOA SELL",
    "Altman_Z": "Altman Z-Score",
    "Altman_BUY": "Altman BUY",
    "Altman_SELL": "Altman SELL",
    "Altman_Neutral": "Altman Grey Zone",
    "Gross_Profitability": "GPA",
    "GPA_BUY": "GPA BUY",
    "GPA_SELL": "GPA SELL",
    "BUY_Signals": "BUY Signals",
    "SELL_Signals": "SELL Signals",
    "Available_Metrics": "Metrics Available",
}

st.dataframe(
    screened[combined_columns].rename(columns=combined_rename),
    use_container_width=True,
    hide_index=True,
)

combined_csv = screened[combined_columns].to_csv(index=False).encode("utf-8")

st.download_button(
    "Download All Screening Results (CSV)",
    data=combined_csv,
    file_name="combined_screening_results.csv",
    mime="text/csv",
)


# ============================================================
# STEP 7: TOP 10 BUY AND SELL CANDIDATES
# ============================================================

st.divider()
st.header("Step 7 — Top 10 BUY and SELL Candidates")

st.markdown(
    """
    These lists rank companies already classified as BUY or SELL. The ranking
    prioritizes the number of matching signals, then uses the underlying
    financial metrics as tie-breakers.

    The lists are model outputs for further research, not personalized
    investment advice. Check data dates, company circumstances, valuation,
    and sector differences before making an investment decision.
    """
)

buy_candidates = screened[
    screened["Classification"] == "BUY"
].copy()

sell_candidates = screened[
    screened["Classification"] == "SELL"
].copy()

# Rank BUYs: more BUY signals, fewer SELL signals, then favorable
# GPA, STA, SNOA, and Altman Z.
buy_candidates = buy_candidates.sort_values(
    by=[
        "BUY_Signals",
        "SELL_Signals",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
    ],
    ascending=[False, True, False, True, True, False],
    na_position="last",
)

# Rank SELLs: more SELL signals, fewer BUY signals, then unfavorable
# GPA, STA, SNOA, and Altman Z.
sell_candidates = sell_candidates.sort_values(
    by=[
        "SELL_Signals",
        "BUY_Signals",
        "Altman_Z",
        "Gross_Profitability",
        "STA",
        "SNOA",
    ],
    ascending=[False, True, True, True, False, False],
    na_position="last",
)

top_buy = buy_candidates.head(10)
top_sell = sell_candidates.head(10)

buy_tab, sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

rank_columns = [
    "Name",
    "Ticker",
    "Market Cap",
    "Price",
    "BUY_Signals",
    "SELL_Signals",
    "STA",
    "SNOA",
    "Altman_Z",
    "Gross_Profitability",
]

rank_rename = {
    "Name": "Company",
    "Market Cap": "Market Cap ($)",
    "BUY_Signals": "BUY Signals",
    "SELL_Signals": "SELL Signals",
    "Altman_Z": "Altman Z-Score",
    "Gross_Profitability": "GPA",
}

with buy_tab:
    st.subheader("Top 10 BUY Candidates")
    st.caption(
        "Companies with the strongest positive signal counts under the "
        "current settings."
    )

    if top_buy.empty:
        st.info(
            "No companies currently meet the BUY classification rules. "
            "Try reviewing the percentile cutoffs or minimum signal setting."
        )
    else:
        display_table(top_buy, rank_columns, rank_rename)

        st.download_button(
            "Download Top 10 BUY (CSV)",
            data=top_buy[rank_columns].to_csv(index=False).encode("utf-8"),
            file_name="top_10_buy.csv",
            mime="text/csv",
        )

with sell_tab:
    st.subheader("Top 10 SELL Candidates")
    st.caption(
        "Companies with the strongest negative signal counts under the "
        "current settings."
    )

    if top_sell.empty:
        st.info(
            "No companies currently meet the SELL classification rules. "
            "Try reviewing the percentile cutoffs or minimum signal setting."
        )
    else:
        display_table(top_sell, rank_columns, rank_rename)

        st.download_button(
            "Download Top 10 SELL (CSV)",
            data=top_sell[rank_columns].to_csv(index=False).encode("utf-8"),
            file_name="top_10_sell.csv",
            mime="text/csv",
        )


# ============================================================
# COMPANY EXPLORER
# ============================================================

st.divider()
st.header("Company Explorer")

company_options = screened.copy()
company_options["Company_Label"] = (
    company_options["Ticker"].replace("", np.nan).fillna("No ticker")
    + " — "
    + company_options["Name"]
)

company_options = company_options.sort_values("Company_Label")

selected_label = st.selectbox(
    "Select a company to inspect",
    company_options["Company_Label"].tolist(),
)

selected_company = company_options[
    company_options["Company_Label"] == selected_label
].iloc[0]

company_col1, company_col2, company_col3, company_col4 = st.columns(4)

with company_col1:
    st.metric("Classification", selected_company["Classification"])

with company_col2:
    st.metric("BUY Signals", int(selected_company["BUY_Signals"]))

with company_col3:
    st.metric("SELL Signals", int(selected_company["SELL_Signals"]))

with company_col4:
    st.metric(
        "Available Metrics",
        f"{int(selected_company['Available_Metrics'])} / 4",
    )

st.subheader("Company Financial Metrics")

company_metrics = pd.DataFrame(
    {
        "Metric": [
            "STA",
            "SNOA",
            "Altman Z-Score",
            "Gross Profitability (GPA)",
        ],
        "Value": [
            format_number(selected_company["STA"], 4),
            format_number(selected_company["SNOA"], 4),
            format_number(selected_company["Altman_Z"], 2),
            format_number(selected_company["Gross_Profitability"], 4),
        ],
        "BUY Signal": [
            bool(selected_company["STA_BUY"]),
            bool(selected_company["SNOA_BUY"]),
            bool(selected_company["Altman_BUY"]),
            bool(selected_company["GPA_BUY"]),
        ],
        "SELL Signal": [
            bool(selected_company["STA_SELL"]),
            bool(selected_company["SNOA_SELL"]),
            bool(selected_company["Altman_SELL"]),
            bool(selected_company["GPA_SELL"]),
        ],
    }
)

st.dataframe(company_metrics, use_container_width=True, hide_index=True)

st.subheader("Why Was This Company Classified This Way?")

buy_count = int(selected_company["BUY_Signals"])
sell_count = int(selected_company["SELL_Signals"])

st.write(
    f"This company has **{buy_count} BUY signals** and "
    f"**{sell_count} SELL signals**, with "
    f"**{int(selected_company['Available_Metrics'])} of 4 metrics available**."
)

if pd.notna(selected_company["Altman_Z"]):
    z_value = selected_company["Altman_Z"]

    if z_value > safe_threshold:
        st.write(
            f"- **Altman Z-Score:** {z_value:.2f} — BUY signal because "
            f"the score is above {safe_threshold:.2f}."
        )
    elif z_value < distress_threshold:
        st.write(
            f"- **Altman Z-Score:** {z_value:.2f} — SELL signal because "
            f"the score is below {distress_threshold:.2f}."
        )
    else:
        st.write(
            f"- **Altman Z-Score:** {z_value:.2f} — neutral because the "
            f"score is within the grey zone "
            f"({distress_threshold:.2f} to {safe_threshold:.2f})."
        )
else:
    st.write("- **Altman Z-Score:** unavailable; no signal assigned.")

if selected_company["Classification"] == "BUY":
    st.success(
        "The company meets the BUY rules under the current model settings."
    )
elif selected_company["Classification"] == "SELL":
    st.error(
        "The company meets the SELL rules under the current model settings."
    )
else:
    st.info(
        "The company does not currently meet the final BUY or SELL rules. "
        "It may have mixed signals, insufficient matching signals, or "
        "missing metrics."
    )

# Show source dates when present.
date_columns = [
    ("STA", "Fiscal_Date"),
    ("SNOA", "Fiscal_Date"),
    ("Altman Z-Score", "Altman_Fiscal_Date"),
    ("GPA", "GP_Fiscal_Date"),
]

date_rows = []

for metric_name, date_column in date_columns:
    if date_column in screened.columns:
        date_value = selected_company.get(date_column, None)
        date_rows.append(
            {
                "Metric": metric_name,
                "Source Date": (
                    str(date_value)
                    if pd.notna(date_value) and str(date_value).strip()
                    else "Not available"
                ),
            }
        )

if date_rows:
    st.subheader("Metric Data Dates")
    st.dataframe(
        pd.DataFrame(date_rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DATA COVERAGE & METHODOLOGY
# ============================================================

st.divider()
st.header("Data Coverage & Methodology")

st.markdown(
    """
    The screening model uses the financial metrics provided in `universe.csv`.
    It does not automatically refresh market prices or financial statements.

    **How to interpret the output**

    - Each of the four metrics contributes at most one BUY or one SELL signal.
    - A metric that is neutral or missing contributes no signal.
    - Altman Z-Score values in the grey zone are explicitly neutral.
    - BUY and SELL classifications depend on the minimum matching-signal
      setting and whether one signal direction exceeds the other.
    - Percentile rules are relative to the filtered investment universe.
    - The Top 10 rankings are conditional on the active settings.
    """
)

coverage = pd.DataFrame(
    {
        "Metric": [
            "STA",
            "SNOA",
            "Altman Z-Score",
            "Gross Profitability (GPA)",
        ],
        "Available Observations": [
            screened["STA"].notna().sum(),
            screened["SNOA"].notna().sum(),
            screened["Altman_Z"].notna().sum(),
            screened["Gross_Profitability"].notna().sum(),
        ],
    }
)

coverage["Coverage (%)"] = (
    coverage["Available Observations"] / len(screened) * 100
).round(1)

st.dataframe(coverage, use_container_width=True, hide_index=True)

st.caption(
    "This tool is an analytical screening aid, not a guarantee of returns. "
    "The Altman Z-Score formula and thresholds should be interpreted in "
    "the context of the model version and company type. Relative percentile "
    "signals can be affected by sector composition, outliers, and missing data."
)
