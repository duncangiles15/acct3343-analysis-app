
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Screening Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Stock Screening Dashboard")
st.caption(
    "A data-driven stock screener using Altman Z-Score, "
    "Gross Profitability, accruals, and operating assets."
)


# ============================================================
# DEFAULT SETTINGS AND SYNCHRONIZED WIDGET KEYS
# ============================================================

DEFAULTS = {
    "min_market_cap": 0.0,
    "gpa_pct": 10,
    "sta_pct": 10,
    "snoa_pct": 10,
    "safe_z": 2.99,
    "distress_z": 1.81,
    "min_signals": 2,
}

# IMPORTANT:
# These names must match the actual keys used by the sliders.
WIDGET_KEYS = {
    "min_market_cap": ["cap_sidebar", "cap_main"],
    "gpa_pct": ["gpa_pct_sidebar", "gpa_pct_main"],
    "sta_pct": ["sta_pct_sidebar", "sta_pct_main"],
    "snoa_pct": ["snoa_pct_sidebar", "snoa_pct_main"],
    "safe_z": ["safe_z_sidebar", "safe_z_main"],
    "distress_z": ["distress_z_sidebar", "distress_z_main"],
    "min_signals": ["signals_sidebar", "signals_main"],
}


def initialize_settings():
    """Initialize shared settings and both copies of each widget."""
    for setting, default in DEFAULTS.items():
        if setting not in st.session_state:
            st.session_state[setting] = default

        for widget_key in WIDGET_KEYS[setting]:
            if widget_key not in st.session_state:
                st.session_state[widget_key] = default


def sync_setting(setting, source_key):
    """Keep the main-page and sidebar versions of a setting synchronized."""
    value = st.session_state[source_key]
    st.session_state[setting] = value

    for widget_key in WIDGET_KEYS[setting]:
        st.session_state[widget_key] = value


initialize_settings()


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

DATA_FILE = Path(__file__).parent / "universe.csv"


@st.cache_data
def load_data(file_path, modified_time):
    """Load and clean the stock universe."""
    data = pd.read_csv(file_path)

    # Convert numeric columns that may have been read as text.
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
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip(),
                errors="coerce",
            )

    # Convert market-cap text such as "$2.4T", "$850B", "$45.2M".
    if "Market Cap" in data.columns:
        data["Market_Cap_Value"] = (
            data["Market Cap"].apply(parse_market_cap)
        )
    else:
        data["Market_Cap_Value"] = np.nan

    # Ensure text columns have consistent missing-value handling.
    for column in [
        "Ticker",
        "Ticker_guess",
        "Name",
        "Country",
        "Altman_Status",
        "STA_Status",
        "SNOA_Status",
        "GP_Status",
    ]:
        if column in data.columns:
            data[column] = data[column].fillna("").astype(str)

    # If Ticker is blank, fall back to Ticker_guess.
    if "Ticker" in data.columns and "Ticker_guess" in data.columns:
        blank_ticker = data["Ticker"].str.strip().eq("")
        data.loc[blank_ticker, "Ticker"] = data.loc[
            blank_ticker, "Ticker_guess"
        ]

    return data


def parse_market_cap(value):
    """Convert market-cap strings into numeric dollar values."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().upper()

    if text in {"", "NONE", "NAN", "N/A", "NA", "-"}:
        return np.nan

    text = text.replace(",", "").replace("$", "").replace(" ", "")

    multiplier = 1.0

    if text.endswith("T"):
        multiplier = 1e12
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1e9
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1e6
        text = text[:-1]
    elif text.endswith("K"):
        multiplier = 1e3
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


if not DATA_FILE.exists():
    st.error(
        "Could not find universe.csv. Make sure it is in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()

data = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)

if data.empty:
    st.error("The stock universe is empty.")
    st.stop()


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Ticker",
    "Name",
    "Market Cap",
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
    st.error(
        "The following required columns are missing from universe.csv: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# SLIDER HELPERS
# ============================================================

def add_cap_slider(location):
    key = f"cap_{location}"

    st.slider(
        "Minimum market cap ($ billions)",
        min_value=0.0,
        max_value=1000.0,
        step=10.0,
        key=key,
        on_change=sync_setting,
        args=("min_market_cap", key),
        help="Companies below this market capitalization are excluded.",
    )


def add_percent_slider(setting, location, label, help_text):
    key = f"{setting}_{location}"

    st.slider(
        label,
        min_value=0,
        max_value=50,
        step=1,
        key=key,
        on_change=sync_setting,
        args=(setting, key),
        help=help_text,
    )


def add_z_slider(setting, location, label, min_value, max_value, step):
    key = f"{setting}_{location}"

    st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=key,
        on_change=sync_setting,
        args=(setting, key),
    )


def add_signal_slider(location):
    key = f"signals_{location}"

    st.slider(
        "Minimum BUY signals for combined BUY",
        min_value=1,
        max_value=4,
        step=1,
        key=key,
        on_change=sync_setting,
        args=("min_signals", key),
        help=(
            "A company must meet this number of individual BUY criteria "
            "to qualify for the combined BUY list."
        ),
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

with st.sidebar:
    st.header("Screen Settings")
    st.caption(
        "Changing a setting updates the corresponding control on the "
        "main page as well."
    )

    add_cap_slider("sidebar")

    st.divider()
    st.subheader("Gross Profitability")
    add_percent_slider(
        "gpa_pct",
        "sidebar",
        "Minimum Gross Profitability (%)",
        "Gross Profit divided by total assets. Higher is better.",
    )

    st.divider()
    st.subheader("Accruals (STA)")
    add_percent_slider(
        "sta_pct",
        "sidebar",
        "STA signal threshold (%)",
        "Lower STA is treated as the BUY direction; higher STA as SELL.",
    )

    st.divider()
    st.subheader("Net Operating Assets (SNOA)")
    add_percent_slider(
        "snoa_pct",
        "sidebar",
        "SNOA signal threshold (%)",
        "Lower SNOA is treated as the BUY direction; higher SNOA as SELL.",
    )

    st.divider()
    st.subheader("Altman Z-Score")
    add_z_slider(
        "safe_z",
        "sidebar",
        "Safe-zone threshold",
        1.0,
        5.0,
        0.01,
    )
    add_z_slider(
        "distress_z",
        "sidebar",
        "Distress threshold",
        0.5,
        3.0,
        0.01,
    )

    st.divider()
    add_signal_slider("sidebar")

    if st.button("Reset all settings", use_container_width=True):
        for setting, default in DEFAULTS.items():
            st.session_state[setting] = default
            for widget_key in WIDGET_KEYS[setting]:
                st.session_state[widget_key] = default
        st.rerun()


# ============================================================
# MARKET CAP FILTER
# ============================================================

min_market_cap_dollars = (
    st.session_state["min_market_cap"] * 1e9
)

filtered_data = data.copy()

filtered_data = filtered_data[
    filtered_data["Market_Cap_Value"].notna()
    & (
        filtered_data["Market_Cap_Value"]
        >= min_market_cap_dollars
    )
].copy()


# ============================================================
# SCORING AND SIGNAL CLASSIFICATION
# ============================================================

def score_companies(universe):
    """Calculate individual signals and combined classifications."""
    scored = universe.copy()

    gpa_threshold = st.session_state["gpa_pct"] / 100
    sta_threshold = st.session_state["sta_pct"] / 100
    snoa_threshold = st.session_state["snoa_pct"] / 100
    safe_z = st.session_state["safe_z"]
    distress_z = st.session_state["distress_z"]

    # Gross Profitability: higher is better.
    scored["GPA_BUY"] = (
        scored["Gross_Profitability"].notna()
        & (scored["Gross_Profitability"] >= gpa_threshold)
    )
    scored["GPA_SELL"] = (
        scored["Gross_Profitability"].notna()
        & (scored["Gross_Profitability"] < 0)
    )

    # STA = (Net Income - Operating Cash Flow) / Total Assets.
    # Lower STA is treated as a positive signal.
    scored["STA_BUY"] = (
        scored["STA"].notna()
        & (scored["STA"] <= -sta_threshold)
    )
    scored["STA_SELL"] = (
        scored["STA"].notna()
        & (scored["STA"] >= sta_threshold)
    )

    # SNOA: lower is treated as better.
    scored["SNOA_BUY"] = (
        scored["SNOA"].notna()
        & (scored["SNOA"] <= -snoa_threshold)
    )
    scored["SNOA_SELL"] = (
        scored["SNOA"].notna()
        & (scored["SNOA"] >= snoa_threshold)
    )

    # Altman Z: higher is better.
    scored["Z_BUY"] = (
        scored["Altman_Z"].notna()
        & (scored["Altman_Z"] >= safe_z)
    )
    scored["Z_SELL"] = (
        scored["Altman_Z"].notna()
        & (scored["Altman_Z"] <= distress_z)
    )

    buy_columns = [
        "GPA_BUY",
        "STA_BUY",
        "SNOA_BUY",
        "Z_BUY",
    ]
    sell_columns = [
        "GPA_SELL",
        "STA_SELL",
        "SNOA_SELL",
        "Z_SELL",
    ]

    scored["BUY_Signals"] = scored[buy_columns].sum(axis=1)
    scored["SELL_Signals"] = scored[sell_columns].sum(axis=1)

    scored["Available_Metrics"] = scored[
        [
            "Gross_Profitability",
            "STA",
            "SNOA",
            "Altman_Z",
        ]
    ].notna().sum(axis=1)

    min_signals = st.session_state["min_signals"]

    # Avoid classifying a company as BUY if it also has a strong SELL
    # signal. A company with enough BUY signals and no SELL signals
    # qualifies as BUY.
    scored["Combined_Result"] = np.select(
        [
            (
                (scored["BUY_Signals"] >= min_signals)
                & (scored["SELL_Signals"] == 0)
            ),
            (
                (scored["SELL_Signals"] >= min_signals)
                & (scored["BUY_Signals"] == 0)
            ),
        ],
        ["BUY", "SELL"],
        default="NEUTRAL",
    )

    return scored


scored_data = score_companies(filtered_data)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_percent(value):
    if pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%"


def display_table(frame, columns):
    """Create a display-friendly copy of the requested columns."""
    result = frame[columns].copy()

    if "Market Cap" in result.columns:
        result["Market Cap"] = result["Market Cap"].apply(
            format_market_cap
        )

    percent_columns = [
        column for column in [
            "STA",
            "SNOA",
            "Gross_Profitability",
        ]
        if column in result.columns
    ]

    for column in percent_columns:
        result[column] = result[column].apply(format_percent)

    if "Altman_Z" in result.columns:
        result["Altman_Z"] = result["Altman_Z"].apply(
            lambda x: "—" if pd.isna(x) else f"{x:.2f}"
        )

    return result


def format_market_cap(value):
    """Format the original market-cap string for display."""
    if pd.isna(value) or str(value).strip() == "":
        return "—"
    return str(value)


def export_csv(frame):
    return frame.to_csv(index=False).encode("utf-8")


def show_metric_table(data_with_signals, metric, buy_col, sell_col):
    """
    Display only companies that qualify for the individual metric screen.

    BUY and SELL candidates are shown in separate groups. Companies that
    qualify for both directions are marked as BOTH rather than silently
    appearing in only one group.
    """
    required = [metric, buy_col, sell_col]

    missing = [
        column for column in required
        if column not in data_with_signals.columns
    ]

    if missing:
        st.error(
            "Required screen columns are missing: "
            + ", ".join(missing)
        )
        return

    valid = data_with_signals[
        data_with_signals[metric].notna()
    ].copy()

    buy_mask = valid[buy_col].fillna(False).astype(bool)
    sell_mask = valid[sell_col].fillna(False).astype(bool)

    valid["Screen Result"] = np.select(
        [buy_mask & sell_mask, buy_mask, sell_mask],
        ["BOTH signals", "BUY signal", "SELL signal"],
        default="No signal",
    )

    buy = valid[buy_mask & ~sell_mask].copy()
    sell = valid[sell_mask & ~buy_mask].copy()
    both = valid[buy_mask & sell_mask].copy()

    higher_is_better = metric in [
        "Gross_Profitability",
        "Altman_Z",
    ]

    buy = buy.sort_values(
        metric,
        ascending=not higher_is_better,
        na_position="last",
    )
    sell = sell.sort_values(
        metric,
        ascending=higher_is_better,
        na_position="last",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid observations", f"{len(valid):,}")
    c2.metric("BUY candidates", f"{len(buy):,}")
    c3.metric("SELL candidates", f"{len(sell):,}")
    c4.metric("Both signals", f"{len(both):,}")

    columns = [
        column for column in [
            "Rank",
            "Ticker",
            "Name",
            "Market Cap",
            metric,
            "Screen Result",
        ]
        if column in valid.columns
    ]

    tab_buy, tab_sell, tab_both = st.tabs(
        ["BUY candidates", "SELL candidates", "Both signals"]
    )

    with tab_buy:
        if buy.empty:
            st.info("No companies currently qualify for this BUY screen.")
        else:
            st.dataframe(
                display_table(buy, columns),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download BUY candidates",
                data=export_csv(buy),
                file_name=f"{metric.lower()}_buy_candidates.csv",
                mime="text/csv",
                key=f"{metric}_buy_download",
            )

    with tab_sell:
        if sell.empty:
            st.info("No companies currently qualify for this SELL screen.")
        else:
            st.dataframe(
                display_table(sell, columns),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download SELL candidates",
                data=export_csv(sell),
                file_name=f"{metric.lower()}_sell_candidates.csv",
                mime="text/csv",
                key=f"{metric}_sell_download",
            )

    with tab_both:
        if both.empty:
            st.info("No companies currently qualify for both directions.")
        else:
            st.dataframe(
                display_table(both, columns),
                use_container_width=True,
                hide_index=True,
            )


def show_coverage(frame):
    metrics = {
        "Altman Z-Score": "Altman_Z",
        "STA": "STA",
        "SNOA": "SNOA",
        "Gross Profitability": "Gross_Profitability",
    }

    coverage_rows = []

    for label, column in metrics.items():
        if column not in frame.columns:
            continue

        valid_count = int(frame[column].notna().sum())
        missing_count = int(frame[column].isna().sum())
        coverage_rows.append(
            {
                "Metric": label,
                "Valid observations": valid_count,
                "Missing observations": missing_count,
                "Coverage (%)": (
                    100 * valid_count / len(frame)
                    if len(frame) > 0
                    else 0
                ),
            }
        )

    if coverage_rows:
        coverage = pd.DataFrame(coverage_rows)
        coverage["Coverage (%)"] = coverage["Coverage (%)"].map(
            lambda value: f"{value:.1f}%"
        )
        st.dataframe(
            coverage,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MAIN-PAGE FILTERS
# ============================================================

st.header("Screen Controls")
st.caption(
    "These controls are synchronized with the sidebar. "
    "Changing either copy updates the screening results."
)

control1, control2 = st.columns(2)

with control1:
    add_cap_slider("main")
    add_percent_slider(
        "gpa_pct",
        "main",
        "Minimum Gross Profitability (%)",
        "Higher gross profitability is treated as better.",
    )
    add_percent_slider(
        "sta_pct",
        "main",
        "STA signal threshold (%)",
        "Low negative STA is treated as BUY; high positive STA as SELL.",
    )
    add_percent_slider(
        "snoa_pct",
        "main",
        "SNOA signal threshold (%)",
        "Low negative SNOA is treated as BUY; high positive SNOA as SELL.",
    )

with control2:
    add_z_slider(
        "safe_z",
        "main",
        "Altman Z safe-zone threshold",
        1.0,
        5.0,
        0.01,
    )
    add_z_slider(
        "distress_z",
        "main",
        "Altman Z distress threshold",
        0.5,
        3.0,
        0.01,
    )
    add_signal_slider("main")


# Recalculate after all controls are rendered, using the latest settings.
scored_data = score_companies(filtered_data)


# ============================================================
# STEP 1 — UNIVERSE OVERVIEW
# ============================================================

st.divider()
st.header("Step 1 — Stock Universe")

total_universe = len(data)
market_cap_universe = len(filtered_data)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total companies", f"{total_universe:,}")
m2.metric("After market-cap filter", f"{market_cap_universe:,}")
m3.metric(
    "Combined BUY",
    f"{int((scored_data['Combined_Result'] == 'BUY').sum()):,}",
)
m4.metric(
    "Combined SELL",
    f"{int((scored_data['Combined_Result'] == 'SELL').sum()):,}",
)

if scored_data.empty:
    st.warning(
        "No companies remain after the market-cap filter. "
        "Lower the minimum market cap to include more companies."
    )
    st.stop()

with st.expander("View stock universe", expanded=False):
    universe_columns = [
        column for column in [
            "Rank",
            "Ticker",
            "Name",
            "Market Cap",
            "Price",
            "Country",
        ]
        if column in scored_data.columns
    ]

    st.dataframe(
        display_table(scored_data, universe_columns),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STEP 2 — STA SCREEN
# ============================================================

st.divider()
st.header("Step 2 — STA (Accruals)")

st.markdown(
    """
    **Definition:** STA = (Net Income − Operating Cash Flow) / Total Assets.

    Lower STA is generally preferred in this screen. The selected threshold
    is applied in both directions: sufficiently negative values can qualify
    as BUY signals, while sufficiently positive values can qualify as SELL
    signals.
    """
)

show_metric_table(
    scored_data,
    "STA",
    "STA_BUY",
    "STA_SELL",
)


# ============================================================
# STEP 3 — SNOA SCREEN
# ============================================================

st.divider()
st.header("Step 3 — SNOA (Net Operating Assets)")

st.markdown(
    """
    **Definition:** SNOA = (Operating Assets − Operating Liabilities) /
    Total Assets.

    Lower SNOA is treated as the preferred direction for this screen.
    """
)

show_metric_table(
    scored_data,
    "SNOA",
    "SNOA_BUY",
    "SNOA_SELL",
)


# ============================================================
# STEP 4 — ALTMAN Z-SCORE
# ============================================================

st.divider()
st.header("Step 4 — Altman Z-Score")

st.markdown(
    """
    The manufacturing Altman Z-Score combines working capital, retained
    earnings, EBIT, market equity relative to liabilities, and sales relative
    to assets. Higher scores indicate a stronger financial position under
    this model.

    The traditional reference points are **2.99** for the safe zone and
    **1.81** for the distress zone. You can change these thresholds above.
    """
)

show_metric_table(
    scored_data,
    "Altman_Z",
    "Z_BUY",
    "Z_SELL",
)

z_data = scored_data["Altman_Z"].dropna()

if not z_data.empty:
    fig_z = px.histogram(
        x=z_data,
        nbins=40,
        title="Distribution of Altman Z-Scores",
        labels={"x": "Altman Z-Score", "count": "Companies"},
    )

    fig_z.add_vline(
        x=st.session_state["safe_z"],
        line_dash="dash",
        annotation_text="Safe threshold",
    )
    fig_z.add_vline(
        x=st.session_state["distress_z"],
        line_dash="dash",
        annotation_text="Distress threshold",
    )

    st.plotly_chart(fig_z, use_container_width=True)


# ============================================================
# STEP 5 — GROSS PROFITABILITY
# ============================================================

st.divider()
st.header("Step 5 — Gross Profitability")

st.markdown(
    """
    **Definition:** Gross Profitability = Gross Profit / Total Assets.

    Higher gross profitability is treated as better. The BUY threshold is
    controlled by the Gross Profitability slider. Negative values are marked
    as SELL signals.
    """
)

show_metric_table(
    scored_data,
    "Gross_Profitability",
    "GPA_BUY",
    "GPA_SELL",
)

gpa_data = scored_data["Gross_Profitability"].dropna()

if not gpa_data.empty:
    fig_gpa = px.histogram(
        x=gpa_data * 100,
        nbins=40,
        title="Distribution of Gross Profitability",
        labels={"x": "Gross Profitability (%)", "count": "Companies"},
    )

    fig_gpa.add_vline(
        x=st.session_state["gpa_pct"],
        line_dash="dash",
        annotation_text="BUY threshold",
    )

    st.plotly_chart(fig_gpa, use_container_width=True)


# ============================================================
# STEP 6 — COMBINED CLASSIFICATION
# ============================================================

st.divider()
st.header("Step 6 — Combined Screening Results")

st.markdown(
    """
    Each company can receive up to four individual BUY signals and four
    individual SELL signals: Gross Profitability, STA, SNOA, and Altman Z.

    A combined **BUY** requires at least the selected minimum number of
    BUY signals and no SELL signals. A combined **SELL** requires at least
    the selected minimum number of SELL signals and no BUY signals.
    Other companies are classified as **NEUTRAL**.
    """
)

result_counts = (
    scored_data["Combined_Result"]
    .value_counts()
    .reindex(["BUY", "SELL", "NEUTRAL"], fill_value=0)
)

r1, r2, r3 = st.columns(3)
r1.metric("BUY", f"{int(result_counts['BUY']):,}")
r2.metric("SELL", f"{int(result_counts['SELL']):,}")
r3.metric("NEUTRAL", f"{int(result_counts['NEUTRAL']):,}")

fig_results = px.bar(
    x=result_counts.index,
    y=result_counts.values,
    title="Companies by Combined Classification",
    labels={"x": "Classification", "y": "Number of companies"},
)

st.plotly_chart(fig_results, use_container_width=True)

combined_filter = st.selectbox(
    "Choose which combined results to view",
    ["All companies", "BUY", "SELL", "NEUTRAL"],
)

if combined_filter == "All companies":
    combined_view = scored_data.copy()
else:
    combined_view = scored_data[
        scored_data["Combined_Result"] == combined_filter
    ].copy()

combined_view = combined_view.sort_values(
    ["BUY_Signals", "SELL_Signals"],
    ascending=[False, True],
)

combined_columns = [
    column for column in [
        "Rank",
        "Ticker",
        "Name",
        "Market Cap",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
        "BUY_Signals",
        "SELL_Signals",
        "Combined_Result",
    ]
    if column in combined_view.columns
]

st.dataframe(
    display_table(combined_view, combined_columns),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download combined screening results",
    data=export_csv(combined_view),
    file_name="combined_screening_results.csv",
    mime="text/csv",
)


# ============================================================
# STEP 7 — TOP 10 BUY AND SELL
# ============================================================

st.divider()
st.header("Step 7 — Top 10 BUY and SELL")

buy_candidates = scored_data[
    scored_data["Combined_Result"] == "BUY"
].copy()

sell_candidates = scored_data[
    scored_data["Combined_Result"] == "SELL"
].copy()

buy_candidates = buy_candidates.sort_values(
    ["BUY_Signals", "Gross_Profitability", "Altman_Z"],
    ascending=[False, False, False],
    na_position="last",
)

sell_candidates = sell_candidates.sort_values(
    ["SELL_Signals", "Gross_Profitability", "Altman_Z"],
    ascending=[False, True, True],
    na_position="last",
)

top_buy = buy_candidates.head(10)
top_sell = sell_candidates.head(10)

buy_tab, sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

top_columns = [
    column for column in [
        "Rank",
        "Ticker",
        "Name",
        "Market Cap",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
        "BUY_Signals",
        "SELL_Signals",
        "Combined_Result",
    ]
    if column in scored_data.columns
]

with buy_tab:
    if top_buy.empty:
        st.info("No companies currently meet the combined BUY criteria.")
    else:
        st.dataframe(
            display_table(top_buy, top_columns),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Top 10 BUY",
            data=export_csv(top_buy),
            file_name="top_10_buy.csv",
            mime="text/csv",
        )

with sell_tab:
    if top_sell.empty:
        st.info("No companies currently meet the combined SELL criteria.")
    else:
        st.dataframe(
            display_table(top_sell, top_columns),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Top 10 SELL",
            data=export_csv(top_sell),
            file_name="top_10_sell.csv",
            mime="text/csv",
        )


# ============================================================
# COMPANY EXPLORER
# ============================================================

st.divider()
st.header("Company Explorer")

company_options = scored_data.copy()
company_options["_label"] = (
    company_options["Ticker"].astype(str)
    + " — "
    + company_options["Name"].astype(str)
)

company_options = company_options.sort_values("_label")

if not company_options.empty:
    selected_label = st.selectbox(
        "Select a company",
        company_options["_label"].tolist(),
    )

    selected_company = company_options[
        company_options["_label"] == selected_label
    ].iloc[0]

    st.subheader(
        f"{selected_company['Ticker']} — {selected_company['Name']}"
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Market Cap",
        format_market_cap(selected_company.get("Market Cap", np.nan)),
    )
    e2.metric(
        "Gross Profitability",
        format_percent(selected_company.get("Gross_Profitability", np.nan)),
    )
    e3.metric(
        "STA",
        format_percent(selected_company.get("STA", np.nan)),
    )
    e4.metric(
        "SNOA",
        format_percent(selected_company.get("SNOA", np.nan)),
    )

    e5, e6, e7 = st.columns(3)

    z_value = selected_company.get("Altman_Z", np.nan)

    e5.metric(
        "Altman Z-Score",
        "—" if pd.isna(z_value) else f"{z_value:.2f}",
    )
    e6.metric(
        "BUY signals",
        int(selected_company["BUY_Signals"]),
    )
    e7.metric(
        "SELL signals",
        int(selected_company["SELL_Signals"]),
    )

    st.write(
        "**Combined classification:**",
        selected_company["Combined_Result"],
    )

    details = {
        "Metric": [
            "Gross Profitability",
            "STA",
            "SNOA",
            "Altman Z-Score",
        ],
        "Value": [
            selected_company.get("Gross_Profitability", np.nan),
            selected_company.get("STA", np.nan),
            selected_company.get("SNOA", np.nan),
            selected_company.get("Altman_Z", np.nan),
        ],
        "Individual BUY signal": [
            bool(selected_company["GPA_BUY"]),
            bool(selected_company["STA_BUY"]),
            bool(selected_company["SNOA_BUY"]),
            bool(selected_company["Z_BUY"]),
        ],
        "Individual SELL signal": [
            bool(selected_company["GPA_SELL"]),
            bool(selected_company["STA_SELL"]),
            bool(selected_company["SNOA_SELL"]),
            bool(selected_company["Z_SELL"]),
        ],
    }

    detail_frame = pd.DataFrame(details)

    detail_frame["Value"] = detail_frame.apply(
        lambda row: (
            "—"
            if pd.isna(row["Value"])
            else (
                f"{row['Value']:.2f}"
                if row["Metric"] == "Altman Z-Score"
                else format_percent(row["Value"])
            )
        ),
        axis=1,
    )

    st.dataframe(
        detail_frame,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# METHODOLOGY AND DATA COVERAGE
# ============================================================

st.divider()
st.header("Methodology and Data Coverage")

with st.expander("Metric definitions", expanded=False):
    st.markdown(
        """
        **Altman Z-Score**

        The manufacturing version of the Altman Z-Score is calculated as:

        - 1.2 × working capital / total assets
        - 1.4 × retained earnings / total assets
        - 3.3 × EBIT / total assets
        - 0.6 × market equity / total liabilities
        - 1.0 × sales / total assets

        The supplied Z-Score is used when available. This app does not
        independently reconstruct the score if the required inputs are absent.

        **STA**

        STA = (Net Income − Operating Cash Flow) / Total Assets.
        Lower values are treated as better by this screen.

        **SNOA**

        SNOA = (Operating Assets − Operating Liabilities) / Total Assets.
        Lower values are treated as better by this screen.

        **Gross Profitability**

        Gross Profitability = Gross Profit / Total Assets.
        Higher values are treated as better by this screen.

        **Combined screen**

        The combined classification uses the four individual metric signals.
        A company with both BUY and SELL signals does not qualify for either
        combined category under the current rule.
        """
    )

with st.expander("Data coverage and missing values", expanded=False):
    st.write(
        "Coverage is calculated using the current market-cap-filtered "
        "universe. Missing metric values are excluded from that metric's "
        "individual screen."
    )
    show_coverage(filtered_data)

with st.expander("Screening limitations", expanded=False):
    st.markdown(
        """
        - These screens are quantitative filters, not personalized investment
          advice or a guarantee of future performance.
        - The thresholds are configurable and can materially change the
          number of companies classified as BUY or SELL.
        - Missing or outdated financial data can affect the results.
        - The Altman Z-Score was designed for a particular financial-distress
          context and may not be equally appropriate for every industry.
        - The STA and SNOA signal directions used here are screening choices;
          they should be evaluated alongside other financial information.
        - Market-cap and financial data may come from different dates.
        - A BUY or SELL label is a screen result, not a recommendation to
          purchase or sell a security.
        """
    )

st.caption(
    "Screening outputs depend on the data in universe.csv and the "
    "thresholds selected above."
)
