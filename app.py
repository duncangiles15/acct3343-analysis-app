
import io
import re

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
    initial_sidebar_state="expanded",
)

st.title("Manufacturing Stock Screening Dashboard")

st.markdown(
    """
    This dashboard screens a universe of publicly traded manufacturing
    companies using four financial metrics: Altman Z-Score, STA, SNOA,
    and Gross Profitability.

    The three percentile-based metrics rank companies relative to one
    another. Altman Z-Score retains its established fixed thresholds.
    Results are rule-based screening outputs, not investment recommendations.
    """
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULTS = {
    "min_market_cap": 0.0,
    "sta_buy_exclude": 10,
    "sta_sell_exclude": 10,
    "snoa_buy_exclude": 10,
    "snoa_sell_exclude": 10,
    "gpa_buy_exclude": 10,
    "gpa_sell_exclude": 10,
    "safe_z": 2.99,
    "distress_z": 1.81,
    "min_signals": 2,
}


def initialize_settings():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_settings()


# ============================================================
# FORMATTING AND DATA HELPERS
# ============================================================

def parse_market_cap(value):
    """Convert market-cap strings such as $2.4T or $850B to dollars."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace(",", "").replace("$", "")

    if not text or text.lower() in {"nan", "none", "n/a", "-"}:
        return np.nan

    match = re.match(
        r"^(-?\d+(?:\.\d+)?)\s*([KMBT]?)$",
        text,
        re.IGNORECASE,
    )

    if not match:
        return np.nan

    number = float(match.group(1))
    suffix = match.group(2).upper()

    multipliers = {
        "": 1,
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
        "T": 1e12,
    }

    return number * multipliers[suffix]


def format_market_cap(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1e12:
        return f"${value / 1e12:,.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"

    return f"${value:,.0f}"


def format_percent(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def display_table(frame, columns=None):
    """Format financial ratios for display without altering the source data."""
    result = frame.copy()

    if columns is not None:
        existing = [col for col in columns if col in result.columns]
        result = result[existing]

    percent_columns = [
        "STA",
        "SNOA",
        "Gross_Profitability",
        "X1_Working_Capital_Pct",
        "X2_Retained_Earnings_Pct",
        "X3_EBIT_Pct",
        "X4_Market_Equity_to_Liabilities_Pct",
    ]

    for col in percent_columns:
        if col in result.columns:
            result[col] = result[col].map(format_percent)

    if "Market_Cap_Value" in result.columns:
        result["Market Cap"] = result["Market_Cap_Value"].map(
            format_market_cap
        )
        result = result.drop(columns=["Market_Cap_Value"])

    if "Altman_Z" in result.columns:
        result["Altman_Z"] = result["Altman_Z"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

    if "Price" in result.columns:
        result["Price"] = result["Price"].map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
        )

    if "Today" in result.columns:
        result["Today"] = result["Today"].map(
            lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )

    if "BUY_Signals" in result.columns:
        result["BUY_Signals"] = result["BUY_Signals"].astype(str)

    if "SELL_Signals" in result.columns:
        result["SELL_Signals"] = result["SELL_Signals"].astype(str)

    return result


def export_csv(frame):
    return frame.to_csv(index=False).encode("utf-8")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(file_path, modified_time):
    data = pd.read_csv(file_path)

    numeric_columns = [
        "Rank",
        "Price",
        "Today",
        "Price (30 days)",
        "STA",
        "SNOA",
        "Altman_Z",
        "Gross_Profit",
        "Total_Assets",
        "Gross_Profitability",
        "X1_Working_Capital_Pct",
        "X2_Retained_Earnings_Pct",
        "X3_EBIT_Pct",
        "X4_Market_Equity_to_Liabilities_Pct",
        "X5_Sales_to_Assets",
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Market Cap" in data.columns:
        data["Market_Cap_Value"] = data["Market Cap"].map(
            parse_market_cap
        )
    else:
        data["Market_Cap_Value"] = np.nan

    string_columns = [
        "Name",
        "Country",
        "Ticker",
        "Ticker_guess",
        "Altman_Status",
        "STA_Status",
        "SNOA_Status",
        "Fiscal_Date",
        "Altman_Fiscal_Date",
        "GP_Fiscal_Date",
    ]

    for col in string_columns:
        if col not in data.columns:
            data[col] = ""

        data[col] = data[col].fillna("").astype(str)

    if "Ticker" in data.columns and "Ticker_guess" in data.columns:
        data["Ticker"] = data["Ticker"].where(
            data["Ticker"].str.strip() != "",
            data["Ticker_guess"],
        )

    if "Name" not in data.columns:
        data["Name"] = ""

    if "Ticker" not in data.columns:
        data["Ticker"] = ""

    data["Display_Name"] = data["Name"].where(
        data["Name"].str.strip() != "",
        data["Ticker"],
    )

    return data


try:
    import os

    data_path = "universe.csv"

    if not os.path.exists(data_path):
        st.error(
            "Could not find universe.csv. Make sure it is in the same "
            "GitHub repository directory as app.py."
        )
        st.stop()

    data = load_data(
        data_path,
        os.path.getmtime(data_path),
    )

except Exception as error:
    st.error(f"Unable to load the stock universe: {error}")
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Screening Settings")
st.sidebar.caption(
    "Adjust the filters below. The tables and results update automatically."
)

st.sidebar.subheader("Step 1 — Stock Universe")

market_cap_billions = st.sidebar.number_input(
    "Minimum market cap ($ billions)",
    min_value=0.0,
    max_value=1000.0,
    value=float(st.session_state["min_market_cap"]),
    step=1.0,
    key="cap_sidebar",
    help="Companies below this market capitalization are excluded.",
)

st.session_state["min_market_cap"] = market_cap_billions

st.sidebar.divider()
st.sidebar.subheader("Step 2 — STA")

st.sidebar.caption(
    "STA = (Net Income − Operating Cash Flow) / Total Assets. "
    "Lower values are generally more favourable."
)

sta_buy_exclude = st.sidebar.slider(
    "STA: Exclude worst % for BUY",
    min_value=0,
    max_value=50,
    value=int(st.session_state["sta_buy_exclude"]),
    step=1,
    key="sta_buy_sidebar",
    help=(
        "Ranks stocks from best to worst for a BUY signal. "
        "Excludes the specified worst percentage from BUY eligibility."
    ),
)

sta_sell_exclude = st.sidebar.slider(
    "STA: Exclude worst % for SELL",
    min_value=0,
    max_value=50,
    value=int(st.session_state["sta_sell_exclude"]),
    step=1,
    key="sta_sell_sidebar",
    help=(
        "Ranks stocks from best to worst for a SELL signal. "
        "Excludes the specified worst percentage from SELL eligibility."
    ),
)

st.session_state["sta_buy_exclude"] = sta_buy_exclude
st.session_state["sta_sell_exclude"] = sta_sell_exclude

st.sidebar.divider()
st.sidebar.subheader("Step 3 — SNOA")

st.sidebar.caption(
    "SNOA = (Operating Assets − Operating Liabilities) / Total Assets. "
    "Lower values are generally more favourable."
)

snoa_buy_exclude = st.sidebar.slider(
    "SNOA: Exclude worst % for BUY",
    min_value=0,
    max_value=50,
    value=int(st.session_state["snoa_buy_exclude"]),
    step=1,
    key="snoa_buy_sidebar",
)

snoa_sell_exclude = st.sidebar.slider(
    "SNOA: Exclude worst % for SELL",
    min_value=0,
    max_value=50,
    value=int(st.session_state["snoa_sell_exclude"]),
    step=1,
    key="snoa_sell_sidebar",
)

st.session_state["snoa_buy_exclude"] = snoa_buy_exclude
st.session_state["snoa_sell_exclude"] = snoa_sell_exclude

st.sidebar.divider()
st.sidebar.subheader("Step 4 — Altman Z-Score")

st.sidebar.caption(
    "Altman Z uses fixed thresholds rather than percentile rankings."
)

safe_z = st.sidebar.number_input(
    "Safe-zone threshold",
    min_value=0.0,
    max_value=10.0,
    value=float(st.session_state["safe_z"]),
    step=0.01,
    key="safe_z_sidebar",
)

distress_z = st.sidebar.number_input(
    "Distress threshold",
    min_value=-10.0,
    max_value=10.0,
    value=float(st.session_state["distress_z"]),
    step=0.01,
    key="distress_z_sidebar",
)

st.session_state["safe_z"] = safe_z
st.session_state["distress_z"] = distress_z

if distress_z >= safe_z:
    st.sidebar.warning(
        "The distress threshold should be lower than the safe-zone "
        "threshold. Adjust the values to avoid overlapping classifications."
    )

st.sidebar.divider()
st.sidebar.subheader("Step 5 — Gross Profitability")

st.sidebar.caption(
    "Gross Profitability = Gross Profit / Total Assets. "
    "Higher values are generally more favourable."
)

gpa_buy_exclude = st.sidebar.slider(
    "Gross Profitability: Exclude worst % for BUY",
    min_value=0,
    max_value=50,
    value=int(st.session_state["gpa_buy_exclude"]),
    step=1,
    key="gpa_buy_sidebar",
)

gpa_sell_exclude = st.sidebar.slider(
    "Gross Profitability: Exclude worst % for SELL",
    min_value=0,
    max_value=50,
    value=int(st.session_state["gpa_sell_exclude"]),
    step=1,
    key="gpa_sell_sidebar",
)

st.session_state["gpa_buy_exclude"] = gpa_buy_exclude
st.session_state["gpa_sell_exclude"] = gpa_sell_exclude

st.sidebar.divider()
st.sidebar.subheader("Step 6 — Combined Classification")

min_signals = st.sidebar.slider(
    "Minimum matching signals",
    min_value=1,
    max_value=4,
    value=int(st.session_state["min_signals"]),
    step=1,
    key="signals_sidebar",
    help=(
        "A company must meet at least this many BUY or SELL metric signals "
        "and have no opposing signals to receive that combined classification."
    ),
)

st.session_state["min_signals"] = min_signals

if st.sidebar.button("Reset All Filters", use_container_width=True):
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    st.rerun()


# ============================================================
# APPLY MARKET-CAP FILTER
# ============================================================

minimum_market_cap = market_cap_billions * 1e9

filtered = data.copy()

if minimum_market_cap > 0:
    filtered = filtered[
        filtered["Market_Cap_Value"].notna()
        & (filtered["Market_Cap_Value"] >= minimum_market_cap)
    ].copy()


# ============================================================
# PERCENTILE RANKING
# ============================================================

def signal_from_percentile(series, exclusion_pct, higher_is_better):
    """
    Returns a Boolean signal for stocks that remain eligible after
    excluding the worst percentage of non-missing observations.

    The score is a percentile rank where higher means better for the
    particular signal. Exclusion of 10% retains roughly the best 90%;
    exclusion of 50% retains roughly the best half.

    Missing values remain missing and cannot receive a signal.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()

    result = pd.Series(False, index=series.index, dtype=bool)

    if not valid.any():
        return result

    ranked_values = numeric.loc[valid]

    if higher_is_better:
        quality_percentile = ranked_values.rank(
            pct=True,
            ascending=True,
            method="average",
        )
    else:
        quality_percentile = ranked_values.rank(
            pct=True,
            ascending=False,
            method="average",
        )

    cutoff = exclusion_pct / 100.0

    result.loc[valid] = quality_percentile >= cutoff

    return result


# ============================================================
# SCORE COMPANIES
# ============================================================

def score_companies(frame):
    result = frame.copy()

    required_metrics = [
        "STA",
        "SNOA",
        "Gross_Profitability",
        "Altman_Z",
    ]

    for metric in required_metrics:
        if metric not in result.columns:
            result[metric] = np.nan

        result[metric] = pd.to_numeric(
            result[metric],
            errors="coerce",
        )

    # STA:
    # Lower STA is better for BUY.
    # Higher STA is better for SELL.
    result["STA_BUY"] = signal_from_percentile(
        result["STA"],
        st.session_state["sta_buy_exclude"],
        higher_is_better=False,
    ) & result["STA"].notna()

    result["STA_SELL"] = signal_from_percentile(
        result["STA"],
        st.session_state["sta_sell_exclude"],
        higher_is_better=True,
    ) & result["STA"].notna()

    # SNOA:
    # Lower SNOA is better for BUY.
    # Higher SNOA is better for SELL.
    result["SNOA_BUY"] = signal_from_percentile(
        result["SNOA"],
        st.session_state["snoa_buy_exclude"],
        higher_is_better=False,
    ) & result["SNOA"].notna()

    result["SNOA_SELL"] = signal_from_percentile(
        result["SNOA"],
        st.session_state["snoa_sell_exclude"],
        higher_is_better=True,
    ) & result["SNOA"].notna()

    # Gross Profitability:
    # Higher is better for BUY.
    # Lower is better for SELL.
    result["GPA_BUY"] = signal_from_percentile(
        result["Gross_Profitability"],
        st.session_state["gpa_buy_exclude"],
        higher_is_better=True,
    ) & result["Gross_Profitability"].notna()

    result["GPA_SELL"] = signal_from_percentile(
        result["Gross_Profitability"],
        st.session_state["gpa_sell_exclude"],
        higher_is_better=False,
    ) & result["Gross_Profitability"].notna()

    # Altman Z retains its existing fixed thresholds.
    result["Z_BUY"] = (
        result["Altman_Z"] >= st.session_state["safe_z"]
    ) & result["Altman_Z"].notna()

    result["Z_SELL"] = (
        result["Altman_Z"] <= st.session_state["distress_z"]
    ) & result["Altman_Z"].notna()

    buy_columns = [
        "STA_BUY",
        "SNOA_BUY",
        "GPA_BUY",
        "Z_BUY",
    ]

    sell_columns = [
        "STA_SELL",
        "SNOA_SELL",
        "GPA_SELL",
        "Z_SELL",
    ]

    result["BUY_Signals"] = result[buy_columns].sum(axis=1)
    result["SELL_Signals"] = result[sell_columns].sum(axis=1)

    # Count only metrics with actual available data.
    result["Available_Metrics"] = result[required_metrics].notna().sum(axis=1)

    # Combined classification requires enough signals and no opposing
    # signals. Missing metrics do not count as signals.
    result["Combined_Signal"] = "NEUTRAL"

    buy_mask = (
        (result["BUY_Signals"] >= st.session_state["min_signals"])
        & (result["SELL_Signals"] == 0)
    )

    sell_mask = (
        (result["SELL_Signals"] >= st.session_state["min_signals"])
        & (result["BUY_Signals"] == 0)
    )

    result.loc[buy_mask, "Combined_Signal"] = "BUY"
    result.loc[sell_mask, "Combined_Signal"] = "SELL"

    return result


scored = score_companies(filtered)


# ============================================================
# TABLE HELPERS
# ============================================================

base_columns = [
    "Rank",
    "Display_Name",
    "Ticker",
    "Country",
    "Market_Cap_Value",
    "Price",
    "STA",
    "SNOA",
    "Altman_Z",
    "Gross_Profitability",
    "BUY_Signals",
    "SELL_Signals",
    "Combined_Signal",
]


def table_columns(frame, requested):
    return [col for col in requested if col in frame.columns]


def metric_candidates(frame, metric, side):
    """Return companies eligible for a particular metric signal."""
    flag = f"{metric}_{side}"
    candidates = frame.loc[frame[flag]].copy()

    # Sort the most attractive candidates for that particular side first.
    if metric == "STA":
        ascending = side == "BUY"
    elif metric == "SNOA":
        ascending = side == "BUY"
    elif metric == "GPA":
        ascending = side == "SELL"
    elif metric == "Z":
        ascending = side == "SELL"
    else:
        ascending = True

    metric_column = {
        "STA": "STA",
        "SNOA": "SNOA",
        "GPA": "Gross_Profitability",
        "Z": "Altman_Z",
    }[metric]

    candidates = candidates.sort_values(
        metric_column,
        ascending=ascending,
        na_position="last",
    )

    return candidates


def show_metric_table(frame, metric, label):
    buy_candidates = metric_candidates(frame, metric, "BUY")
    sell_candidates = metric_candidates(frame, metric, "SELL")

    tab_buy, tab_sell, tab_both = st.tabs(
        ["BUY Candidates", "SELL Candidates", "Both Signals"]
    )

    with tab_buy:
        st.caption(
            f"{len(buy_candidates):,} companies remain eligible for the "
            f"{label} BUY signal after percentile exclusion."
        )

        if buy_candidates.empty:
            st.info("No companies qualify under the current settings.")
        else:
            cols = table_columns(buy_candidates, base_columns)
            st.dataframe(
                display_table(buy_candidates, cols),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                f"Download {label} BUY Candidates",
                export_csv(buy_candidates),
                file_name=f"{metric.lower()}_buy_candidates.csv",
                mime="text/csv",
                key=f"{metric}_buy_download",
            )

    with tab_sell:
        st.caption(
            f"{len(sell_candidates):,} companies remain eligible for the "
            f"{label} SELL signal after percentile exclusion."
        )

        if sell_candidates.empty:
            st.info("No companies qualify under the current settings.")
        else:
            cols = table_columns(sell_candidates, base_columns)
            st.dataframe(
                display_table(sell_candidates, cols),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                f"Download {label} SELL Candidates",
                export_csv(sell_candidates),
                file_name=f"{metric.lower()}_sell_candidates.csv",
                mime="text/csv",
                key=f"{metric}_sell_download",
            )

    with tab_both:
        both = frame.loc[
            frame[f"{metric}_BUY"] & frame[f"{metric}_SELL"]
        ].copy()

        st.caption(
            "These companies meet both side-specific eligibility rules "
            "under the current exclusion settings."
        )

        if both.empty:
            st.info("No companies qualify for both signals.")
        else:
            cols = table_columns(both, base_columns)
            st.dataframe(
                display_table(both, cols),
                use_container_width=True,
                hide_index=True,
            )


def show_coverage(frame, metric_columns):
    rows = []

    for column, label in metric_columns:
        if column not in frame.columns:
            available = 0
        else:
            available = int(frame[column].notna().sum())

        total = len(frame)
        missing = total - available
        coverage_pct = available / total if total else 0.0

        rows.append(
            {
                "Metric": label,
                "Available": available,
                "Missing": missing,
                "Coverage": coverage_pct,
            }
        )

    coverage = pd.DataFrame(rows)

    if not coverage.empty:
        coverage["Coverage"] = coverage["Coverage"].map(
            lambda value: f"{value:.1%}"
        )

    return coverage


# ============================================================
# SECTION 1 — STOCK UNIVERSE
# ============================================================

st.header("1. Manufacturing Stock Universe")

total_universe = len(data)
filtered_universe = len(scored)

col1, col2, col3 = st.columns(3)

col1.metric("Stocks in Source Universe", f"{total_universe:,}")
col2.metric("Stocks After Market-Cap Filter", f"{filtered_universe:,}")

if "Market_Cap_Value" in scored.columns:
    median_cap = scored["Market_Cap_Value"].median()
else:
    median_cap = np.nan

col3.metric("Median Market Cap", format_market_cap(median_cap))

with st.expander("View Stock Universe"):
    universe_columns = table_columns(
        scored,
        [
            "Rank",
            "Display_Name",
            "Ticker",
            "Country",
            "Market_Cap_Value",
            "Price",
            "Today",
        ],
    )

    st.dataframe(
        display_table(scored, universe_columns),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Filtered Stock Universe",
        export_csv(scored),
        file_name="filtered_stock_universe.csv",
        mime="text/csv",
        key="universe_download",
    )


# ============================================================
# SECTION 2 — STA
# ============================================================

st.header("2. Earnings Quality - (STA)")

st.markdown(
    """
    **Formula:** (Net Income − Operating Cash Flow) / Total Assets

    Lower STA values are generally more favourable because operating cash
    flow is stronger relative to reported net income. Stocks are ranked
    against one another. The BUY and SELL exclusion sliders independently
    remove the worst-ranked portion for their respective signal.
    """
)

st.caption(
    f"BUY exclusion: {sta_buy_exclude}% · "
    f"SELL exclusion: {sta_sell_exclude}%"
)

show_metric_table(scored, "STA", "STA")


# ============================================================
# SECTION 3 — SNOA
# ============================================================

st.header("3. Net Operating Assets (SNOA)")

st.markdown(
    """
    **Formula:** (Operating Assets − Operating Liabilities) / Total Assets

    Lower SNOA values are generally more favourable. Companies are ranked
    relative to the filtered stock universe, with separate BUY and SELL
    exclusion settings.
    """
)

st.caption(
    f"BUY exclusion: {snoa_buy_exclude}% · "
    f"SELL exclusion: {snoa_sell_exclude}%"
)

show_metric_table(scored, "SNOA", "SNOA")


# ============================================================
# SECTION 4 — ALTMAN Z
# ============================================================

st.header("4. Altman Z-Score")

st.markdown(
    """
    Altman Z-Score uses fixed thresholds, not percentile exclusions.

    - **Safe zone:** Z-Score at or above the safe-zone threshold.
    - **Distress zone:** Z-Score at or below the distress threshold.
    - **Between thresholds:** Neither an Altman Z BUY nor SELL signal.
    """
)

z_buy = scored.loc[scored["Z_BUY"]].copy()
z_sell = scored.loc[scored["Z_SELL"]].copy()

z_tab_buy, z_tab_sell = st.tabs(
    ["Safe-Zone Candidates", "Distress-Zone Candidates"]
)

with z_tab_buy:
    st.caption(
        f"{len(z_buy):,} companies have an Altman Z-Score "
        f"at or above {safe_z:.2f}."
    )

    z_buy = z_buy.sort_values("Altman_Z", ascending=False)

    if z_buy.empty:
        st.info("No companies are in the safe zone under the current settings.")
    else:
        st.dataframe(
            display_table(
                z_buy,
                table_columns(z_buy, base_columns),
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Safe-Zone Candidates",
            export_csv(z_buy),
            file_name="altman_z_safe_candidates.csv",
            mime="text/csv",
            key="z_buy_download",
        )

with z_tab_sell:
    st.caption(
        f"{len(z_sell):,} companies have an Altman Z-Score "
        f"at or below {distress_z:.2f}."
    )

    z_sell = z_sell.sort_values("Altman_Z", ascending=True)

    if z_sell.empty:
        st.info(
            "No companies are in the distress zone under the current settings."
        )
    else:
        st.dataframe(
            display_table(
                z_sell,
                table_columns(z_sell, base_columns),
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Distress-Zone Candidates",
            export_csv(z_sell),
            file_name="altman_z_distress_candidates.csv",
            mime="text/csv",
            key="z_sell_download",
        )

z_values = scored["Altman_Z"].dropna()

if not z_values.empty:
    z_fig = px.histogram(
        pd.DataFrame({"Altman Z-Score": z_values}),
        x="Altman Z-Score",
        nbins=40,
        title="Altman Z-Score Distribution",
    )

    z_fig.add_vline(
        x=safe_z,
        line_dash="dash",
        annotation_text=f"Safe: {safe_z:.2f}",
    )

    z_fig.add_vline(
        x=distress_z,
        line_dash="dash",
        annotation_text=f"Distress: {distress_z:.2f}",
    )

    st.plotly_chart(z_fig, use_container_width=True)


# ============================================================
# SECTION 5 — GROSS PROFITABILITY
# ============================================================

st.header("5. Gross Profitability")

st.markdown(
    """
    **Formula:** Gross Profit / Total Assets

    Higher Gross Profitability generally indicates that a company generates
    more gross profit relative to its asset base. The BUY and SELL lists
    use opposite rankings, with separate percentile-exclusion sliders.
    """
)

st.caption(
    f"BUY exclusion: {gpa_buy_exclude}% · "
    f"SELL exclusion: {gpa_sell_exclude}%"
)

show_metric_table(scored, "GPA", "Gross Profitability")

gpa_values = scored["Gross_Profitability"].dropna()

if not gpa_values.empty:
    gpa_fig = px.histogram(
        pd.DataFrame(
            {"Gross Profitability": gpa_values * 100}
        ),
        x="Gross Profitability",
        nbins=40,
        title="Gross Profitability Distribution",
        labels={"Gross Profitability": "Gross Profitability (%)"},
    )

    st.plotly_chart(gpa_fig, use_container_width=True)


# ============================================================
# SECTION 6 — COMBINED SCREENING
# ============================================================

st.header("6. Combined Screening Results")

st.markdown(
    f"""
    Combined classifications require at least **{min_signals} matching
    metric signals**, with no opposing signals. Missing metric values do
    not count as signals.
    """
)

buy_results = scored.loc[scored["Combined_Signal"] == "BUY"].copy()
sell_results = scored.loc[scored["Combined_Signal"] == "SELL"].copy()
neutral_results = scored.loc[scored["Combined_Signal"] == "NEUTRAL"].copy()

col1, col2, col3 = st.columns(3)

col1.metric("BUY", f"{len(buy_results):,}")
col2.metric("SELL", f"{len(sell_results):,}")
col3.metric("NEUTRAL", f"{len(neutral_results):,}")

signal_counts = pd.DataFrame(
    {
        "Classification": ["BUY", "SELL", "NEUTRAL"],
        "Companies": [
            len(buy_results),
            len(sell_results),
            len(neutral_results),
        ],
    }
)

signal_fig = px.bar(
    signal_counts,
    x="Classification",
    y="Companies",
    title="Companies by Combined Classification",
    text="Companies",
)

st.plotly_chart(signal_fig, use_container_width=True)

selected_classification = st.selectbox(
    "Choose classification to view",
    ["All", "BUY", "SELL", "NEUTRAL"],
)

if selected_classification == "All":
    combined_view = scored.copy()
else:
    combined_view = scored.loc[
        scored["Combined_Signal"] == selected_classification
    ].copy()

combined_view = combined_view.sort_values(
    ["BUY_Signals", "SELL_Signals"],
    ascending=[False, True],
)

st.dataframe(
    display_table(
        combined_view,
        table_columns(combined_view, base_columns),
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Combined Screening Results",
    export_csv(combined_view),
    file_name="combined_screening_results.csv",
    mime="text/csv",
    key="combined_download",
)


# ============================================================
# SECTION 7 — TOP 10 BUY AND SELL
# ============================================================

st.header("7. Top 10 BUY and SELL Stocks")

top_buy_tab, top_sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

ranking_columns = table_columns(
    scored,
    [
        "Display_Name",
        "Ticker",
        "Country",
        "Market_Cap_Value",
        "Price",
        "STA",
        "SNOA",
        "Altman_Z",
        "Gross_Profitability",
        "BUY_Signals",
        "SELL_Signals",
        "Combined_Signal",
    ],
)

with top_buy_tab:
    top_buy = buy_results.sort_values(
        ["BUY_Signals", "Available_Metrics"],
        ascending=[False, False],
    ).head(10)

    if top_buy.empty:
        st.info("No combined BUY stocks under the current settings.")
    else:
        st.dataframe(
            display_table(top_buy, ranking_columns),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Top 10 BUY",
            export_csv(top_buy),
            file_name="top_10_buy.csv",
            mime="text/csv",
            key="top_buy_download",
        )

with top_sell_tab:
    top_sell = sell_results.sort_values(
        ["SELL_Signals", "Available_Metrics"],
        ascending=[False, False],
    ).head(10)

    if top_sell.empty:
        st.info("No combined SELL stocks under the current settings.")
    else:
        st.dataframe(
            display_table(top_sell, ranking_columns),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Top 10 SELL",
            export_csv(top_sell),
            file_name="top_10_sell.csv",
            mime="text/csv",
            key="top_sell_download",
        )


# ============================================================
# SECTION 8 — COMPANY EXPLORER
# ============================================================

st.header("8. Company Explorer")

company_options = scored["Display_Name"].fillna("").astype(str)
ticker_options = scored["Ticker"].fillna("").astype(str)

company_labels = (
    company_options
    + " ("
    + ticker_options
    + ")"
)

company_labels = company_labels.where(
    ticker_options.str.strip() != "",
    company_options,
)

company_map = pd.Series(
    scored.index.to_list(),
    index=company_labels,
)

unique_labels = company_map.index.unique().tolist()

if unique_labels:
    selected_company_label = st.selectbox(
        "Select a company",
        unique_labels,
    )

    selected_index = company_map.loc[selected_company_label]

    if isinstance(selected_index, pd.Series):
        selected_index = selected_index.iloc[0]

    company = scored.loc[selected_index]

    st.subheader(str(company["Display_Name"]))

    if str(company["Ticker"]).strip():
        st.caption(f"Ticker: {company['Ticker']}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Combined Classification",
        str(company["Combined_Signal"]),
    )

    col2.metric(
        "BUY Signals",
        int(company["BUY_Signals"]),
    )

    col3.metric(
        "SELL Signals",
        int(company["SELL_Signals"]),
    )

    details = pd.DataFrame(
        {
            "Metric": [
                "Market Cap",
                "STA",
                "SNOA",
                "Altman Z-Score",
                "Gross Profitability",
                "Available Metrics",
            ],
            "Value": [
                format_market_cap(company["Market_Cap_Value"]),
                format_percent(company["STA"]),
                format_percent(company["SNOA"]),
                (
                    f"{company['Altman_Z']:.2f}"
                    if pd.notna(company["Altman_Z"])
                    else "N/A"
                ),
                format_percent(company["Gross_Profitability"]),
                (
                    f"{int(company['Available_Metrics'])} / 4"
                ),
            ],
        }
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Individual Metric Signals")

    signal_details = pd.DataFrame(
        {
            "Metric": [
                "STA",
                "SNOA",
                "Gross Profitability",
                "Altman Z-Score",
            ],
            "BUY Signal": [
                bool(company["STA_BUY"]),
                bool(company["SNOA_BUY"]),
                bool(company["GPA_BUY"]),
                bool(company["Z_BUY"]),
            ],
            "SELL Signal": [
                bool(company["STA_SELL"]),
                bool(company["SNOA_SELL"]),
                bool(company["GPA_SELL"]),
                bool(company["Z_SELL"]),
            ],
        }
    )

    st.dataframe(
        signal_details,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("No companies are available under the current filters.")


# ============================================================
# SECTION 9 — METHODOLOGY AND DATA COVERAGE
# ============================================================

st.header("9. Methodology and Data Coverage")

with st.expander("How the Screening Rules Work"):
    st.markdown(
        """
        **STA — Earnings Quality**

        Formula: (Net Income − Operating Cash Flow) / Total Assets.

        Lower values are ranked more favourably for BUY screening;
        higher values are ranked more favourably for SELL screening.

        **SNOA — Net Operating Assets**

        Formula: (Operating Assets − Operating Liabilities) / Total Assets.

        Lower values are ranked more favourably for BUY screening;
        higher values are ranked more favourably for SELL screening.

        **Gross Profitability**

        Formula: Gross Profit / Total Assets.

        Higher values are ranked more favourably for BUY screening;
        lower values are ranked more favourably for SELL screening.

        **Altman Z-Score**

        The dashboard retains fixed thresholds for the safe zone and
        distress zone. These are not affected by percentile sliders.

        **Percentile exclusion sliders**

        Each of STA, SNOA, and Gross Profitability has separate BUY and
        SELL exclusion sliders, ranging from 0% to 50%.

        For each side, the app ranks companies according to how favourable
        their metric value is for that signal, then excludes the specified
        worst-ranked percentage from eligibility. At 0%, no observations
        are excluded by the percentile rule. At 50%, approximately the
        worst half of the eligible observations are excluded.

        The ranking is calculated using the companies remaining after
        the minimum-market-cap filter. Missing metric values are not
        assigned a score and cannot receive that metric's signal.

        **Combined classification**

        A company is classified BUY or SELL only when it meets the
        minimum number of matching signals and has no opposing signals.
        Companies that fail these conditions are classified NEUTRAL.
        """
    )

with st.expander("Data Coverage"):
    coverage = show_coverage(
        scored,
        [
            ("STA", "STA"),
            ("SNOA", "SNOA"),
            ("Altman_Z", "Altman Z-Score"),
            ("Gross_Profitability", "Gross Profitability"),
        ],
    )

    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Important Limitations"):
    st.markdown(
        """
        - The dashboard is a quantitative screening tool, not personalized
          investment advice.
        - Percentile rankings are relative to the companies in the filtered
          universe. A company can rank well relative to peers without
          necessarily being financially healthy.
        - A BUY or SELL signal is not a forecast of future stock performance.
        - Financial metrics may use different fiscal reporting dates.
        - Missing data are retained as missing rather than treated as zero.
        - Results depend on the accuracy, completeness, and timing of the
          source data in universe.csv.
        - Market capitalization and stock prices can change after the source
          data were collected.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Results are calculated from universe.csv using the current sidebar "
    "settings. Percentile rankings update within the filtered manufacturing "
    "stock universe."
)
