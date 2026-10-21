
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
    initial_sidebar_state="expanded",
)

st.title("Manufacturing Stock Screening Dashboard")

st.markdown(
    """
    This dashboard screens a universe of publicly traded manufacturing
    companies using four financial measures: Altman Z-Score, accruals
    (STA), net operating assets (SNOA), and Gross Profitability.

    The objective is to compare companies using consistent financial
    criteria, identify firms that meet the selected BUY or SELL screening
    conditions, and explore the financial characteristics behind each result.

    Results depend on the financial data in `universe.csv` and the screening
    thresholds selected in the sidebar.
    """
)

st.caption(
    "Quantitative screening tool for financial analysis. "
    "Classifications are rule-based and are not investment recommendations."
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULTS = {
    "min_market_cap": 0.0,
    "sta_pct": 10,
    "snoa_pct": 10,
    "safe_z": 2.99,
    "distress_z": 1.81,
    "gpa_pct": 10,
    "min_signals": 2,
}

# Sidebar-only widgets: no duplicate controls on the main page.
WIDGET_KEYS = {
    "min_market_cap": ["cap_sidebar"],
    "sta_pct": ["sta_pct_sidebar"],
    "snoa_pct": ["snoa_pct_sidebar"],
    "safe_z": ["safe_z_sidebar"],
    "distress_z": ["distress_z_sidebar"],
    "gpa_pct": ["gpa_pct_sidebar"],
    "min_signals": ["signals_sidebar"],
}


def initialize_settings():
    for setting, default in DEFAULTS.items():
        if setting not in st.session_state:
            st.session_state[setting] = default

        for widget_key in WIDGET_KEYS[setting]:
            if widget_key not in st.session_state:
                st.session_state[widget_key] = default


def sync_setting(setting, source_key):
    st.session_state[setting] = st.session_state[source_key]


initialize_settings()


# ============================================================
# DATA AND FORMATTING HELPERS
# ============================================================

def parse_market_cap(value):
    """Convert market-cap values such as $2.4T, $850B, and $45.2M to dollars."""
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


def format_market_cap(value):
    if pd.isna(value) or str(value).strip() == "":
        return "—"
    return str(value)


def format_percent(value):
    if pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%"


def display_table(frame, columns):
    """Format ratios for display while preserving the underlying numeric data."""
    result = frame[columns].copy()

    if "Market Cap" in result.columns:
        result["Market Cap"] = result["Market Cap"].apply(
            format_market_cap
        )

    for column in [
        "STA",
        "SNOA",
        "Gross_Profitability",
    ]:
        if column in result.columns:
            result[column] = result[column].apply(format_percent)

    if "Altman_Z" in result.columns:
        result["Altman_Z"] = result["Altman_Z"].apply(
            lambda x: "—" if pd.isna(x) else f"{x:.2f}"
        )

    return result


def export_csv(frame):
    return frame.to_csv(index=False).encode("utf-8")


# ============================================================
# LOAD STOCK UNIVERSE
# ============================================================

DATA_FILE = Path(__file__).parent / "universe.csv"

if not DATA_FILE.exists():
    st.error(
        "Could not find universe.csv. Ensure it is in the same repository "
        "folder as app.py."
    )
    st.stop()


@st.cache_data
def load_data(file_path, modified_time):
    df = pd.read_csv(file_path)

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
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip(),
                errors="coerce",
            )

    if "Market Cap" in df.columns:
        df["Market_Cap_Value"] = df["Market Cap"].apply(
            parse_market_cap
        )
    else:
        df["Market_Cap_Value"] = np.nan

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
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    if "Ticker" in df.columns and "Ticker_guess" in df.columns:
        blank_ticker = df["Ticker"].str.strip().eq("")
        df.loc[blank_ticker, "Ticker"] = df.loc[
            blank_ticker, "Ticker_guess"
        ]

    return df


data = load_data(
    str(DATA_FILE),
    DATA_FILE.stat().st_mtime,
)

if data.empty:
    st.error("The stock universe is empty.")
    st.stop()


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
        "Missing required columns in universe.csv: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# SIDEBAR SLIDER HELPERS
# ============================================================

def add_cap_slider():
    key = "cap_sidebar"

    st.slider(
        "Minimum market cap ($ billions)",
        min_value=0.0,
        max_value=1000.0,
        step=10.0,
        key=key,
        on_change=sync_setting,
        args=("min_market_cap", key),
        help="Exclude companies below the selected market capitalization.",
    )


def add_percent_slider(setting, label, help_text):
    key = f"{setting}_sidebar"

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


def add_z_slider(setting, label, min_value, max_value, step):
    key = f"{setting}_sidebar"

    st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=key,
        on_change=sync_setting,
        args=(setting, key),
    )


def add_signal_slider():
    key = "signals_sidebar"

    st.slider(
        "Minimum BUY/SELL signals",
        min_value=1,
        max_value=4,
        step=1,
        key=key,
        on_change=sync_setting,
        args=("min_signals", key),
        help=(
            "Minimum number of matching signals required for a combined "
            "classification."
        ),
    )


# ============================================================
# SIDEBAR FILTERS — MATCH THE DASHBOARD STEP ORDER
# ============================================================

with st.sidebar:
    st.header("Screen Filters")
    st.caption(
        "Adjust the filters below to update the screening results."
    )

    st.subheader("Step 1 — Stock Universe")
    add_cap_slider()

    st.divider()

    st.subheader("Step 2 — STA (Accruals)")
    add_percent_slider(
        "sta_pct",
        "STA threshold (%)",
        "Lower STA is treated as better by this screening rule.",
    )

    st.divider()

    st.subheader("Step 3 — SNOA")
    add_percent_slider(
        "snoa_pct",
        "SNOA threshold (%)",
        "Lower SNOA is treated as better by this screening rule.",
    )

    st.divider()

    st.subheader("Step 4 — Altman Z-Score")
    add_z_slider(
        "safe_z",
        "Safe-zone threshold",
        1.0,
        5.0,
        0.01,
    )
    add_z_slider(
        "distress_z",
        "Distress threshold",
        0.5,
        3.0,
        0.01,
    )

    st.divider()

    st.subheader("Step 5 — Gross Profitability")
    add_percent_slider(
        "gpa_pct",
        "Minimum Gross Profitability (%)",
        "Higher Gross Profitability is treated as better.",
    )

    st.divider()

    st.subheader("Step 6 — Combined Screen")
    add_signal_slider()

    st.divider()

    if st.button("Reset all filters", use_container_width=True):
        for setting, default in DEFAULTS.items():
            st.session_state[setting] = default

            for widget_key in WIDGET_KEYS[setting]:
                st.session_state[widget_key] = default

        st.rerun()


# ============================================================
# STEP 1 — APPLY MARKET-CAP FILTER
# ============================================================

min_market_cap_dollars = st.session_state["min_market_cap"] * 1e9

filtered_data = data[
    data["Market_Cap_Value"].notna()
    & (
        data["Market_Cap_Value"]
        >= min_market_cap_dollars
    )
].copy()


# ============================================================
# CALCULATE INDIVIDUAL SIGNALS AND COMBINED CLASSIFICATION
# ============================================================

def score_companies(universe):
    scored = universe.copy()

    gpa_threshold = st.session_state["gpa_pct"] / 100
    sta_threshold = st.session_state["sta_pct"] / 100
    snoa_threshold = st.session_state["snoa_pct"] / 100

    safe_z = st.session_state["safe_z"]
    distress_z = st.session_state["distress_z"]
    min_signals = st.session_state["min_signals"]

    # STA: lower values are treated as better.
    scored["STA_BUY"] = (
        scored["STA"].notna()
        & (scored["STA"] <= -sta_threshold)
    )
    scored["STA_SELL"] = (
        scored["STA"].notna()
        & (scored["STA"] >= sta_threshold)
    )

    # SNOA: lower values are treated as better.
    scored["SNOA_BUY"] = (
        scored["SNOA"].notna()
        & (scored["SNOA"] <= -snoa_threshold)
    )
    scored["SNOA_SELL"] = (
        scored["SNOA"].notna()
        & (scored["SNOA"] >= snoa_threshold)
    )

    # Altman Z-Score: higher scores are treated as better.
    scored["Z_BUY"] = (
        scored["Altman_Z"].notna()
        & (scored["Altman_Z"] >= safe_z)
    )
    scored["Z_SELL"] = (
        scored["Altman_Z"].notna()
        & (scored["Altman_Z"] <= distress_z)
    )

    # Gross Profitability: higher values are treated as better.
    scored["GPA_BUY"] = (
        scored["Gross_Profitability"].notna()
        & (scored["Gross_Profitability"] >= gpa_threshold)
    )
    scored["GPA_SELL"] = (
        scored["Gross_Profitability"].notna()
        & (scored["Gross_Profitability"] < 0)
    )

    buy_columns = [
        "STA_BUY",
        "SNOA_BUY",
        "Z_BUY",
        "GPA_BUY",
    ]

    sell_columns = [
        "STA_SELL",
        "SNOA_SELL",
        "Z_SELL",
        "GPA_SELL",
    ]

    scored["BUY_Signals"] = scored[buy_columns].sum(axis=1)
    scored["SELL_Signals"] = scored[sell_columns].sum(axis=1)

    scored["Available_Metrics"] = scored[
        ["STA", "SNOA", "Altman_Z", "Gross_Profitability"]
    ].notna().sum(axis=1)

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
# INDIVIDUAL SCREEN TABLES
# ============================================================

def show_metric_table(frame, metric, buy_col, sell_col):
    valid = frame[frame[metric].notna()].copy()

    buy_mask = valid[buy_col].fillna(False).astype(bool)
    sell_mask = valid[sell_col].fillna(False).astype(bool)

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
        ]
        if column in valid.columns
    ]

    buy_tab, sell_tab, both_tab = st.tabs(
        ["BUY candidates", "SELL candidates", "Both signals"]
    )

    with buy_tab:
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
                file_name=f"{metric.lower()}_buy.csv",
                mime="text/csv",
                key=f"{metric}_buy_download",
            )

    with sell_tab:
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
                file_name=f"{metric.lower()}_sell.csv",
                mime="text/csv",
                key=f"{metric}_sell_download",
            )

    with both_tab:
        if both.empty:
            st.info("No companies currently qualify for both directions.")
        else:
            st.dataframe(
                display_table(both, columns),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# DATA COVERAGE
# ============================================================

def show_coverage(frame):
    metric_columns = {
        "Altman Z-Score": "Altman_Z",
        "STA": "STA",
        "SNOA": "SNOA",
        "Gross Profitability": "Gross_Profitability",
    }

    rows = []

    for label, column in metric_columns.items():
        valid_count = int(frame[column].notna().sum())
        missing_count = int(frame[column].isna().sum())

        rows.append(
            {
                "Metric": label,
                "Valid observations": valid_count,
                "Missing observations": missing_count,
                "Coverage (%)": (
                    valid_count / len(frame) * 100
                    if len(frame)
                    else 0
                ),
            }
        )

    coverage = pd.DataFrame(rows)
    coverage["Coverage (%)"] = coverage["Coverage (%)"].map(
        lambda x: f"{x:.1f}%"
    )

    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STEP 1 — STOCK UNIVERSE OVERVIEW
# ============================================================

st.divider()
st.header("Step 1 — Manufacturing Stock Universe")

st.markdown(
    """
    The screening process begins with the companies included in the
    dataset. The minimum market-cap filter determines which companies
    remain in the active universe.

    Financial data availability varies by metric, so the number of
    companies assessed may differ across individual screens.
    """
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Companies in dataset", f"{len(data):,}")
m2.metric("Companies in active universe", f"{len(scored_data):,}")
m3.metric(
    "Combined BUY",
    f"{int((scored_data['Combined_Result'] == 'BUY').sum()):,}",
)
m4.metric(
    "Combined SELL",
    f"{int((scored_data['Combined_Result'] == 'SELL').sum()):,}",
)

with st.expander("View active stock universe"):
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
# STEP 2 — STA
# ============================================================

st.divider()
st.header("Step 2 — STA (Accruals)")

st.markdown(
    """
    **Purpose:** STA measures accruals relative to a company's asset base.

    **Calculation:** (Net Income − Operating Cash Flow) / Total Assets.

    In this screen, lower STA values are treated as more favourable.
    Companies at or below the negative threshold receive a BUY signal,
    while companies at or above the positive threshold receive a SELL signal.
    """
)

show_metric_table(
    scored_data,
    "STA",
    "STA_BUY",
    "STA_SELL",
)


# ============================================================
# STEP 3 — SNOA
# ============================================================

st.divider()
st.header("Step 3 — SNOA (Net Operating Assets)")

st.markdown(
    """
    **Purpose:** SNOA measures net operating assets relative to total assets.

    **Calculation:** (Operating Assets − Operating Liabilities) / Total Assets.

    Lower SNOA values are treated as more favourable under this screening
    rule. The sidebar threshold determines which values qualify for BUY
    or SELL signals.
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
    **Purpose:** The Altman Z-Score combines financial ratios to assess
    a manufacturing company's financial condition and potential distress risk.

    The manufacturing model uses working capital, retained earnings, EBIT,
    market equity relative to total liabilities, and sales relative to assets.

    Traditional reference points are 2.99 for the safe zone and 1.81 for
    the distress zone. The sidebar lets you adjust the thresholds used to
    generate the dashboard's BUY and SELL signals.
    """
)

show_metric_table(
    scored_data,
    "Altman_Z",
    "Z_BUY",
    "Z_SELL",
)

z_values = scored_data["Altman_Z"].dropna()

if not z_values.empty:
    fig_z = px.histogram(
        x=z_values,
        nbins=40,
        title="Distribution of Altman Z-Scores",
        labels={
            "x": "Altman Z-Score",
            "count": "Companies",
        },
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
    **Purpose:** Gross Profitability measures gross profit relative to
    the company's asset base.

    **Calculation:** Gross Profit / Total Assets.

    Higher values indicate more gross profit generated per dollar of assets.
    Companies meeting the selected minimum threshold receive a BUY signal;
    negative values receive a SELL signal under this screen's rules.
    """
)

show_metric_table(
    scored_data,
    "Gross_Profitability",
    "GPA_BUY",
    "GPA_SELL",
)

gpa_values = scored_data["Gross_Profitability"].dropna()

if not gpa_values.empty:
    fig_gpa = px.histogram(
        x=gpa_values * 100,
        nbins=40,
        title="Distribution of Gross Profitability",
        labels={
            "x": "Gross Profitability (%)",
            "count": "Companies",
        },
    )

    fig_gpa.add_vline(
        x=st.session_state["gpa_pct"],
        line_dash="dash",
        annotation_text="BUY threshold",
    )

    st.plotly_chart(fig_gpa, use_container_width=True)


# ============================================================
# STEP 6 — COMBINED SCREEN
# ============================================================

st.divider()
st.header("Step 6 — Combined Screening Results")

st.markdown(
    """
    The combined screen brings together the four individual metrics to
    identify companies that meet multiple screening criteria.

    Each company can receive up to four BUY signals and four SELL signals,
    with one possible signal in each direction from each metric. The
    minimum signal count is adjustable in the sidebar.

    - **BUY:** Meets the minimum BUY-signal requirement and has no SELL signals.
    - **SELL:** Meets the minimum SELL-signal requirement and has no BUY signals.
    - **NEUTRAL:** Does not meet either combined classification rule.

    These labels represent the dashboard's screening methodology, not a
    prediction of future returns or a standalone investment decision.
    """
)

result_counts = (
    scored_data["Combined_Result"]
    .value_counts()
    .reindex(["BUY", "SELL", "NEUTRAL"], fill_value=0)
)

c1, c2, c3 = st.columns(3)
c1.metric("BUY", f"{int(result_counts['BUY']):,}")
c2.metric("SELL", f"{int(result_counts['SELL']):,}")
c3.metric("NEUTRAL", f"{int(result_counts['NEUTRAL']):,}")

fig_results = px.bar(
    x=result_counts.index,
    y=result_counts.values,
    title="Companies by Combined Classification",
    labels={
        "x": "Classification",
        "y": "Number of companies",
    },
)

st.plotly_chart(fig_results, use_container_width=True)

combined_filter = st.selectbox(
    "Choose results to view",
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

st.markdown(
    """
    These lists highlight companies that meet the combined screening rules.
    They are sorted by signal count and selected financial metrics to make
    the results easier to review. The ranking is specific to this screening
    methodology and should not be interpreted as a comprehensive investment
    ranking.
    """
)

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

buy_tab, sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

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

st.markdown(
    """
    Select an individual company to review its available metric values,
    individual screening signals, and combined classification.
    """
)

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

    company = company_options[
        company_options["_label"] == selected_label
    ].iloc[0]

    st.subheader(f"{company['Ticker']} — {company['Name']}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Market Cap",
        format_market_cap(company.get("Market Cap", np.nan)),
    )
    c2.metric(
        "Gross Profitability",
        format_percent(company.get("Gross_Profitability", np.nan)),
    )
    c3.metric(
        "STA",
        format_percent(company.get("STA", np.nan)),
    )
    c4.metric(
        "SNOA",
        format_percent(company.get("SNOA", np.nan)),
    )

    c5, c6, c7 = st.columns(3)

    z_value = company.get("Altman_Z", np.nan)

    c5.metric(
        "Altman Z-Score",
        "—" if pd.isna(z_value) else f"{z_value:.2f}",
    )
    c6.metric("BUY signals", int(company["BUY_Signals"]))
    c7.metric("SELL signals", int(company["SELL_Signals"]))

    st.write(
        "**Combined classification:**",
        company["Combined_Result"],
    )

    detail = pd.DataFrame(
        {
            "Metric": [
                "STA",
                "SNOA",
                "Altman Z-Score",
                "Gross Profitability",
            ],
            "Value": [
                company.get("STA", np.nan),
                company.get("SNOA", np.nan),
                company.get("Altman_Z", np.nan),
                company.get("Gross_Profitability", np.nan),
            ],
            "BUY signal": [
                bool(company["STA_BUY"]),
                bool(company["SNOA_BUY"]),
                bool(company["Z_BUY"]),
                bool(company["GPA_BUY"]),
            ],
            "SELL signal": [
                bool(company["STA_SELL"]),
                bool(company["SNOA_SELL"]),
                bool(company["Z_SELL"]),
                bool(company["GPA_SELL"]),
            ],
        }
    )

    def format_detail_value(row):
        value = row["Value"]

        if pd.isna(value):
            return "—"

        if row["Metric"] == "Altman Z-Score":
            return f"{value:.2f}"

        return format_percent(value)

    detail["Value"] = detail.apply(format_detail_value, axis=1)

    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# METHODOLOGY, COVERAGE, AND LIMITATIONS
# ============================================================

st.divider()
st.header("Methodology and Data Coverage")

with st.expander("Metric definitions and screening rules"):
    st.markdown(
        """
        **Altman Z-Score — Manufacturing model**

        The manufacturing model combines five financial ratios:

        - 1.2 × Working Capital / Total Assets
        - 1.4 × Retained Earnings / Total Assets
        - 3.3 × EBIT / Total Assets
        - 0.6 × Market Equity / Total Liabilities
        - 1.0 × Sales / Total Assets

        The dashboard uses the Altman Z-Score values supplied in the dataset.
        The traditional reference points are 2.99 for the safe zone and 1.81
        for the distress zone; the sidebar thresholds determine the signals
        actually used by this dashboard.

        **STA — Accruals**

        (Net Income − Operating Cash Flow) / Total Assets.

        Lower values are treated as more favourable by this screening rule.

        **SNOA — Net Operating Assets**

        (Operating Assets − Operating Liabilities) / Total Assets.

        Lower values are treated as more favourable by this screening rule.

        **Gross Profitability**

        Gross Profit / Total Assets.

        Higher values are treated as more favourable by this screening rule.

        **Combined classification**

        A company must meet the selected minimum number of BUY or SELL signals
        and have no signals in the opposite direction to receive that
        combined classification. All other companies are classified as
        NEUTRAL.
        """
    )

with st.expander("Data coverage and missing values"):
    st.markdown(
        """
        Coverage is reported for the active universe after applying the
        market-cap filter. Companies with missing values are excluded from
        the relevant individual metric screen.

        The number of companies assessed may differ across metrics because
        financial data availability varies. Market capitalization and
        financial figures may also refer to different dates.
        """
    )

    show_coverage(filtered_data)

with st.expander("Interpretation and limitations"):
    st.markdown(
        """
        This dashboard is a quantitative screening tool, not a complete
        valuation model or a comprehensive assessment of a company's
        investment merits.

        - Results depend on the selected thresholds.
        - Historical financial data may not reflect current conditions.
        - Accounting differences and unusual events can affect ratios.
        - Altman Z-Score thresholds are model-specific and should be
          interpreted in the context of manufacturing companies.
        - STA and SNOA screening directions are methodological choices and
          should be considered alongside other financial information.
        - Missing or outdated financial data may affect results.
        - A BUY or SELL classification does not guarantee future performance
          and should not be used in isolation.
        """
    )

st.caption(
    "Screening results are generated from universe.csv using the "
    "selected sidebar filters. Review the methodology and data coverage "
    "before interpreting classifications."
)
