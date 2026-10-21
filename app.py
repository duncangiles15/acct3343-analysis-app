
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE SETUP AND DATA
# ============================================================

st.set_page_config(
    page_title="Stock Screening Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Stock Screening Analysis")
st.markdown(
    """
    This application screens a universe of publicly traded companies using
    four financial techniques. Each technique identifies companies with
    favorable or unfavorable characteristics. The individual signals are
    then combined to produce a ranked shortlist for further investigation.
    """
)


@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")
    df.columns = df.columns.str.strip()

    if "Ticker" not in df.columns:
        if "Ticker_guess" in df.columns:
            df["Ticker"] = df["Ticker_guess"]
        else:
            raise ValueError("The dataset requires a Ticker column.")

    if "Name" not in df.columns:
        df["Name"] = df["Ticker"]

    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip()
    df["Name"] = df["Name"].fillna(df["Ticker"]).astype(str).str.strip()

    def parse_market_cap(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.number)):
            return float(value)

        text = str(value).strip().replace(",", "").replace("$", "")

        if not text or text.lower() in {"nan", "none", "n/a", "na", "-"}:
            return np.nan

        multiplier = 1.0
        suffix = text[-1:].upper()

        if suffix == "T":
            multiplier = 1e12
            text = text[:-1]
        elif suffix == "B":
            multiplier = 1e9
            text = text[:-1]
        elif suffix == "M":
            multiplier = 1e6
            text = text[:-1]
        elif suffix == "K":
            multiplier = 1e3
            text = text[:-1]

        try:
            return float(text) * multiplier
        except (ValueError, TypeError):
            return np.nan

    if "Market Cap" in df.columns:
        df["Market Cap"] = df["Market Cap"].apply(parse_market_cap)
    else:
        df["Market Cap"] = np.nan

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
        "X1_Working_Capital_Pct",
        "X2_Retained_Earnings_Pct",
        "X3_EBIT_Pct",
        "X4_Market_Equity_to_Liabilities_Pct",
        "X5_Sales_to_Assets",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df["Ticker"].ne("")].copy()
    df = df.drop_duplicates(subset="Ticker", keep="first")

    return df


try:
    raw_df = load_data()
except Exception as error:
    st.error(f"Could not load universe.csv: {error}")
    st.stop()


required_columns = [
    "Gross_Profitability",
    "STA",
    "SNOA",
    "Altman_Z",
]

missing_columns = [
    column for column in required_columns
    if column not in raw_df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# 2. SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Screening Settings")

minimum_market_cap_millions = st.sidebar.number_input(
    "Minimum market cap ($ millions)",
    min_value=0,
    max_value=10_000_000,
    value=0,
    step=100,
)

st.sidebar.subheader("Gross Profitability (GPA)")

gpa_top_pct = st.sidebar.slider(
    "GPA: top percentile for BUY",
    5, 30, 10, 5
) / 100

gpa_bottom_pct = st.sidebar.slider(
    "GPA: bottom percentile for SELL",
    5, 30, 10, 5
) / 100

st.sidebar.subheader("Accruals (STA)")

sta_bottom_pct = st.sidebar.slider(
    "STA: bottom percentile for BUY",
    5, 30, 10, 5
) / 100

sta_top_pct = st.sidebar.slider(
    "STA: top percentile for SELL",
    5, 30, 10, 5
) / 100

st.sidebar.subheader("Scaled Net Operating Assets (SNOA)")

snoa_bottom_pct = st.sidebar.slider(
    "SNOA: bottom percentile for BUY",
    5, 30, 10, 5
) / 100

snoa_top_pct = st.sidebar.slider(
    "SNOA: top percentile for SELL",
    5, 30, 10, 5
) / 100

st.sidebar.subheader("Altman Z-Score")

altman_distress_threshold = st.sidebar.number_input(
    "Distress threshold",
    min_value=-20.0,
    max_value=20.0,
    value=1.81,
    step=0.01,
    format="%.2f",
)

altman_safe_threshold = st.sidebar.number_input(
    "Safe-zone threshold",
    min_value=-20.0,
    max_value=20.0,
    value=2.99,
    step=0.01,
    format="%.2f",
)

st.sidebar.subheader("Final Classification")

minimum_buy_signals = st.sidebar.selectbox(
    "Minimum BUY signals",
    [1, 2, 3],
    index=1,
)

minimum_sell_signals = st.sidebar.selectbox(
    "Minimum SELL signals",
    [1, 2, 3, 4],
    index=1,
)

if gpa_top_pct + gpa_bottom_pct > 1:
    st.sidebar.error("GPA percentiles overlap.")
    st.stop()

if sta_top_pct + sta_bottom_pct > 1:
    st.sidebar.error("STA percentiles overlap.")
    st.stop()

if snoa_top_pct + snoa_bottom_pct > 1:
    st.sidebar.error("SNOA percentiles overlap.")
    st.stop()

if altman_safe_threshold <= altman_distress_threshold:
    st.sidebar.error(
        "The safe-zone threshold must exceed the distress threshold."
    )
    st.stop()


# ============================================================
# 3. DEFINE THE INVESTMENT UNIVERSE
# ============================================================

df = raw_df.copy()

minimum_market_cap = minimum_market_cap_millions * 1_000_000

if minimum_market_cap > 0:
    df = df[
        df["Market Cap"].notna()
        & (df["Market Cap"] >= minimum_market_cap)
    ].copy()

if df.empty:
    st.warning("No companies match the current market-cap filter.")
    st.stop()


def percentile_flag(series, tickers, percentile, select_lowest):
    """Select the lowest or highest percentile of nonmissing values."""
    flags = pd.Series(False, index=series.index, dtype=bool)
    eligible = series.notna()

    if not eligible.any():
        return flags

    ranked = pd.DataFrame({
        "value": series.loc[eligible],
        "ticker": tickers.loc[eligible].astype(str),
    })

    ranked = ranked.sort_values(
        ["value", "ticker"],
        ascending=[select_lowest, True],
        kind="mergesort",
    )

    count = max(1, math.ceil(len(ranked) * percentile))
    flags.loc[ranked.head(count).index] = True

    return flags


def flagged_table(data, mask, metric, ascending, limit=10):
    """Compact company table for one screening step."""
    result = data.loc[mask].copy()

    if result.empty:
        return result

    result = result.sort_values(
        [metric, "Ticker"],
        ascending=[ascending, True],
        na_position="last",
    ).head(limit)

    columns = [
        column for column in [
            "Ticker",
            "Name",
            "Market Cap",
            metric,
        ]
        if column in result.columns
    ]

    return result[columns]


def display_table(data):
    """Prepare readable tables without changing the underlying values."""
    result = data.copy()

    if "Market Cap" in result.columns:
        def format_market_cap(value):
            if pd.isna(value):
                return "N/A"
            if value >= 1e12:
                return f"${value / 1e12:.2f}T"
            if value >= 1e9:
                return f"${value / 1e9:.2f}B"
            if value >= 1e6:
                return f"${value / 1e6:.2f}M"
            if value >= 1e3:
                return f"${value / 1e3:.2f}K"
            return f"${value:,.0f}"

        result["Market Cap"] = result["Market Cap"].apply(format_market_cap)

    result = result.rename(columns={
        "Gross_Profitability": "GPA",
        "Altman_Z": "Altman Z",
        "Buy_Signals": "BUY Signals",
        "Sell_Signals": "SELL Signals",
        "Available_Core_Metrics": "Metrics Available",
        "Screening_Result": "Screening Result",
    })

    return result


st.header("Step 1 — Define the Investment Universe")

st.markdown(
    """
    Before comparing financial characteristics, establish which companies
    are included in the analysis. The market-cap filter is optional and
    applies to all four screening techniques.
    """
)

universe1, universe2, universe3 = st.columns(3)

universe1.metric("Companies in Universe", f"{len(df):,}")

universe2.metric(
    "Companies with GPA",
    f"{df['Gross_Profitability'].notna().sum():,}",
)

universe3.metric(
    "Companies with all four metrics",
    f"{(
        df['Gross_Profitability'].notna()
        & df['STA'].notna()
        & df['SNOA'].notna()
        & df['Altman_Z'].notna()
    ).sum():,}",
)

st.caption(
    "Missing financial data is not treated as a favorable or unfavorable "
    "signal. Each metric ranks only companies with an available value."
)


# ============================================================
# 4. SCREENING TECHNIQUE 1 — GPA
# ============================================================

st.divider()
st.header("Step 2 — Screen for Gross Profitability (GPA)")

st.markdown(
    """
    **Question:** How effectively does a company generate gross profit
    relative to its asset base?

    **Formula:** Gross Profit ÷ Total Assets

    **Why it matters:** Gross profitability provides a measure of how
    productive a company's assets are at generating gross profit. Under
    this screening strategy, higher values are treated as favorable.
    """
)

st.markdown(
    f"""
    **Active rule**

    - **BUY signal:** GPA is in the highest {gpa_top_pct:.0%} of eligible companies.
    - **SELL signal:** GPA is in the lowest {gpa_bottom_pct:.0%} of eligible companies.
    """
)

df["GPA_BUY"] = percentile_flag(
    df["Gross_Profitability"], df["Ticker"],
    gpa_top_pct, select_lowest=False,
)

df["GPA_SELL"] = percentile_flag(
    df["Gross_Profitability"], df["Ticker"],
    gpa_bottom_pct, select_lowest=True,
)

gpa1, gpa2, gpa3 = st.columns(3)

gpa1.metric(
    "Companies Ranked",
    f"{df['Gross_Profitability'].notna().sum():,}",
)
gpa2.metric("BUY Signals", f"{df['GPA_BUY'].sum():,}")
gpa3.metric("SELL Signals", f"{df['GPA_SELL'].sum():,}")

gpa_buy_tab, gpa_sell_tab = st.tabs(
    ["Highest GPA — BUY Signals", "Lowest GPA — SELL Signals"]
)

with gpa_buy_tab:
    st.caption("Examples from the highest-GPA group.")
    st.dataframe(
        display_table(
            flagged_table(
                df, df["GPA_BUY"], "Gross_Profitability", False
            ).rename(columns={"Gross_Profitability": "GPA"})
        ),
        use_container_width=True,
        hide_index=True,
    )

with gpa_sell_tab:
    st.caption("Examples from the lowest-GPA group.")
    st.dataframe(
        display_table(
            flagged_table(
                df, df["GPA_SELL"], "Gross_Profitability", True
            ).rename(columns={"Gross_Profitability": "GPA"})
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 5. SCREENING TECHNIQUE 2 — STA
# ============================================================

st.divider()
st.header("Step 3 — Screen Earnings Quality Using STA")

st.markdown(
    """
    **Question:** How different are accounting earnings from operating
    cash flow, relative to the company's assets?

    **Formula:** (Net Income − Operating Cash Flow) ÷ Total Assets

    **Why it matters:** Earnings and cash flow can differ because of
    accrual accounting. High positive accruals can indicate that reported
    earnings are less strongly supported by operating cash flow. For this
    screening strategy, lower STA values are treated as favorable.
    """
)

st.markdown(
    f"""
    **Active rule**

    - **BUY signal:** STA is in the lowest {sta_bottom_pct:.0%} of eligible companies.
    - **SELL signal:** STA is in the highest {sta_top_pct:.0%} of eligible companies.
    """
)

df["STA_BUY"] = percentile_flag(
    df["STA"], df["Ticker"],
    sta_bottom_pct, select_lowest=True,
)

df["STA_SELL"] = percentile_flag(
    df["STA"], df["Ticker"],
    sta_top_pct, select_lowest=False,
)

sta1, sta2, sta3 = st.columns(3)

sta1.metric("Companies Ranked", f"{df['STA'].notna().sum():,}")
sta2.metric("BUY Signals", f"{df['STA_BUY'].sum():,}")
sta3.metric("SELL Signals", f"{df['STA_SELL'].sum():,}")

sta_buy_tab, sta_sell_tab = st.tabs(
    ["Lowest STA — BUY Signals", "Highest STA — SELL Signals"]
)

with sta_buy_tab:
    st.caption("Examples from the lowest-STA group.")
    st.dataframe(
        display_table(flagged_table(df, df["STA_BUY"], "STA", True)),
        use_container_width=True,
        hide_index=True,
    )

with sta_sell_tab:
    st.caption("Examples from the highest-STA group.")
    st.dataframe(
        display_table(flagged_table(df, df["STA_SELL"], "STA", False)),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 6. SCREENING TECHNIQUE 3 — SNOA
# ============================================================

st.divider()
st.header("Step 4 — Screen Operating Assets Using SNOA")

st.markdown(
    """
    **Question:** How large are a company's net operating assets relative
    to its total assets?

    **Metric:** The SNOA values already supplied in `universe.csv`.

    **Why it matters:** Net operating assets reflect operating assets net
    of operating liabilities. A high level of net operating assets can be
    relevant when investigating how much of a company's asset base is tied
    up in net operating resources. This screening strategy treats lower
    SNOA values as favorable.

    **Data note:** This app uses the existing SNOA column; it does not
    recalculate SNOA from the underlying accounts.
    """
)

st.markdown(
    f"""
    **Active rule**

    - **BUY signal:** SNOA is in the lowest {snoa_bottom_pct:.0%} of eligible companies.
    - **SELL signal:** SNOA is in the highest {snoa_top_pct:.0%} of eligible companies.
    """
)

df["SNOA_BUY"] = percentile_flag(
    df["SNOA"], df["Ticker"],
    snoa_bottom_pct, select_lowest=True,
)

df["SNOA_SELL"] = percentile_flag(
    df["SNOA"], df["Ticker"],
    snoa_top_pct, select_lowest=False,
)

snoa1, snoa2, snoa3 = st.columns(3)

snoa1.metric("Companies Ranked", f"{df['SNOA'].notna().sum():,}")
snoa2.metric("BUY Signals", f"{df['SNOA_BUY'].sum():,}")
snoa3.metric("SELL Signals", f"{df['SNOA_SELL'].sum():,}")

snoa_buy_tab, snoa_sell_tab = st.tabs(
    ["Lowest SNOA — BUY Signals", "Highest SNOA — SELL Signals"]
)

with snoa_buy_tab:
    st.caption("Examples from the lowest-SNOA group.")
    st.dataframe(
        display_table(flagged_table(df, df["SNOA_BUY"], "SNOA", True)),
        use_container_width=True,
        hide_index=True,
    )

with snoa_sell_tab:
    st.caption("Examples from the highest-SNOA group.")
    st.dataframe(
        display_table(flagged_table(df, df["SNOA_SELL"], "SNOA", False)),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 7. SCREENING TECHNIQUE 4 — ALTMAN Z
# ============================================================

st.divider()
st.header("Step 5 — Assess Financial Distress Using Altman Z-Score")

st.markdown(
    """
    **Question:** Does the company's Z-Score indicate elevated financial
    distress risk under the selected thresholds?

    **Metric:** The existing Altman Z-Score in `universe.csv`.

    **Why it matters:** Altman Z-Score combines financial ratios to assess
    the risk of corporate financial distress. It provides a different
    perspective from profitability and accrual-based measures.

    The default thresholds shown here are commonly used for the
    manufacturing version of the model. They may not be appropriate for
    every industry.
    """
)

st.markdown(
    f"""
    **Active rule**

    - **SELL signal:** Altman Z-Score is below **{altman_distress_threshold:.2f}**.
    - **Safe-zone context:** Altman Z-Score is at or above **{altman_safe_threshold:.2f}**.
    - **Important:** The safe zone does not automatically generate a BUY signal.
    """
)

df["ALTMAN_SELL"] = (
    df["Altman_Z"].notna()
    & (df["Altman_Z"] < altman_distress_threshold)
)

df["ALTMAN_SAFE"] = (
    df["Altman_Z"].notna()
    & (df["Altman_Z"] >= altman_safe_threshold)
)

alt1, alt2, alt3, alt4 = st.columns(4)

alt1.metric("Companies Ranked", f"{df['Altman_Z'].notna().sum():,}")
alt2.metric("Distress SELL Signals", f"{df['ALTMAN_SELL'].sum():,}")
alt3.metric("Safe-Zone Companies", f"{df['ALTMAN_SAFE'].sum():,}")
alt4.metric(
    "Between Thresholds",
    f"{(
        df['Altman_Z'].notna()
        & (df['Altman_Z'] >= altman_distress_threshold)
        & (df['Altman_Z'] < altman_safe_threshold)
    ).sum():,}",
)

alt_distress_tab, alt_safe_tab = st.tabs(
    ["Below Distress Threshold", "At or Above Safe-Zone Threshold"]
)

with alt_distress_tab:
    distress_df = df[df["ALTMAN_SELL"]].sort_values(
        ["Altman_Z", "Ticker"], ascending=[True, True]
    ).head(10)

    st.caption("Companies with the lowest Z-Scores among distress-flagged companies.")
    st.dataframe(
        display_table(
            distress_df[["Ticker", "Name", "Market Cap", "Altman_Z"]]
            .rename(columns={"Altman_Z": "Altman Z"})
        ),
        use_container_width=True,
        hide_index=True,
    )

with alt_safe_tab:
    safe_df = df[df["ALTMAN_SAFE"]].sort_values(
        ["Altman_Z", "Ticker"], ascending=[False, True]
    ).head(10)

    st.caption("Examples of companies in the selected safe zone.")
    st.dataframe(
        display_table(
            safe_df[["Ticker", "Name", "Market Cap", "Altman_Z"]]
            .rename(columns={"Altman_Z": "Altman Z"})
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 8. COMBINE THE FOUR TECHNIQUES
# ============================================================

st.divider()
st.header("Step 6 — Combine the Screening Signals")

st.markdown(
    """
    Each company now has signals from the four techniques. GPA, STA, and
    SNOA can contribute BUY or SELL signals according to their percentile
    rules. Altman Z contributes a SELL signal when it falls below the
    distress threshold.

    The final classification compares the total BUY and SELL signals.
    A company must have enough signals under the selected rules, and the
    winning side must have more signals than the other side.
    """
)

df["Buy_Signals"] = (
    df["GPA_BUY"].astype(int)
    + df["STA_BUY"].astype(int)
    + df["SNOA_BUY"].astype(int)
)

df["Sell_Signals"] = (
    df["GPA_SELL"].astype(int)
    + df["STA_SELL"].astype(int)
    + df["SNOA_SELL"].astype(int)
    + df["ALTMAN_SELL"].astype(int)
)

df["Available_Core_Metrics"] = (
    df["Gross_Profitability"].notna().astype(int)
    + df["STA"].notna().astype(int)
    + df["SNOA"].notna().astype(int)
    + df["Altman_Z"].notna().astype(int)
)

df["Enough_Data"] = df["Available_Core_Metrics"] >= 2


def classify_company(row):
    if not row["Enough_Data"]:
        return "REVIEW — Limited Data"

    buy = int(row["Buy_Signals"])
    sell = int(row["Sell_Signals"])

    if buy >= minimum_buy_signals and buy > sell:
        return "Potential LONG"

    if sell >= minimum_sell_signals and sell > buy:
        return "Potential SHORT"

    if buy > 0 and sell > 0:
        return "MIXED Signals"

    return "NEUTRAL"


df["Screening_Result"] = df.apply(classify_company, axis=1)

outcome_order = [
    "Potential LONG",
    "Potential SHORT",
    "MIXED Signals",
    "NEUTRAL",
    "REVIEW — Limited Data",
]

outcome_counts = (
    df["Screening_Result"]
    .value_counts()
    .reindex(outcome_order, fill_value=0)
)

overview1, overview2, overview3, overview4, overview5 = st.columns(5)

overview1.metric("Potential LONG", f"{outcome_counts['Potential LONG']:,}")
overview2.metric("Potential SHORT", f"{outcome_counts['Potential SHORT']:,}")
overview3.metric("Mixed", f"{outcome_counts['MIXED Signals']:,}")
overview4.metric("Neutral", f"{outcome_counts['NEUTRAL']:,}")
overview5.metric(
    "Limited Data",
    f"{outcome_counts['REVIEW — Limited Data']:,}",
)

st.subheader("How the Individual Signals Add Up")

signal_summary = pd.DataFrame({
    "Screening Technique": [
        "GPA",
        "STA",
        "SNOA",
        "Altman Z",
    ],
    "BUY Signals": [
        int(df["GPA_BUY"].sum()),
        int(df["STA_BUY"].sum()),
        int(df["SNOA_BUY"].sum()),
        0,
    ],
    "SELL Signals": [
        int(df["GPA_SELL"].sum()),
        int(df["STA_SELL"].sum()),
        int(df["SNOA_SELL"].sum()),
        int(df["ALTMAN_SELL"].sum()),
    ],
})

st.dataframe(signal_summary, use_container_width=True, hide_index=True)

st.caption(
    "Signal counts can overlap: a company may receive signals from multiple "
    "techniques. These are signal totals, not unique company counts."
)

outcome_chart_data = outcome_counts.rename_axis(
    "Screening Result"
).reset_index(name="Companies")

outcome_chart = px.bar(
    outcome_chart_data,
    x="Screening Result",
    y="Companies",
    text="Companies",
    title="Companies by Final Screening Classification",
)

outcome_chart.update_layout(
    xaxis_title="Screening Result",
    yaxis_title="Number of Companies",
    showlegend=False,
)

outcome_chart.update_traces(textposition="outside")

st.plotly_chart(outcome_chart, use_container_width=True)

scatter_df = df[
    df["Gross_Profitability"].notna()
    & df["STA"].notna()
].copy()

if not scatter_df.empty:
    st.subheader("Visual Check: GPA vs. STA")

    scatter_chart = px.scatter(
        scatter_df,
        x="Gross_Profitability",
        y="STA",
        color="Screening_Result",
        hover_name="Ticker",
        hover_data={
            "Name": True,
            "SNOA": ":.4f",
            "Altman_Z": ":.2f",
            "Buy_Signals": True,
            "Sell_Signals": True,
        },
        labels={
            "Gross_Profitability": "Gross Profitability / Assets",
            "STA": "STA",
            "Screening_Result": "Screening Result",
        },
        title="Gross Profitability vs. Accruals",
    )

    st.plotly_chart(scatter_chart, use_container_width=True)


# ============================================================
# 9. ARRIVE AT THE TOP 10 CANDIDATES
# ============================================================

st.divider()
st.header("Step 7 — Final Shortlist: Top 10 BUY and SELL Candidates")

st.markdown(
    """
    The final lists are generated only after all four techniques have been
    applied and their signals combined.

    The lists contain companies classified as Potential LONG or Potential
    SHORT. Companies with mixed signals, insufficient data, or no qualifying
    signal advantage are not included in these two shortlists.
    """
)

buy_candidates = df[
    df["Screening_Result"] == "Potential LONG"
].copy()

buy_candidates = buy_candidates.sort_values(
    [
        "Buy_Signals",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
        "Ticker",
    ],
    ascending=[False, False, True, True, False, True],
    na_position="last",
)

sell_candidates = df[
    df["Screening_Result"] == "Potential SHORT"
].copy()

sell_candidates = sell_candidates.sort_values(
    [
        "Sell_Signals",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
        "Ticker",
    ],
    ascending=[False, True, False, False, True, True],
    na_position="last",
)

buy_top10 = buy_candidates.head(10)
sell_top10 = sell_candidates.head(10)


def shortlist_columns(data):
    columns = [
        "Ticker",
        "Name",
        "Market Cap",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
        "Buy_Signals",
        "Sell_Signals",
        "Available_Core_Metrics",
        "Screening_Result",
    ]

    return data[[column for column in columns if column in data.columns]]


buy_tab, sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

with buy_tab:
    st.subheader("Top 10 Potential LONG Candidates")
    st.caption(
        "Ordered first by BUY signal count, then by higher GPA, lower STA, "
        "lower SNOA, and higher Altman Z as tie-breakers."
    )

    if buy_top10.empty:
        st.info(
            "No companies currently meet the Potential LONG criteria. "
            "Adjust the screening settings in the sidebar to explore alternatives."
        )
    else:
        st.dataframe(
            display_table(shortlist_columns(buy_top10)),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Top 10 BUY as CSV",
            data=buy_top10.to_csv(index=False).encode("utf-8"),
            file_name="top_10_buy_candidates.csv",
            mime="text/csv",
        )

with sell_tab:
    st.subheader("Top 10 Potential SHORT Candidates")
    st.caption(
        "Ordered first by SELL signal count, then by lower GPA, higher STA, "
        "higher SNOA, and lower Altman Z as tie-breakers."
    )

    if sell_top10.empty:
        st.info(
            "No companies currently meet the Potential SHORT criteria. "
            "Adjust the screening settings in the sidebar to explore alternatives."
        )
    else:
        st.dataframe(
            display_table(shortlist_columns(sell_top10)),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Top 10 SELL as CSV",
            data=sell_top10.to_csv(index=False).encode("utf-8"),
            file_name="top_10_sell_candidates.csv",
            mime="text/csv",
        )


# ============================================================
# 10. COMPANY EXPLORER
# ============================================================

st.divider()
st.header("Company Explorer")

search_text = st.text_input(
    "Search by ticker or company name",
    placeholder="Enter a ticker or company name...",
)

search_df = df.copy()

if search_text.strip():
    query = search_text.strip().lower()

    matches = (
        search_df["Ticker"].str.lower().str.contains(query, na=False)
        | search_df["Name"].str.lower().str.contains(query, na=False)
    )

    search_df = search_df[matches]

if search_df.empty:
    st.info("No companies matched that search.")
else:
    options = (
        search_df["Ticker"] + " — " + search_df["Name"]
    ).tolist()

    selected_label = st.selectbox("Select a company", options)
    selected_ticker = selected_label.split(" — ", 1)[0]
    selected_row = df[df["Ticker"] == selected_ticker].iloc[0]

    st.subheader(f"{selected_row['Name']} ({selected_row['Ticker']})")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Screening Result", selected_row["Screening_Result"])
    c2.metric("BUY Signals", int(selected_row["Buy_Signals"]))
    c3.metric("SELL Signals", int(selected_row["Sell_Signals"]))
    c4.metric(
        "Metrics Available",
        f"{int(selected_row['Available_Core_Metrics'])} / 4",
    )

    st.markdown("#### Explanation of This Company's Signals")

    explanations = []

    metric_rules = [
        (
            "GPA",
            "Gross_Profitability",
            "GPA_BUY",
            "GPA_SELL",
            "top",
            "bottom",
        ),
        (
            "STA",
            "STA",
            "STA_BUY",
            "STA_SELL",
            "bottom",
            "top",
        ),
        (
            "SNOA",
            "SNOA",
            "SNOA_BUY",
            "SNOA_SELL",
            "bottom",
            "top",
        ),
    ]

    for label, value_column, buy_column, sell_column, buy_side, sell_side in metric_rules:
        value = selected_row[value_column]

        if pd.isna(value):
            explanations.append(
                f"**{label}:** No value available; this metric was not ranked."
            )
        elif selected_row[buy_column]:
            explanations.append(
                f"**{label}:** BUY signal — within the selected {buy_side} "
                "percentile group."
            )
        elif selected_row[sell_column]:
            explanations.append(
                f"**{label}:** SELL signal — within the selected {sell_side} "
                "percentile group."
            )
        else:
            explanations.append(
                f"**{label}:** Available, but outside both selected signal groups."
            )

    if pd.isna(selected_row["Altman_Z"]):
        explanations.append(
            "**Altman Z:** Missing; no distress signal was assigned."
        )
    elif selected_row["ALTMAN_SELL"]:
        explanations.append(
            f"**Altman Z:** SELL signal because the score is below "
            f"{altman_distress_threshold:.2f}."
        )
    elif selected_row["ALTMAN_SAFE"]:
        explanations.append(
            f"**Altman Z:** In the selected safe zone (at or above "
            f"{altman_safe_threshold:.2f}); this does not create a BUY signal."
        )
    else:
        explanations.append(
            "**Altman Z:** Between the selected distress and safe-zone thresholds."
        )

    for explanation in explanations:
        st.markdown(f"- {explanation}")

    company_columns = [
        column for column in [
            "Ticker",
            "Name",
            "Rank",
            "Market Cap",
            "Price",
            "Gross_Profit",
            "Total_Assets",
            "Gross_Profitability",
            "STA",
            "SNOA",
            "Altman_Z",
            "Buy_Signals",
            "Sell_Signals",
            "Available_Core_Metrics",
            "Screening_Result",
        ]
        if column in selected_row.index
    ]

    st.markdown("#### Company Data")
    st.dataframe(
        display_table(
            selected_row[company_columns].to_frame(name="Value").reset_index()
            .rename(columns={"index": "Field"})
            .set_index("Field")
            .T
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 11. ALL SCREENED COMPANIES
# ============================================================

st.divider()
st.header("All Screened Companies")

long_tab, short_tab, mixed_tab, neutral_tab, all_tab = st.tabs(
    [
        "Potential LONG",
        "Potential SHORT",
        "Mixed / Review",
        "Neutral",
        "All Companies",
    ]
)

with long_tab:
    subset = df[df["Screening_Result"] == "Potential LONG"]
    st.write(f"{len(subset):,} companies")
    st.dataframe(
        display_table(shortlist_columns(subset)),
        use_container_width=True,
        hide_index=True,
    )

with short_tab:
    subset = df[df["Screening_Result"] == "Potential SHORT"]
    st.write(f"{len(subset):,} companies")
    st.dataframe(
        display_table(shortlist_columns(subset)),
        use_container_width=True,
        hide_index=True,
    )

with mixed_tab:
    subset = df[
        df["Screening_Result"].isin(
            ["MIXED Signals", "REVIEW — Limited Data"]
        )
    ]
    st.write(f"{len(subset):,} companies")
    st.dataframe(
        display_table(shortlist_columns(subset)),
        use_container_width=True,
        hide_index=True,
    )

with neutral_tab:
    subset = df[df["Screening_Result"] == "NEUTRAL"]
    st.write(f"{len(subset):,} companies")
    st.dataframe(
        display_table(shortlist_columns(subset)),
        use_container_width=True,
        hide_index=True,
    )

with all_tab:
    st.write(f"{len(df):,} companies")
    st.dataframe(
        display_table(shortlist_columns(df)),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download All Screened Companies as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="all_screened_companies.csv",
        mime="text/csv",
    )


# ============================================================
# 12. METHODOLOGY AND LIMITATIONS
# ============================================================

with st.expander("Methodology and Definitions"):
    st.markdown(
        """
        ### Gross Profitability (GPA)

        Gross Profit / Total Assets. Higher values are treated as favorable.
        Companies in the selected top percentile receive BUY signals; those
        in the bottom percentile receive SELL signals.

        ### STA

        (Net Income − Operating Cash Flow) / Total Assets. Lower values are
        treated as favorable for this strategy. Companies in the bottom
        percentile receive BUY signals; those in the top percentile receive
        SELL signals.

        ### SNOA

        Uses the SNOA column already present in the dataset. Lower values
        are treated as favorable. Companies in the bottom percentile receive
        BUY signals; those in the top percentile receive SELL signals.

        ### Altman Z-Score

        Uses the existing Altman Z-Score. Values below the configured distress
        threshold generate SELL signals. Values at or above the safe-zone
        threshold are shown as context only, not as automatic BUY signals.

        ### Combining Signals

        BUY signals come from GPA, STA, and SNOA. SELL signals can come from
        all three metrics plus Altman Z. The selected minimum signal counts
        and the relative BUY/SELL totals determine the final classification.

        ### Limitations

        - Percentile groups depend on the filtered investment universe.
        - Missing values reduce the information available for a company.
        - Metrics may refer to different reporting dates.
        - SNOA uses the existing dataset definition and should be verified
          against the original data source.
        - Altman Z-Score thresholds may not fit every industry.
        - Screening results do not establish fair value or predict future returns.
        """
    )

with st.expander("Data Coverage"):
    coverage_rows = []

    for label, column in [
        ("GPA", "Gross_Profitability"),
        ("STA", "STA"),
        ("SNOA", "SNOA"),
        ("Altman Z-Score", "Altman_Z"),
    ]:
        available = int(df[column].notna().sum())

        coverage_rows.append({
            "Metric": label,
            "Available": available,
            "Missing": len(df) - available,
            "Coverage (%)": round(100 * available / len(df), 1),
        })

    st.dataframe(
        pd.DataFrame(coverage_rows),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "For educational analysis only. Screening classifications are not "
    "personalized investment advice or guarantees of future performance."
)
