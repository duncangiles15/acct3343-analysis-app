
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Stock Screening Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Stock Screening Analysis")
st.caption(
    "A quantitative screening tool using gross profitability, "
    "accruals, operating assets and liabilities, and Altman Z-Score."
)


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")

    # Standardize column names by removing surrounding whitespace.
    df.columns = df.columns.str.strip()

    # Use the main ticker column where available.
    if "Ticker" not in df.columns:
        if "Ticker_guess" in df.columns:
            df["Ticker"] = df["Ticker_guess"]
        else:
            raise ValueError(
                "The dataset needs a Ticker or Ticker_guess column."
            )

    if "Name" not in df.columns:
        df["Name"] = df["Ticker"]

    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip()
    df["Name"] = df["Name"].fillna(df["Ticker"]).astype(str).str.strip()

    # Parse market capitalization values such as $2.4B or 850M.
    def parse_market_cap(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.number)):
            return float(value)

        text = str(value).strip().replace(",", "").replace("$", "")

        if not text or text.lower() in {"nan", "none", "n/a", "na", "-"}:
            return np.nan

        multiplier = 1.0

        if text[-1:].upper() == "T":
            multiplier = 1_000_000_000_000
            text = text[:-1]
        elif text[-1:].upper() == "B":
            multiplier = 1_000_000_000
            text = text[:-1]
        elif text[-1:].upper() == "M":
            multiplier = 1_000_000
            text = text[:-1]
        elif text[-1:].upper() == "K":
            multiplier = 1_000
            text = text[:-1]

        try:
            return float(text) * multiplier
        except (TypeError, ValueError):
            return np.nan

    if "Market Cap" in df.columns:
        df["Market Cap"] = df["Market Cap"].apply(parse_market_cap)
    else:
        df["Market Cap"] = np.nan

    # Convert the metrics used by the screener to numeric values.
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

    # Remove rows without a usable ticker.
    df = df[df["Ticker"].ne("")].copy()

    # Keep the first row if a ticker appears more than once.
    df = df.drop_duplicates(subset=["Ticker"], keep="first")

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
        "The dataset is missing required screening columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# SIDEBAR FILTERS AND SCREENING SETTINGS
# ============================================================

st.sidebar.header("Screening Settings")

minimum_market_cap_millions = st.sidebar.number_input(
    "Minimum market cap ($ millions)",
    min_value=0,
    max_value=10_000_000,
    value=0,
    step=100,
    help="Set to 0 to include all companies with available market-cap data.",
)

st.sidebar.subheader("Gross Profitability")

gpa_top_pct = st.sidebar.slider(
    "GPA: top percentile for BUY signal",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
) / 100

gpa_bottom_pct = st.sidebar.slider(
    "GPA: bottom percentile for SELL signal",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
) / 100

st.sidebar.subheader("Accruals (STA)")

sta_bottom_pct = st.sidebar.slider(
    "STA: bottom percentile for BUY signal",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
) / 100

sta_top_pct = st.sidebar.slider(
    "STA: top percentile for SELL signal",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
) / 100

st.sidebar.subheader("Scaled Net Operating Assets (SNOA)")

snoa_bottom_pct = st.sidebar.slider(
    "SNOA: bottom percentile for BUY signal",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
) / 100

snoa_top_pct = st.sidebar.slider(
    "SNOA: top percentile for SELL signal",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
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

minimum_buy_signals = st.sidebar.selectbox(
    "Minimum BUY signals",
    options=[1, 2, 3],
    index=1,
)

minimum_sell_signals = st.sidebar.selectbox(
    "Minimum SELL signals",
    options=[1, 2, 3, 4],
    index=1,
)

if gpa_top_pct + gpa_bottom_pct > 1:
    st.sidebar.error("GPA top and bottom percentiles overlap.")
    st.stop()

if sta_top_pct + sta_bottom_pct > 1:
    st.sidebar.error("STA top and bottom percentiles overlap.")
    st.stop()

if snoa_top_pct + snoa_bottom_pct > 1:
    st.sidebar.error("SNOA top and bottom percentiles overlap.")
    st.stop()

if altman_safe_threshold <= altman_distress_threshold:
    st.sidebar.error(
        "The safe-zone threshold should be greater than the distress threshold."
    )
    st.stop()


# ============================================================
# FILTER DATASET
# ============================================================

df = raw_df.copy()

minimum_market_cap = minimum_market_cap_millions * 1_000_000

if minimum_market_cap > 0:
    df = df[
        df["Market Cap"].notna()
        & (df["Market Cap"] >= minimum_market_cap)
    ].copy()

if df.empty:
    st.warning(
        "No companies match the current market-cap filter. "
        "Reduce the minimum market cap."
    )
    st.stop()


# ============================================================
# PERCENTILE-RANKING HELPER
# ============================================================

def percentile_flag(series, tickers, percentile, select_lowest):
    """
    Flag a percentile of eligible companies.

    select_lowest=True:
        Select companies with the lowest metric values.

    select_lowest=False:
        Select companies with the highest metric values.

    Ticker is used as a deterministic tie-breaker.
    Missing metric values are not ranked.
    """
    flags = pd.Series(False, index=series.index, dtype=bool)

    eligible = series.notna()

    if not eligible.any():
        return flags

    ranked = pd.DataFrame(
        {
            "value": series.loc[eligible],
            "ticker": tickers.loc[eligible].astype(str),
        },
        index=series.loc[eligible].index,
    )

    ranked = ranked.sort_values(
        ["value", "ticker"],
        ascending=[select_lowest, True],
        kind="mergesort",
    )

    count = max(1, math.ceil(len(ranked) * percentile))
    selected_indices = ranked.head(count).index

    flags.loc[selected_indices] = True

    return flags


# ============================================================
# SCREENING SIGNALS
# ============================================================

# Gross Profitability / Total Assets:
# Higher is better.
df["GPA_BUY"] = percentile_flag(
    df["Gross_Profitability"],
    df["Ticker"],
    gpa_top_pct,
    select_lowest=False,
)

df["GPA_SELL"] = percentile_flag(
    df["Gross_Profitability"],
    df["Ticker"],
    gpa_bottom_pct,
    select_lowest=True,
)


# STA = (Net Income - Operating Cash Flow) / Total Assets:
# Lower is better for this screening strategy.
df["STA_BUY"] = percentile_flag(
    df["STA"],
    df["Ticker"],
    sta_bottom_pct,
    select_lowest=True,
)

df["STA_SELL"] = percentile_flag(
    df["STA"],
    df["Ticker"],
    sta_top_pct,
    select_lowest=False,
)


# SNOA:
# Lower is better, as requested.
# Use independent bottom and top percentile settings.
df["SNOA_BUY"] = percentile_flag(
    df["SNOA"],
    df["Ticker"],
    snoa_bottom_pct,
    select_lowest=True,
)

df["SNOA_SELL"] = percentile_flag(
    df["SNOA"],
    df["Ticker"],
    snoa_top_pct,
    select_lowest=False,
)


# Altman Z:
# Below the distress threshold contributes a SELL signal.
df["ALTMAN_SELL"] = (
    df["Altman_Z"].notna()
    & (df["Altman_Z"] < altman_distress_threshold)
)

df["ALTMAN_DISTRESS"] = df["ALTMAN_SELL"]

df["ALTMAN_SAFE"] = (
    df["Altman_Z"].notna()
    & (df["Altman_Z"] >= altman_safe_threshold)
)


# ============================================================
# SIGNAL COUNTS AND DATA AVAILABILITY
# ============================================================

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

# Require at least two available metrics to make a screening classification.
df["Enough_Data"] = df["Available_Core_Metrics"] >= 2


def classify_company(row):
    if not row["Enough_Data"]:
        return "REVIEW — Limited Data"

    buy_signals = int(row["Buy_Signals"])
    sell_signals = int(row["Sell_Signals"])

    if (
        buy_signals >= minimum_buy_signals
        and buy_signals > sell_signals
    ):
        return "Potential LONG"

    if (
        sell_signals >= minimum_sell_signals
        and sell_signals > buy_signals
    ):
        return "Potential SHORT"

    if buy_signals > 0 and sell_signals > 0:
        return "MIXED Signals"

    return "NEUTRAL"


df["Screening_Result"] = df.apply(classify_company, axis=1)


# ============================================================
# OVERVIEW METRICS
# ============================================================

total_companies = len(df)
long_count = int((df["Screening_Result"] == "Potential LONG").sum())
short_count = int((df["Screening_Result"] == "Potential SHORT").sum())
mixed_count = int((df["Screening_Result"] == "MIXED Signals").sum())
neutral_count = int((df["Screening_Result"] == "NEUTRAL").sum())
review_count = int(
    (df["Screening_Result"] == "REVIEW — Limited Data").sum()
)

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

metric1.metric("Companies Screened", f"{total_companies:,}")
metric2.metric("Potential LONG", f"{long_count:,}")
metric3.metric("Potential SHORT", f"{short_count:,}")
metric4.metric("Mixed Signals", f"{mixed_count:,}")
metric5.metric("Needs Review", f"{review_count:,}")

st.caption(
    "These are quantitative screening classifications, not personalized "
    "investment advice or guarantees of future performance."
)


# ============================================================
# TABLE FORMATTING HELPERS
# ============================================================

def display_columns(data):
    preferred = [
        "Rank",
        "Ticker",
        "Name",
        "Market Cap",
        "Price",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
        "Buy_Signals",
        "Sell_Signals",
        "Available_Core_Metrics",
        "Screening_Result",
    ]

    return [column for column in preferred if column in data.columns]


def format_market_cap(value):
    if pd.isna(value):
        return "N/A"

    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.2f}K"

    return f"${value:,.0f}"


def prepare_display_table(data):
    table = data[display_columns(data)].copy()

    if "Market Cap" in table.columns:
        table["Market Cap"] = table["Market Cap"].apply(format_market_cap)

    rename_map = {
        "Rank": "Universe Rank",
        "Gross_Profitability": "GPA",
        "Buy_Signals": "BUY Signals",
        "Sell_Signals": "SELL Signals",
        "Available_Core_Metrics": "Metrics Available",
        "Screening_Result": "Screening Result",
        "Altman_Z": "Altman Z",
    }

    return table.rename(columns=rename_map)


# ============================================================
# TOP 10 BUY AND SELL CANDIDATES
# ============================================================

st.header("Top 10 Screening Candidates")

buy_candidates = df[
    (df["Screening_Result"] == "Potential LONG")
].copy()

buy_candidates = buy_candidates.sort_values(
    by=[
        "Buy_Signals",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
        "Ticker",
    ],
    ascending=[
        False,
        False,
        True,
        True,
        False,
        True,
    ],
    na_position="last",
)

sell_candidates = df[
    (df["Screening_Result"] == "Potential SHORT")
].copy()

sell_candidates = sell_candidates.sort_values(
    by=[
        "Sell_Signals",
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
        "Ticker",
    ],
    ascending=[
        False,
        True,
        False,
        False,
        True,
        True,
    ],
    na_position="last",
)

buy_top10 = buy_candidates.head(10)
sell_top10 = sell_candidates.head(10)

buy_tab, sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

with buy_tab:
    st.subheader("Top 10 Potential LONG Candidates")
    st.caption(
        "Sorted by BUY signal count, then higher GPA, lower STA, "
        "lower SNOA, and higher Altman Z as tie-breakers."
    )

    if buy_top10.empty:
        st.info(
            "No companies currently meet the Potential LONG criteria. "
            "You can adjust the percentile settings or minimum BUY signals."
        )
    else:
        st.dataframe(
            prepare_display_table(buy_top10),
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
        "Sorted by SELL signal count, then lower GPA, higher STA, "
        "higher SNOA, and lower Altman Z as tie-breakers."
    )

    if sell_top10.empty:
        st.info(
            "No companies currently meet the Potential SHORT criteria. "
            "You can adjust the percentile settings or minimum SELL signals."
        )
    else:
        st.dataframe(
            prepare_display_table(sell_top10),
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
# VISUAL ANALYSIS
# ============================================================

st.header("Visual Analysis")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Companies by Screening Result")

    outcome_counts = (
        df["Screening_Result"]
        .value_counts()
        .rename_axis("Screening Result")
        .reset_index(name="Companies")
    )

    outcome_chart = px.bar(
        outcome_counts,
        x="Screening Result",
        y="Companies",
        title="Distribution of Screening Outcomes",
        text="Companies",
    )

    outcome_chart.update_layout(
        xaxis_title="Screening Result",
        yaxis_title="Number of Companies",
        showlegend=False,
    )

    outcome_chart.update_traces(textposition="outside")

    st.plotly_chart(outcome_chart, use_container_width=True)

with chart_right:
    st.subheader("GPA vs. STA")

    scatter_df = df[
        df["Gross_Profitability"].notna()
        & df["STA"].notna()
    ].copy()

    if scatter_df.empty:
        st.info("There are no companies with both GPA and STA available.")
    else:
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
            title="Gross Profitability vs. Accruals",
            labels={
                "Gross_Profitability": "Gross Profitability / Assets",
                "STA": "STA",
                "Screening_Result": "Screening Result",
            },
        )

        st.plotly_chart(scatter_chart, use_container_width=True)

st.caption(
    "GPA is generally interpreted as higher-is-better; STA and SNOA "
    "are treated as lower-is-better in this screening strategy. "
    "The scatter plot displays GPA and STA, while SNOA is available "
    "in each point's hover details."
)


# ============================================================
# COMPANY EXPLORER
# ============================================================

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
        search_df["Ticker"]
        + " — "
        + search_df["Name"]
    ).tolist()

    selected_company_label = st.selectbox(
        "Select a company",
        options=options,
    )

    selected_ticker = selected_company_label.split(" — ", 1)[0]

    selected_matches = df[df["Ticker"] == selected_ticker]

    if not selected_matches.empty:
        selected_row = selected_matches.iloc[0]

        st.subheader(
            f"{selected_row['Name']} ({selected_row['Ticker']})"
        )

        info1, info2, info3, info4 = st.columns(4)

        info1.metric(
            "Screening Result",
            selected_row["Screening_Result"],
        )

        info2.metric(
            "BUY Signals",
            int(selected_row["Buy_Signals"]),
        )

        info3.metric(
            "SELL Signals",
            int(selected_row["Sell_Signals"]),
        )

        info4.metric(
            "Metrics Available",
            f"{int(selected_row['Available_Core_Metrics'])} / 4",
        )

        detail1, detail2, detail3, detail4 = st.columns(4)

        detail1.metric(
            "Gross Profitability (GPA)",
            (
                f"{selected_row['Gross_Profitability']:.4f}"
                if pd.notna(selected_row["Gross_Profitability"])
                else "N/A"
            ),
        )

        detail2.metric(
            "STA",
            (
                f"{selected_row['STA']:.4f}"
                if pd.notna(selected_row["STA"])
                else "N/A"
            ),
        )

        detail3.metric(
            "SNOA",
            (
                f"{selected_row['SNOA']:.4f}"
                if pd.notna(selected_row["SNOA"])
                else "N/A"
            ),
        )

        detail4.metric(
            "Altman Z-Score",
            (
                f"{selected_row['Altman_Z']:.2f}"
                if pd.notna(selected_row["Altman_Z"])
                else "N/A"
            ),
        )

        st.markdown("#### Why did this company receive this result?")

        explanations = []

        # GPA explanation
        if pd.isna(selected_row["Gross_Profitability"]):
            explanations.append(
                "GPA is missing, so the company was not ranked on gross profitability."
            )
        elif selected_row["GPA_BUY"]:
            explanations.append(
                "GPA is within the selected top percentile and contributes a BUY signal."
            )
        elif selected_row["GPA_SELL"]:
            explanations.append(
                "GPA is within the selected bottom percentile and contributes a SELL signal."
            )
        else:
            explanations.append(
                "GPA is available but is outside the selected BUY and SELL percentile groups."
            )

        # STA explanation
        if pd.isna(selected_row["STA"]):
            explanations.append(
                "STA is missing, so the company was not ranked on STA."
            )
        elif selected_row["STA_BUY"]:
            explanations.append(
                "STA is within the selected bottom percentile and contributes a BUY signal."
            )
        elif selected_row["STA_SELL"]:
            explanations.append(
                "STA is within the selected top percentile and contributes a SELL signal."
            )
        else:
            explanations.append(
                "STA is available but is outside the selected BUY and SELL percentile groups."
            )

        # SNOA explanation
        if pd.isna(selected_row["SNOA"]):
            explanations.append(
                "SNOA is missing, so the company was not ranked on SNOA."
            )
        elif selected_row["SNOA_BUY"]:
            explanations.append(
                "SNOA is within the selected bottom percentile and contributes a BUY signal."
            )
        elif selected_row["SNOA_SELL"]:
            explanations.append(
                "SNOA is within the selected top percentile and contributes a SELL signal."
            )
        else:
            explanations.append(
                "SNOA is available but is outside the selected BUY and SELL percentile groups."
            )

        # Altman Z explanation
        if pd.isna(selected_row["Altman_Z"]):
            explanations.append(
                "Altman Z-Score is missing, so no Altman distress signal was assigned."
            )
        elif selected_row["ALTMAN_SELL"]:
            explanations.append(
                f"Altman Z-Score is below {altman_distress_threshold:.2f}, "
                "contributing a SELL signal under the selected distress rule."
            )
        elif selected_row["ALTMAN_SAFE"]:
            explanations.append(
                f"Altman Z-Score is at or above {altman_safe_threshold:.2f}, "
                "placing it in the selected safe zone. This does not automatically "
                "create a BUY signal."
            )
        else:
            explanations.append(
                "Altman Z-Score is between the selected distress and safe-zone thresholds."
            )

        for explanation in explanations:
            st.markdown(f"- {explanation}")

        st.markdown("#### Company data")

        company_data = selected_row[
            [
                column for column in [
                    "Ticker",
                    "Name",
                    "Rank",
                    "Market Cap",
                    "Price",
                    "Today",
                    "Price (30 days)",
                    "Gross_Profit",
                    "Total_Assets",
                    "Gross_Profitability",
                    "STA",
                    "SNOA",
                    "Altman_Z",
                    "X1_Working_Capital_Pct",
                    "X2_Retained_Earnings_Pct",
                    "X3_EBIT_Pct",
                    "X4_Market_Equity_to_Liabilities_Pct",
                    "X5_Sales_to_Assets",
                    "Available_Core_Metrics",
                    "Buy_Signals",
                    "Sell_Signals",
                    "Screening_Result",
                ]
                if column in selected_row.index
            ]
        ]

        st.dataframe(
            company_data.rename(
                {
                    "Gross_Profitability": "GPA",
                    "Altman_Z": "Altman Z-Score",
                    "Buy_Signals": "BUY Signals",
                    "Sell_Signals": "SELL Signals",
                    "Screening_Result": "Screening Result",
                    "Available_Core_Metrics": "Metrics Available",
                }
            ).to_frame(name="Value"),
            use_container_width=True,
        )


# ============================================================
# SCREENING TABLES
# ============================================================

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
    subset = df[df["Screening_Result"] == "Potential LONG"].copy()
    st.write(f"{len(subset):,} companies")
    st.dataframe(
        prepare_display_table(subset),
        use_container_width=True,
        hide_index=True,
    )

with short_tab:
    subset = df[df["Screening_Result"] == "Potential SHORT"].copy()
    st.write(f"{len(subset):,} companies")
    st.dataframe(
        prepare_display_table(subset),
        use_container_width=True,
        hide_index=True,
    )

with mixed_tab:
    subset = df[
        df["Screening_Result"].isin(
            ["MIXED Signals", "REVIEW — Limited Data"]
        )
    ].copy()

    st.write(f"{len(subset):,} companies")
    st.dataframe(
        prepare_display_table(subset),
        use_container_width=True,
        hide_index=True,
    )

with neutral_tab:
    subset = df[df["Screening_Result"] == "NEUTRAL"].copy()
    st.write(f"{len(subset):,} companies")
    st.dataframe(
        prepare_display_table(subset),
        use_container_width=True,
        hide_index=True,
    )

with all_tab:
    st.write(f"{len(df):,} companies")
    st.dataframe(
        prepare_display_table(df),
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
# METHODOLOGY AND TRANSPARENCY
# ============================================================

with st.expander("Methodology: How the Screener Works"):
    st.markdown(
        """
        ### 1. Gross Profitability (GPA)

        **Formula:** Gross Profit / Total Assets

        GPA measures gross profit relative to the assets used to generate it.
        This screener treats higher GPA as a positive signal.

        - Top selected percentile: BUY signal
        - Bottom selected percentile: SELL signal

        ### 2. Accruals (STA)

        **Formula:** (Net Income − Operating Cash Flow) / Total Assets

        STA compares accounting earnings with operating cash flow, scaled by
        total assets. This screening strategy treats lower STA values as
        more favorable.

        - Bottom selected percentile: BUY signal
        - Top selected percentile: SELL signal

        ### 3. Scaled Net Operating Assets (SNOA)

        The app uses the SNOA values already provided in `universe.csv`.
        Lower SNOA is treated as more favorable for this screening strategy.

        - Bottom selected percentile: BUY signal
        - Top selected percentile: SELL signal

        SNOA now directly contributes to the BUY and SELL signal counts.

        ### 4. Altman Z-Score

        The dataset's existing Altman Z-Score is used. For the manufacturing
        thresholds configured by default:

        - Below 1.81: distress-related SELL signal
        - At or above 2.99: safe-zone context
        - Between the thresholds: neither distress nor safe-zone classification

        The safe-zone classification does not automatically create a BUY signal.
        Altman Z-Score is a financial-distress indicator, not a direct measure
        of expected stock returns. Thresholds may not be appropriate for every
        industry or company type.

        ### 5. Percentile ranking

        Percentiles are calculated across the companies that remain after the
        market-cap filter. Missing metric values are excluded from that metric's
        ranking.

        A company can receive a BUY signal from GPA, STA, and SNOA, and a SELL
        signal from GPA, STA, SNOA, or Altman Z-Score. The screening result
        depends on the signal counts and minimum-signal settings.

        ### 6. Screening classifications

        - **Potential LONG:** enough BUY signals, with more BUY than SELL signals.
        - **Potential SHORT:** enough SELL signals, with more SELL than BUY signals.
        - **MIXED Signals:** at least one BUY and one SELL signal, without meeting
          the LONG or SHORT rules.
        - **NEUTRAL:** does not meet the LONG or SHORT rules and has no conflicting
          BUY/SELL signals.
        - **REVIEW — Limited Data:** fewer than two of the four core metrics are
          available.

        These classifications are screening outputs, not investment
        recommendations. The model does not account for every relevant factor,
        including valuation, liquidity, industry differences, recent news,
        or future business conditions.
        """
    )


with st.expander("Data Coverage and Limitations"):
    coverage_columns = {
        "GPA": "Gross_Profitability",
        "STA": "STA",
        "SNOA": "SNOA",
        "Altman Z-Score": "Altman_Z",
    }

    coverage_rows = []

    for label, column in coverage_columns.items():
        available = int(df[column].notna().sum())
        coverage_rows.append(
            {
                "Metric": label,
                "Available": available,
                "Missing": int(len(df) - available),
                "Coverage (%)": (
                    round(100 * available / len(df), 1)
                    if len(df)
                    else 0
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(coverage_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **Important limitations**

        - Financial metrics may come from different reporting dates.
        - Data availability differs across companies.
        - Percentile rankings are relative to the filtered universe, so a
          company's signal can change when the universe or filters change.
        - The app uses the SNOA values supplied in the dataset; the underlying
          source formula and accounting classifications should be verified.
        - A signal indicates how a company ranks under the chosen rules. It
          does not establish that the stock is mispriced or predict its return.
        - Altman Z-Score thresholds are context-dependent and may not be
          appropriate for non-manufacturing companies.
        """
    )


st.caption(
    "Quantitative screening tool for educational analysis. "
    "Always verify source data, metric definitions, and company-specific context."
)
