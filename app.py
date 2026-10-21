
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Manufacturing Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("Manufacturing Stock Screening App")
st.caption(
    "An interactive educational screen using profitability, "
    "earnings quality and financial-distress measures."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("universe.csv")

    data["Ticker"] = (
        data["Ticker"].astype("string").str.strip().str.upper()
    )

    numeric_columns = [
        "Gross_Profitability",
        "STA",
        "SNOA",
        "Altman_Z",
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

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load universe.csv: {error}")
    st.stop()

required_columns = [
    "Ticker", "Name", "Country", "Market Cap",
    "Gross_Profitability", "STA", "SNOA", "Altman_Z"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def percentile_flags(data, column, higher_is_better, cutoff):
    """
    Select exactly floor(eligible firms * cutoff) firms
    at each end of a metric's distribution.

    Missing values are excluded from the ranking.
    Ticker breaks ties only when needed to fill a cutoff.
    Ranks use method='min'.
    """
    buy_flag = pd.Series(False, index=data.index)
    sell_flag = pd.Series(False, index=data.index)

    eligible = data.loc[
        data[column].notna(), ["Ticker", column]
    ].copy()

    n = len(eligible)
    k = int(np.floor(n * cutoff))

    if n == 0 or k == 0:
        return buy_flag, sell_flag, n, k

    eligible["_rank"] = eligible[column].rank(
        method="min",
        ascending=not higher_is_better
    )

    # Best end: descending if higher is better, ascending otherwise.
    best = eligible.sort_values(
        [column, "Ticker"],
        ascending=[not higher_is_better, True],
        kind="mergesort"
    )

    # Worst end uses the reverse metric direction.
    worst = eligible.sort_values(
        [column, "Ticker"],
        ascending=[higher_is_better, True],
        kind="mergesort"
    )

    best_tickers = set(best.head(k)["Ticker"])
    worst_tickers = set(worst.head(k)["Ticker"])

    # A firm should not be assigned both flags from the same metric.
    overlap = best_tickers & worst_tickers
    best_tickers -= overlap
    worst_tickers -= overlap

    buy_flag.loc[data["Ticker"].isin(best_tickers)] = True
    sell_flag.loc[data["Ticker"].isin(worst_tickers)] = True

    return buy_flag, sell_flag, n, k


def parse_market_cap(value):
    if pd.isna(value):
        return np.nan

    text = (
        str(value).strip().upper()
        .replace("$", "")
        .replace(",", "")
    )

    try:
        if text.endswith("T"):
            return float(text[:-1]) * 1e12
        if text.endswith("B"):
            return float(text[:-1]) * 1e9
        if text.endswith("M"):
            return float(text[:-1]) * 1e6
        if text.endswith("K"):
            return float(text[:-1]) * 1e3
        return float(text)
    except ValueError:
        return np.nan


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Screening settings")

cutoff_percent = st.sidebar.slider(
    "Percentile cutoff for GPA and STA",
    min_value=5,
    max_value=25,
    value=10,
    step=5,
    help="Select the most favourable and least favourable "
         "percentage of eligible firms for each metric."
)

cutoff = cutoff_percent / 100

countries = sorted(
    df["Country"].dropna().astype(str).unique()
)

selected_countries = st.sidebar.multiselect(
    "Country",
    countries,
    default=countries
)

df["Market_Cap_Number"] = df["Market Cap"].apply(
    parse_market_cap
)

market_cap_min = st.sidebar.number_input(
    "Minimum market cap (USD millions)",
    min_value=0,
    value=0,
    step=100
)

# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

screen = df[
    df["Country"].astype(str).isin(selected_countries)
    & (
        df["Market_Cap_Number"]
        >= market_cap_min * 1_000_000
    )
].copy()

# --------------------------------------------------
# PERCENTILE FLAGS
# --------------------------------------------------

screen["GPA_BUY"], screen["GPA_SELL"], gpa_n, gpa_k = (
    percentile_flags(
        screen,
        "Gross_Profitability",
        higher_is_better=True,
        cutoff=cutoff
    )
)

screen["STA_BUY"], screen["STA_SELL"], sta_n, sta_k = (
    percentile_flags(
        screen,
        "STA",
        higher_is_better=False,
        cutoff=cutoff
    )
)

# --------------------------------------------------
# ALTMAN Z CLASSIFICATION
# Traditional manufacturing model cutoffs
# --------------------------------------------------

screen["Altman_Zone"] = np.select(
    [
        screen["Altman_Z"].notna()
        & (screen["Altman_Z"] < 1.81),

        screen["Altman_Z"].notna()
        & (screen["Altman_Z"] >= 1.81)
        & (screen["Altman_Z"] <= 2.99),

        screen["Altman_Z"].notna()
        & (screen["Altman_Z"] > 2.99),
    ],
    ["Distress zone", "Grey zone", "Safe zone"],
    default="Missing"
)

screen["Altman_SELL"] = (
    screen["Altman_Zone"] == "Distress zone"
)

# --------------------------------------------------
# COMBINED SIGNALS
# --------------------------------------------------

# Altman is shown as a distress classification, not
# automatically treated as a BUY signal.
screen["BUY_Signals"] = (
    screen["GPA_BUY"].astype(int)
    + screen["STA_BUY"].astype(int)
)

screen["SELL_Signals"] = (
    screen["GPA_SELL"].astype(int)
    + screen["STA_SELL"].astype(int)
    + screen["Altman_SELL"].astype(int)
)

screen["Available_Core_Metrics"] = screen[
    ["Gross_Profitability", "STA", "Altman_Z"]
].notna().sum(axis=1)

screen["Decision"] = np.select(
    [
        screen["SELL_Signals"] >= 2,
        screen["BUY_Signals"] >= 2,
        screen["Available_Core_Metrics"] < 2,
    ],
    ["Potential SELL", "Potential BUY", "REVIEW"],
    default="MIXED"
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

st.subheader("Screening overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Companies screened", len(screen))
col2.metric(
    "Potential BUY",
    int((screen["Decision"] == "Potential BUY").sum())
)
col3.metric(
    "Potential SELL",
    int((screen["Decision"] == "Potential SELL").sum())
)
col4.metric(
    "Needs review",
    int((screen["Decision"] == "REVIEW").sum())
)

st.caption(
    f"Percentile cutoff: {cutoff_percent}% of eligible firms "
    f"per GPA and STA screen. GPA eligible: {gpa_n}; "
    f"STA eligible: {sta_n}."
)

st.warning(
    "These are screening classifications for educational analysis, "
    "not personalized investment recommendations. MIXED means the "
    "current signals do not satisfy the BUY or SELL rule."
)

# --------------------------------------------------
# ALTMAN Z ZONES
# --------------------------------------------------

st.subheader("Altman Z: financial-distress classification")

z1, z2, z3, z4 = st.columns(4)

z1.metric(
    "Distress zone",
    int((screen["Altman_Zone"] == "Distress zone").sum())
)
z2.metric(
    "Grey zone",
    int((screen["Altman_Zone"] == "Grey zone").sum())
)
z3.metric(
    "Safe zone",
    int((screen["Altman_Zone"] == "Safe zone").sum())
)
z4.metric(
    "Missing Z-score",
    int(screen["Altman_Z"].isna().sum())
)

st.markdown(
    "- **Distress zone:** Altman Z below 1.81\n"
    "- **Grey zone:** Altman Z from 1.81 through 2.99\n"
    "- **Safe zone:** Altman Z above 2.99"
)

st.caption(
    "These are traditional cutoffs for the original manufacturing "
    "model. A zone classification is not a guarantee of financial "
    "failure or safety."
)

# --------------------------------------------------
# MISSING DATA COVERAGE
# --------------------------------------------------

st.subheader("Data coverage and missing values")

coverage_rows = []

for column, label in [
    ("Gross_Profitability", "GPA"),
    ("STA", "STA"),
    ("SNOA", "Existing SNOA"),
    ("Altman_Z", "Altman Z"),
]:
    valid_count = int(screen[column].notna().sum())
    coverage_rows.append({
        "Metric": label,
        "Valid firms": valid_count,
        "Missing firms": len(screen) - valid_count,
        "Eligible coverage (%)": (
            round(100 * valid_count / len(screen), 1)
            if len(screen) else np.nan
        )
    })

st.dataframe(
    pd.DataFrame(coverage_rows),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# BUY AND SELL TABLES
# --------------------------------------------------

table_columns = [
    "Ticker", "Name", "Country", "Market Cap",
    "Gross_Profitability", "STA", "SNOA",
    "Altman_Z", "Altman_Zone",
    "BUY_Signals", "SELL_Signals", "Decision"
]

available_table_columns = [
    column for column in table_columns
    if column in screen.columns
]

st.subheader("Potential BUY candidates")

buy_table = screen[
    screen["Decision"] == "Potential BUY"
].copy()

st.dataframe(
    buy_table[available_table_columns].sort_values(
        "Gross_Profitability",
        ascending=False,
        na_position="last"
    ),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download BUY candidates",
    data=buy_table.to_csv(index=False),
    file_name="potential_buy_candidates.csv",
    mime="text/csv"
)

st.subheader("Potential SELL candidates")

sell_table = screen[
    screen["Decision"] == "Potential SELL"
].copy()

st.dataframe(
    sell_table[available_table_columns].sort_values(
        "SELL_Signals",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download SELL candidates",
    data=sell_table.to_csv(index=False),
    file_name="potential_sell_candidates.csv",
    mime="text/csv"
)

# --------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------

st.subheader("Gross profitability vs. STA")

st.write(
    "Each point is a company with both measures available. "
    "Higher GPA means more gross profit per dollar of assets. "
    "Lower STA indicates more cash-backed earnings under this measure."
)

chart_data = screen.dropna(
    subset=["Gross_Profitability", "STA"]
).copy()

if not chart_data.empty:
    fig = px.scatter(
        chart_data,
        x="Gross_Profitability",
        y="STA",
        color="Decision",
        hover_name="Ticker",
        hover_data=[
            "Name",
            "Country",
            "Altman_Z",
            "SNOA",
            "BUY_Signals",
            "SELL_Signals"
        ],
        title="Gross Profitability vs. Standardized Total Accruals"
    )

    fig.update_layout(
        xaxis_title="Gross Profitability (Gross Profit / Total Assets)",
        yaxis_title=(
            "STA ((Net Income − Operating Cash Flow) / Total Assets)"
        ),
        legend_title="Screen classification"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No firms have both GPA and STA available for this plot.")

# --------------------------------------------------
# COMPANY DETAIL
# --------------------------------------------------

st.subheader("Explore an individual company")

tickers = sorted(screen["Ticker"].dropna().unique().tolist())

if tickers:
    selected_ticker = st.selectbox("Select ticker", tickers)

    company = screen.loc[
        screen["Ticker"] == selected_ticker
    ].iloc[0]

    detail_columns = [
        "Name", "Ticker", "Country", "Market Cap",
        "Gross_Profitability", "STA", "SNOA", "Altman_Z",
        "Altman_Zone", "GPA_BUY", "GPA_SELL",
        "STA_BUY", "STA_SELL", "Altman_SELL",
        "BUY_Signals", "SELL_Signals", "Decision",
        "Altman_Status", "STA_Status", "SNOA_Status",
        "GP_Status", "Altman_Fiscal_Date",
        "Fiscal_Date", "GP_Fiscal_Date"
    ]

    present_details = [
        column for column in detail_columns
        if column in company.index
    ]

    st.dataframe(
        company[present_details].to_frame("Value"),
        use_container_width=True
    )

# --------------------------------------------------
# METHODOLOGY
# --------------------------------------------------

st.subheader("Screening rules and methodology")

with st.expander("Definitions and BUY/SELL rules", expanded=True):
    st.markdown("""
    **Data source:** `universe.csv`, assembled in Google Colab from
    Yahoo Finance data using `yfinance`. The current universe contains
    851 companies.

    **GPA — Gross Profitability**
    - Formula: gross profit / total assets.
    - Potential BUY signal: top 10% (or selected cutoff) among firms
      with non-missing GPA.
    - Potential SELL signal: bottom 10% among firms with non-missing GPA.

    **STA — Standardized Total Accruals**
    - Formula: (net income − operating cash flow) / total assets.
    - Potential BUY signal: bottom 10% of STA, where lower values are
      more favourable under the earnings-quality interpretation.
    - Potential SELL signal: top 10% of STA, where higher accruals can
      indicate lower earnings quality.

    **Altman Z**
    - Formula: 1.2·WC/TA + 1.4·RE/TA + 3.3·EBIT/TA
      + 0.6·MVE/TL + 1.0·Sales/TA.
    - Below 1.81: distress zone.
    - 1.81 to 2.99 inclusive: grey zone.
    - Above 2.99: safe zone.
    - The distress zone contributes one SELL signal. The safe zone
      is reported separately and does not independently create a BUY.

    **Combined classification**
    - Potential SELL: at least two SELL signals.
    - Potential BUY: at least two BUY signals.
    - REVIEW: fewer than two of GPA, STA and Altman Z are available.
    - MIXED: remaining cases that do not meet the BUY or SELL rule.

    **Percentile methodology**
    - Missing values are excluded from each metric's ranking.
    - The cutoff count is floor(eligible firms × cutoff percentage).
    - Ranks use method='min'. Ticker is used as a deterministic
      tie-breaker to fill the selected count.
    - Missing values are not replaced by zero.
    """)

with st.expander("Important limitations and outstanding metrics"):
    st.markdown("""
    - Book-to-market (B/M) should be book equity / market capitalization.
      It is not available in the current dataset and is not screened here.
    - The current SNOA column was generated using an earlier operating
      assets/operating liabilities formula. It is not assumed to equal
      the required scaled NOA formula.
    - Required scaled NOA is (total debt + equity − cash) /
      prior-year total assets. It must be calculated before it can be
      screened reliably.
    - Sector demeaning is not implemented because the current dataset
      does not include the prepared sector taxonomy.
    - The exact data-pull date and firm exclusion log should be recorded
      in the data-preparation notebook and added to this app.
    - Historical market equity may differ from market equity on the
      exact fiscal year-end.
    - These metrics are signals for screening and analysis, not
      guarantees of future returns or financial distress.
    """)

st.subheader("Your interpretation")

st.text_area(
    "Write your conclusions and limitations",
    placeholder=(
        "Describe the patterns you observe, the firms highlighted by "
        "the screens, and limitations caused by missing financial data."
    ),
    height=150
)

st.caption(
    "Educational analysis only. Screening flags are not personalized "
    "investment advice."
)
