
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Manufacturing Stock Screener",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# 1. LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")

    df["Ticker"] = (
        df["Ticker"].astype("string").str.strip().str.upper()
    )

    numeric_cols = [
        "Gross_Profitability", "STA", "SNOA", "Altman_Z",
        "Gross_Profit", "Total_Assets"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def market_cap_number(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper().replace("$", "").replace(",", "")
        try:
            for suffix, multiplier in [
                ("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)
            ]:
                if s.endswith(suffix):
                    return float(s[:-1]) * multiplier
            return float(s)
        except ValueError:
            return np.nan

    df["Market_Cap_USD"] = df["Market Cap"].apply(market_cap_number)
    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load universe.csv: {e}")
    st.stop()

st.title("Manufacturing Stock Screening App")
st.write(
    "Adjust the filters and screening rules to identify potential long "
    "and short candidates. Every classification is linked to explicit rules."
)

st.info(
    "Current data includes GPA, STA, the existing SNOA calculation, and Altman Z. "
    "Book-to-market and the required scaled-NOA formula are not yet available, "
    "so they are disclosed as outstanding metrics rather than fabricated."
)

# ==================================================
# 2. USER CONTROLS
# ==================================================

st.sidebar.header("1. Universe filters")

countries = sorted(df["Country"].dropna().astype(str).unique())

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=countries
)

min_market_cap = st.sidebar.number_input(
    "Minimum market cap (USD millions)",
    min_value=0,
    value=0,
    step=100
)

st.sidebar.header("2. Percentile screening rules")

gpa_cutoff_pct = st.sidebar.slider(
    "GPA top/bottom percentile (%)",
    min_value=5, max_value=30, value=10, step=5,
    help="Top GPA percentile is favourable; bottom percentile is unfavourable."
)

sta_cutoff_pct = st.sidebar.slider(
    "STA top/bottom percentile (%)",
    min_value=5, max_value=30, value=10, step=5,
    help="Low STA is favourable; high STA is unfavourable."
)

st.sidebar.header("3. Altman Z criteria")

z_distress = st.sidebar.slider(
    "Distress threshold: Z below",
    min_value=0.0, max_value=3.0, value=1.81, step=0.01
)

z_safe = st.sidebar.slider(
    "Safe-zone threshold: Z above",
    min_value=1.81, max_value=5.0, value=2.99, step=0.01
)

if z_safe <= z_distress:
    st.sidebar.error("Safe-zone threshold must be above distress threshold.")

st.sidebar.header("4. Recommendation rules")

buy_signals_required = st.sidebar.slider(
    "Minimum BUY signals for a long candidate",
    min_value=1, max_value=2, value=2
)

sell_signals_required = st.sidebar.slider(
    "Minimum SELL signals for a short candidate",
    min_value=1, max_value=3, value=2
)

# ==================================================
# 3. APPLY UNIVERSE FILTERS
# ==================================================

screen = df[
    df["Country"].astype(str).isin(selected_countries)
    & (df["Market_Cap_USD"] >= min_market_cap * 1_000_000)
].copy()

# ==================================================
# 4. PERCENTILE SCREEN FUNCTION
# ==================================================

def percentile_flags(data, column, higher_is_better, cutoff_pct):
    """
    Rank only non-missing observations.
    Select floor(eligible firms * percentile) at each end.
    Ranks use method='min'. Ticker provides deterministic
    ordering when a cutoff splits tied values.
    """
    buy_flag = pd.Series(False, index=data.index)
    sell_flag = pd.Series(False, index=data.index)

    eligible = data.loc[
        data[column].notna(), ["Ticker", column]
    ].copy()

    n = len(eligible)
    k = int(np.floor(n * cutoff_pct / 100))

    if n == 0 or k == 0:
        return buy_flag, sell_flag, n, k

    # Keep the rank calculation explicit for methodology transparency.
    eligible["Rank"] = eligible[column].rank(
        method="min",
        ascending=not higher_is_better
    )

    best = eligible.sort_values(
        [column, "Ticker"],
        ascending=[not higher_is_better, True],
        kind="mergesort"
    )

    worst = eligible.sort_values(
        [column, "Ticker"],
        ascending=[higher_is_better, True],
        kind="mergesort"
    )

    best_tickers = set(best.head(k)["Ticker"])
    worst_tickers = set(worst.head(k)["Ticker"])

    # In very small universes, prevent the same ticker getting both flags.
    overlap = best_tickers.intersection(worst_tickers)
    best_tickers -= overlap
    worst_tickers -= overlap

    buy_flag.loc[data["Ticker"].isin(best_tickers)] = True
    sell_flag.loc[data["Ticker"].isin(worst_tickers)] = True

    return buy_flag, sell_flag, n, k


# ==================================================
# 5. CALCULATE SCREEN FLAGS
# ==================================================

screen["GPA_BUY"], screen["GPA_SELL"], gpa_n, gpa_k = percentile_flags(
    screen, "Gross_Profitability", True, gpa_cutoff_pct
)

screen["STA_BUY"], screen["STA_SELL"], sta_n, sta_k = percentile_flags(
    screen, "STA", False, sta_cutoff_pct
)

# Altman Z classification
screen["Altman_Zone"] = np.select(
    [
        screen["Altman_Z"].notna()
        & (screen["Altman_Z"] < z_distress),

        screen["Altman_Z"].notna()
        & (screen["Altman_Z"] >= z_distress)
        & (screen["Altman_Z"] <= z_safe),

        screen["Altman_Z"].notna()
        & (screen["Altman_Z"] > z_safe)
    ],
    ["Distress", "Grey zone", "Safe"],
    default="Missing"
)

screen["Altman_SELL"] = screen["Altman_Zone"].eq("Distress")

# GPA and STA are percentile signals.
# Altman Z distress is an additional SELL signal.
# Altman safe-zone status is shown separately, not treated as a
# guaranteed BUY signal.
screen["BUY_Signals"] = (
    screen["GPA_BUY"].astype(int)
    + screen["STA_BUY"].astype(int)
)

screen["SELL_Signals"] = (
    screen["GPA_SELL"].astype(int)
    + screen["STA_SELL"].astype(int)
    + screen["Altman_SELL"].astype(int)
)

screen["Core_Metrics_Available"] = screen[
    ["Gross_Profitability", "STA", "Altman_Z"]
].notna().sum(axis=1)

# A missing metric cannot create a flag.
# REVIEW is used when fewer than two core measures are available.
enough_data = screen["Core_Metrics_Available"] >= 2

long_rule = (
    enough_data
    & (screen["BUY_Signals"] >= buy_signals_required)
    & (screen["SELL_Signals"] < sell_signals_required)
)

short_rule = (
    enough_data
    & (screen["SELL_Signals"] >= sell_signals_required)
    & (screen["BUY_Signals"] < buy_signals_required)
)

screen["Decision"] = np.select(
    [
        ~enough_data,
        long_rule,
        short_rule,
        (screen["BUY_Signals"] > 0) | (screen["SELL_Signals"] > 0)
    ],
    ["REVIEW", "Potential LONG", "Potential SHORT", "MIXED"],
    default="NEUTRAL"
)

# ==================================================
# 6. SUMMARY
# ==================================================

st.header("Screening results")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Firms screened", len(screen))
m2.metric("Potential LONG", int((screen["Decision"] == "Potential LONG").sum()))
m3.metric("Potential SHORT", int((screen["Decision"] == "Potential SHORT").sum()))
m4.metric("Mixed signals", int((screen["Decision"] == "MIXED").sum()))
m5.metric("Needs review", int((screen["Decision"] == "REVIEW").sum()))

st.caption(
    f"GPA: {gpa_n} eligible, {gpa_k} selected at each end. "
    f"STA: {sta_n} eligible, {sta_k} selected at each end."
)

# ==================================================
# 7. SHOW THE RULES AND EACH FIRM'S SIGNALS
# ==================================================

st.header("How the screening rules work")

with st.expander("GPA — Gross Profitability", expanded=True):
    st.markdown(f"""
    **Formula:** Gross profit / total assets.

    - Potential BUY signal: top **{gpa_cutoff_pct}%** of firms with GPA available.
    - Potential SELL signal: bottom **{gpa_cutoff_pct}%** of firms with GPA available.
    - Missing GPA is excluded from this screen and cannot trigger a flag.
    - Eligible firms: **{gpa_n}**. Cutoff count: **{gpa_k}** per end.
    """)

with st.expander("STA — Standardized Total Accruals", expanded=True):
    st.markdown(f"""
    **Formula:** (net income − operating cash flow) / total assets.

    - Potential BUY signal: bottom **{sta_cutoff_pct}%** of eligible firms.
    - Potential SELL signal: top **{sta_cutoff_pct}%** of eligible firms.
    - Higher STA may indicate lower earnings quality; negative STA means
      earnings are more cash-backed under this measure.
    - Missing STA is excluded from this screen and cannot trigger a flag.
    - Eligible firms: **{sta_n}**. Cutoff count: **{sta_k}** per end.
    """)

with st.expander("Altman Z — Financial-distress screen", expanded=True):
    st.markdown(f"""
    **Formula:** 1.2·WC/TA + 1.4·RE/TA + 3.3·EBIT/TA
    + 0.6·MVE/TL + 1.0·Sales/TA.

    - **Distress:** Z below {z_distress:.2f}; contributes one SELL signal.
    - **Grey zone:** Z from {z_distress:.2f} through {z_safe:.2f}.
    - **Safe:** Z above {z_safe:.2f}.
    - The safe zone is not automatically a BUY recommendation.
    - These are traditional manufacturing-model thresholds; they are
      screening guidelines, not guarantees of default or safety.
    """)

with st.expander("Combined recommendation rules", expanded=True):
    st.markdown(f"""
    - **Potential LONG:** at least {buy_signals_required} BUY signals and
      fewer than {sell_signals_required} SELL signals.
    - **Potential SHORT:** at least {sell_signals_required} SELL signals and
      fewer than {buy_signals_required} BUY signals.
    - **MIXED:** at least one signal exists, but neither recommendation rule
      is satisfied.
    - **NEUTRAL:** no BUY or SELL flags.
    - **REVIEW:** fewer than two of GPA, STA and Altman Z are available.
    - All percentile cutoffs are recalculated on the filtered eligible firms.
    """)

st.warning(
    "B/M and the required scaled-NOA formula are not yet present in the data. "
    "The existing SNOA column uses an earlier operating-assets/operating-liabilities "
    "formula, so it is displayed for reference but excluded from the current "
    "recommendation score. Sector demeaning is also not implemented yet."
)

# ==================================================
# 8. BUY / SELL TABLES
# ==================================================

table_columns = [
    "Ticker", "Name", "Country", "Market Cap",
    "Gross_Profitability", "GPA_BUY", "GPA_SELL",
    "STA", "STA_BUY", "STA_SELL",
    "Altman_Z", "Altman_Zone", "Altman_SELL",
    "SNOA", "BUY_Signals", "SELL_Signals",
    "Core_Metrics_Available", "Decision"
]

table_columns = [c for c in table_columns if c in screen.columns]

st.header("Candidate companies")

tab_long, tab_short, tab_mixed, tab_all = st.tabs(
    ["Potential LONG", "Potential SHORT", "Mixed / Review", "All firms"]
)

with tab_long:
    long_df = screen[screen["Decision"] == "Potential LONG"]
    st.dataframe(
        long_df[table_columns],
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        "Download long candidates CSV",
        long_df.to_csv(index=False),
        "potential_long_candidates.csv",
        "text/csv"
    )

with tab_short:
    short_df = screen[screen["Decision"] == "Potential SHORT"]
    st.dataframe(
        short_df[table_columns],
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        "Download short candidates CSV",
        short_df.to_csv(index=False),
        "potential_short_candidates.csv",
        "text/csv"
    )

with tab_mixed:
    mixed_df = screen[
        screen["Decision"].isin(["MIXED", "REVIEW", "NEUTRAL"])
    ]
    st.dataframe(
        mixed_df[table_columns],
        use_container_width=True,
        hide_index=True
    )

with tab_all:
    st.dataframe(
        screen[table_columns],
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        "Download all screened firms CSV",
        screen.to_csv(index=False),
        "screened_firms.csv",
        "text/csv"
    )

# ==================================================
# 9. GRAPH
# ==================================================

st.header("Visual analysis")

st.write(
    "This plot compares gross profitability with earnings quality. "
    "Higher GPA means more gross profit per dollar of assets; lower STA "
    "means lower accruals under this formula. Point colours reflect the "
    "combined screening classification."
)

plot_df = screen.dropna(
    subset=["Gross_Profitability", "STA"]
).copy()

if not plot_df.empty:
    fig = px.scatter(
        plot_df,
        x="Gross_Profitability",
        y="STA",
        color="Decision",
        hover_name="Ticker",
        hover_data=[
            "Name", "Country", "Altman_Z", "Altman_Zone",
            "GPA_BUY", "GPA_SELL", "STA_BUY", "STA_SELL",
            "BUY_Signals", "SELL_Signals"
        ],
        title="Gross Profitability vs. Standardized Total Accruals"
    )

    fig.update_layout(
        xaxis_title="GPA = Gross Profit / Total Assets",
        yaxis_title="STA = (Net Income − Operating Cash Flow) / Total Assets",
        legend_title="Screen classification"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No firms have both GPA and STA available for the graph.")

# ==================================================
# 10. COMPANY DRILL-DOWN
# ==================================================

st.header("Company-level explanation")

tickers = sorted(screen["Ticker"].dropna().unique().tolist())

if tickers:
    chosen_ticker = st.selectbox("Choose a company", tickers)
    company = screen.loc[screen["Ticker"] == chosen_ticker].iloc[0]

    st.subheader(f"{company['Name']} ({chosen_ticker})")

    d1, d2, d3 = st.columns(3)
    d1.metric("Decision", company["Decision"])
    d2.metric("BUY signals", int(company["BUY_Signals"]))
    d3.metric("SELL signals", int(company["SELL_Signals"]))

    st.write("**Why this classification?**")

    explanations = []

    if company["GPA_BUY"]:
        explanations.append("GPA is in the favourable percentile.")
    if company["GPA_SELL"]:
        explanations.append("GPA is in the unfavourable percentile.")
    if company["STA_BUY"]:
        explanations.append("STA is in the favourable percentile.")
    if company["STA_SELL"]:
        explanations.append("STA is in the unfavourable percentile.")
    if company["Altman_SELL"]:
        explanations.append("Altman Z is in the distress zone.")

    if pd.isna(company["Gross_Profitability"]):
        explanations.append("GPA is missing; it cannot trigger a signal.")
    if pd.isna(company["STA"]):
        explanations.append("STA is missing; it cannot trigger a signal.")
    if pd.isna(company["Altman_Z"]):
        explanations.append("Altman Z is missing; its zone is unknown.")

    if explanations:
        for explanation in explanations:
            st.write(f"- {explanation}")
    else:
        st.write(
            "- No BUY or SELL flag was triggered by the currently implemented rules."
        )

    details = [
        "Ticker", "Name", "Country", "Market Cap",
        "Gross_Profitability", "STA", "SNOA",
        "Altman_Z", "Altman_Zone",
        "GP_Status", "STA_Status", "SNOA_Status", "Altman_Status",
        "GP_Fiscal_Date", "Fiscal_Date", "Altman_Fiscal_Date"
    ]
    details = [c for c in details if c in company.index]

    st.dataframe(
        company[details].to_frame("Value"),
        use_container_width=True
    )

# ==================================================
# 11. DATA COVERAGE
# ==================================================

st.header("Missing-data report")

coverage = []

for column, label in [
    ("Gross_Profitability", "GPA"),
    ("STA", "STA"),
    ("SNOA", "Existing SNOA (not required scaled NOA)"),
    ("Altman_Z", "Altman Z")
]:
    valid = int(screen[column].notna().sum())
    missing = len(screen) - valid
    coverage.append({
        "Metric": label,
        "Eligible firms": valid,
        "Missing firms": missing,
        "Coverage (%)": round(100 * valid / len(screen), 1)
        if len(screen) else np.nan
    })

st.dataframe(
    pd.DataFrame(coverage),
    use_container_width=True,
    hide_index=True
)

# ==================================================
# 12. USER CONCLUSIONS
# ==================================================

st.header("Analyst interpretation")

st.text_area(
    "Write your conclusions",
    placeholder=(
        "Which companies have favourable signals? Where do the metrics agree "
        "or disagree? What are the limitations of the data and screening rules?"
    ),
    height=180
)

with st.expander("Data source and methodological limitations"):
    st.markdown("""
    - **Data file:** `universe.csv`, prepared in Colab using `yfinance`.
    - The current universe contains the 851 firms in the prepared company list.
    - Missing accounting values are not silently replaced with zero.
    - Percentile counts use floor(eligible firms × selected percentile).
    - Metric ranks use `method='min'`; ticker provides deterministic ordering
      when a cutoff must select among tied values.
    - Book-to-market, the required scaled-NOA formula, sector demeaning,
      data-pull date, and an explicit exclusion log remain to be added.
    - Historical market equity can differ from the issuer's exact fiscal
      year-end market value.
    - The existing SNOA result is not assumed to equal the required
      scaled-NOA formula.
    """)

st.caption(
    "Educational screening tool only. A long/short label is a rule-based "
    "candidate classification, not a guarantee of performance."
)
