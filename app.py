
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
    This dashboard screens manufacturing companies using financial health,
    earnings quality, operating asset quality, and gross profitability.

    The **sequential funnel** identifies companies that pass the selected
    criteria. The **Top 10 BUY and SELL lists** rank companies using a
    composite quality score. These are screening outputs, not investment
    recommendations.
    """
)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")

    # Ensure required columns exist.
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
        """Convert values such as $2.4T, $850B, and $45.2M to dollars."""
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        text = (
            str(value)
            .strip()
            .upper()
            .replace(",", "")
            .replace("$", "")
            .replace(" ", "")
        )

        match = re.fullmatch(r"([-+]?\d*\.?\d+)([TBMK]?)", text)

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

    # Normalize ticker columns and create one display ticker.
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
        "repository folder as app.py."
    )
    st.stop()

if data.empty:
    st.error("universe.csv contains no companies.")
    st.stop()


# ============================================================
# 3. SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("Screen Controls")
st.sidebar.caption(
    "Each stage filters the companies that survived the previous stage."
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
)

st.sidebar.subheader("1. Financial Health")

min_altman_z = st.sidebar.number_input(
    "Minimum Altman Z-Score",
    min_value=-10.0,
    max_value=20.0,
    value=1.81,
    step=0.1,
    help=(
        "The commonly cited manufacturing distress boundary is 1.81. "
        "This screen excludes companies below the selected value."
    ),
)

st.sidebar.subheader("2. Earnings Quality")

keep_lowest_sta = st.sidebar.slider(
    "Keep lowest STA percentile (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Lower STA is preferred by this screening rule.",
)

st.sidebar.subheader("3. Operating Asset Quality")

keep_lowest_snoa = st.sidebar.slider(
    "Keep lowest SNOA percentile (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Lower SNOA is preferred by this screening rule.",
)

st.sidebar.subheader("4. Gross Profitability")

keep_highest_gp = st.sidebar.slider(
    "Keep highest gross profitability percentile (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Higher gross profitability is preferred by this screening rule.",
)

if st.sidebar.button("Reset screening controls"):
    st.rerun()


# ============================================================
# 4. INITIAL ELIGIBLE UNIVERSE
# ============================================================

eligible = data[
    data["Market_Cap_Value"].notna()
    & (
        data["Market_Cap_Value"]
        >= min_market_cap * 1_000_000_000
    )
].copy()

if eligible.empty:
    st.warning(
        "No companies meet the minimum market-cap filter. "
        "Lower the threshold to continue."
    )
    st.stop()


# ============================================================
# 5. COMPOSITE QUALITY SCORE
# ============================================================

# Higher percentile scores always mean more favourable results:
# Altman Z: higher is better.
# STA: lower is better.
# SNOA: lower is better.
# Gross Profitability: higher is better.
#
# All four metrics are equally weighted. Companies must have all four
# metrics to receive a composite score.

METRICS = [
    "Altman_Z",
    "STA",
    "SNOA",
    "Gross_Profitability",
]


def add_quality_scores(df):
    result = df.copy()

    result["Altman_Pct"] = (
        result["Altman_Z"].rank(pct=True, method="average") * 100
    )

    result["STA_Pct"] = (
        1 - result["STA"].rank(pct=True, method="average")
    ) * 100

    result["SNOA_Pct"] = (
        1 - result["SNOA"].rank(pct=True, method="average")
    ) * 100

    result["GP_Pct"] = (
        result["Gross_Profitability"]
        .rank(pct=True, method="average") * 100
    )

    percentile_columns = [
        "Altman_Pct",
        "STA_Pct",
        "SNOA_Pct",
        "GP_Pct",
    ]

    complete = result[METRICS].notna().all(axis=1)

    result["Quality_Score"] = np.nan
    result.loc[complete, "Quality_Score"] = (
        result.loc[complete, percentile_columns].mean(axis=1)
    )

    return result


eligible = add_quality_scores(eligible)


# ============================================================
# 6. FIXED PERCENTILE CUTOFFS
# ============================================================

# Calculate cutoffs from the eligible universe once, rather than
# recalculating them as companies are removed from the funnel.

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
# 7. SEQUENTIAL SCREENING FUNNEL
# ============================================================

current = eligible.copy()
stage_records = []
stage_snapshots = {}


def apply_stage(stage_name, metric_name, rule_description, condition):
    global current

    before = len(current)
    values = current[metric_name]

    missing_mask = values.isna()
    passes = values.notna() & condition(values)

    missing_count = int(missing_mask.sum())
    failed_rule_count = int((values.notna() & ~passes).sum())

    current = current.loc[passes].copy()
    after = len(current)

    stage_records.append(
        {
            "Stage": stage_name,
            "Rule": rule_description,
            "Before": before,
            "Removed": before - after,
            "Missing data": missing_count,
            "Failed rule": failed_rule_count,
            "Remaining": after,
        }
    )

    stage_snapshots[stage_name] = current.copy()


apply_stage(
    "Financial Health — Altman Z",
    "Altman_Z",
    f"Altman Z ≥ {min_altman_z:.2f}",
    lambda s: s >= min_altman_z,
)

apply_stage(
    "Earnings Quality — STA",
    "STA",
    f"STA ≤ {sta_cutoff:.4f}",
    lambda s: s <= sta_cutoff,
)

apply_stage(
    "Operating Asset Quality — SNOA",
    "SNOA",
    f"SNOA ≤ {snoa_cutoff:.4f}",
    lambda s: s <= snoa_cutoff,
)

apply_stage(
    "Gross Profitability",
    "Gross_Profitability",
    f"Gross Profitability ≥ {gp_cutoff:.4f}",
    lambda s: s >= gp_cutoff,
)

final_candidates = current.copy()


# ============================================================
# 8. LIVE FUNNEL
# ============================================================

st.header("Live Screening Funnel")

st.markdown(
    """
    Each stage filters the companies that survived the previous stage.
    The funnel shows how many companies were removed and how many remain.
    """
)

funnel_rows = [
    {
        "Stage": "Starting eligible universe",
        "Rule": "Market cap meets minimum",
        "Before": len(eligible),
        "Removed": 0,
        "Missing data": 0,
        "Failed rule": 0,
        "Remaining": len(eligible),
    }
]

funnel_rows.extend(stage_records)
funnel_df = pd.DataFrame(funnel_rows)

col1, col2, col3 = st.columns(3)

col1.metric("Eligible companies", f"{len(eligible):,}")
col2.metric("Final funnel candidates", f"{len(final_candidates):,}")

survival_rate = (
    len(final_candidates) / len(eligible) * 100
    if len(eligible) else 0
)

col3.metric("Funnel survival rate", f"{survival_rate:.1f}%")

st.dataframe(
    funnel_df,
    use_container_width=True,
    hide_index=True,
)

fig = px.bar(
    funnel_df,
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
# 9. EXPLAIN EACH SCREENING STAGE
# ============================================================

st.header("What Each Screening Stage Means")

with st.expander("Stage 1 — Financial Health: Altman Z-Score", expanded=True):
    st.markdown(
        """
        **Purpose:** remove companies below the selected financial-distress
        boundary before assessing other characteristics.

        The manufacturing Altman Z-Score combines working capital,
        retained earnings, EBIT, market equity relative to liabilities,
        and sales relative to assets.

        A score below 1.81 is commonly associated with the distress zone
        in the original manufacturing model. A score above 2.99 is
        commonly associated with the safe zone. These classifications
        are not guarantees of solvency.

        This screen uses 1.81 as its default minimum. Passing this stage
        does not necessarily mean a company is in the safe zone.
        """
    )

with st.expander("Stage 2 — Earnings Quality: STA"):
    st.markdown(
        """
        **Formula:**

        STA = (Net Income − Operating Cash Flow) / Total Assets

        Lower STA is preferred by this screen because a lower accrual
        measure can indicate that reported earnings are more strongly
        supported by cash flow. Negative values can occur when operating
        cash flow exceeds net income.

        The cutoff is based on a percentile of the eligible universe.
        Missing STA values cannot pass this stage.
        """
    )

with st.expander("Stage 3 — Operating Asset Quality: SNOA"):
    st.markdown(
        """
        **Purpose:** favour companies with lower net operating assets
        relative to their asset base.

        Lower SNOA is preferred under this screening rule, but the
        relationship can vary by business model and accounting
        characteristics. It should not be interpreted in isolation.

        The cutoff is calculated from the initial eligible universe.
        """
    )

with st.expander("Stage 4 — Gross Profitability"):
    st.markdown(
        """
        **Formula:**

        Gross Profitability = Gross Profit / Total Assets

        Higher gross profitability means the company generates more
        gross profit relative to its assets. The screen keeps companies
        at or above the selected upper-tail cutoff.

        Gross profitability is a business-quality measure, not a
        valuation measure. A profitable company can still be overpriced.
        """
    )


# ============================================================
# 10. TOP 10 BUY AND SELL LISTS
# ============================================================

st.header("Top 10 BUY and SELL Candidates")

st.markdown(
    """
    - **Top 10 BUY:** highest composite scores among companies that passed
      every funnel stage.
    - **Top 10 SELL:** lowest composite scores among all companies in the
      eligible universe with all four metrics available.

    The SELL list is a relative low-quality ranking, not a short-sale
    recommendation. A low score does not establish that a stock is
    overvalued or will deliver negative returns.
    """
)

buy_pool = final_candidates[
    final_candidates["Quality_Score"].notna()
].copy()

top10_buy = buy_pool.sort_values(
    "Quality_Score",
    ascending=False,
).head(10).copy()

sell_pool = eligible[
    eligible["Quality_Score"].notna()
].copy()

top10_sell = sell_pool.sort_values(
    "Quality_Score",
    ascending=True,
).head(10).copy()


def prepare_rank_table(df):
    """
    Build a fresh display DataFrame from explicitly selected columns.
    This avoids duplicate column names in Streamlit/PyArrow.
    """
    columns = {
        "Display_Ticker": "Ticker",
        "Name": "Name",
        "Market Cap": "Market Cap",
        "Price": "Price",
        "Quality_Score": "Composite Quality Score",
        "Altman_Z": "Altman Z-Score",
        "STA": "STA (%)",
        "SNOA": "SNOA (%)",
        "Gross_Profitability": "Gross Profitability (%)",
    }

    available_columns = [
        original for original in columns if original in df.columns
    ]

    output = df[available_columns].copy()
    output = output.rename(columns=columns)

    for col in [
        "STA (%)",
        "SNOA (%)",
        "Gross Profitability (%)",
    ]:
        if col in output.columns:
            output[col] = output[col] * 100

    # Final safety check against duplicate display labels.
    output = output.loc[:, ~output.columns.duplicated()]

    return output


buy_table = prepare_rank_table(top10_buy)
sell_table = prepare_rank_table(top10_sell)

buy_col, sell_col = st.columns(2)

with buy_col:
    st.subheader("Top 10 BUY")
    st.caption(
        "Highest composite quality scores among funnel survivors."
    )

    if buy_table.empty:
        st.info(
            "No companies passed every funnel stage with all four metrics "
            "available. Review the cutoffs and coverage."
        )
    else:
        st.dataframe(
            buy_table,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Top 10 BUY (CSV)",
            data=top10_buy.to_csv(index=False).encode("utf-8"),
            file_name="top10_buy.csv",
            mime="text/csv",
            key="download_top10_buy",
        )

with sell_col:
    st.subheader("Top 10 SELL")
    st.caption(
        "Lowest composite quality scores among eligible companies."
    )

    if sell_table.empty:
        st.info(
            "Not enough companies have all four metrics to create "
            "the SELL ranking."
        )
    else:
        st.dataframe(
            sell_table,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Top 10 SELL (CSV)",
            data=top10_sell.to_csv(index=False).encode("utf-8"),
            file_name="top10_sell.csv",
            mime="text/csv",
            key="download_top10_sell",
        )


# ============================================================
# 11. FINAL FUNNEL CANDIDATES
# ============================================================

st.header("Final Funnel Candidates")

st.caption(
    "These companies passed every sequential stage. They are candidates "
    "for further research, not automatic buy recommendations."
)

if final_candidates.empty:
    st.warning(
        "No companies passed every stage. Review the selected cutoffs "
        "and missing-data counts."
    )
else:
    candidate_table = prepare_rank_table(
        final_candidates.sort_values(
            "Quality_Score",
            ascending=False,
            na_position="last",
        )
    )

    st.dataframe(
        candidate_table,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download all final candidates (CSV)",
        data=final_candidates.to_csv(index=False).encode("utf-8"),
        file_name="final_funnel_candidates.csv",
        mime="text/csv",
        key="download_final_candidates",
    )


# ============================================================
# 12. INSPECT COMPANIES AT EACH STAGE
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

stage_display = prepare_rank_table(
    stage_data.sort_values(
        "Quality_Score",
        ascending=False,
        na_position="last",
    )
)

st.dataframe(
    stage_display,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    f"Download survivors after {selected_stage}",
    data=stage_data.to_csv(index=False).encode("utf-8"),
    file_name="screen_stage_survivors.csv",
    mime="text/csv",
    key=f"download_stage_{selected_stage}",
)


# ============================================================
# 13. METHODOLOGY AND DATA COVERAGE
# ============================================================

st.header("Methodology and Data Coverage")

with st.expander("How the sequential screen works", expanded=True):
    st.markdown(
        """
        **Step 1 — Define the eligible universe**

        Companies must meet the minimum market capitalization.

        **Step 2 — Financial health**

        Exclude companies with Altman Z below the selected minimum.

        **Step 3 — Earnings quality**

        Retain companies at or below the selected STA percentile cutoff.

        **Step 4 — Operating asset quality**

        Retain companies at or below the selected SNOA percentile cutoff.

        **Step 5 — Gross profitability**

        Retain companies at or above the selected gross profitability
        percentile cutoff.

        **Step 6 — Rank candidates**

        The BUY list ranks funnel survivors by composite score. The SELL
        list ranks the lowest scores in the eligible universe. Both lists
        require all four metrics to be present.

        Percentile cutoffs are calculated from the initial eligible
        universe and remain fixed throughout the funnel.
        """
    )

with st.expander("How the composite quality score works"):
    st.markdown(
        """
        Each metric is converted to a percentile from 0 to 100. The
        direction is standardized so a higher score means a more
        favourable relative result under this methodology.

        | Metric | Preferred direction |
        |---|---|
        | Altman Z-Score | Higher |
        | STA | Lower |
        | SNOA | Lower |
        | Gross Profitability | Higher |

        The composite quality score is the arithmetic mean of these
        four percentiles, with equal weight for each metric.

        Companies missing any of the four metrics are excluded from
        composite rankings. This avoids comparing scores calculated
        from different numbers of factors.

        A high score means the company ranks favourably on these measures
        relative to the eligible universe. It does not mean a high
        probability of investment success or that the stock is undervalued.
        """
    )

with st.expander("Coverage and missing values"):
    coverage_rows = []

    for label, column in [
        ("Altman Z-Score", "Altman_Z"),
        ("STA", "STA"),
        ("SNOA", "SNOA"),
        ("Gross Profitability", "Gross_Profitability"),
    ]:
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
        - The screen does not estimate intrinsic value or directly test
          whether a stock is undervalued.
        - Altman Z is a model, not a guarantee of solvency.
        - Lower STA and SNOA and higher gross profitability are screening
          preferences, not universal rules for every manufacturer.
        - Percentile thresholds are configurable choices, not proven
          optimal parameters.
        - The order of the funnel affects which companies survive.
        - Missing or stale financial data can materially affect results.
        - The SELL list is a relative low-quality ranking, not a short-sale
          recommendation.
        - Results depend on the data in universe.csv.
        """
    )

st.caption(
    "Results are calculated from universe.csv using the selected screening "
    "settings. Review company filings and valuation before making decisions."
)
