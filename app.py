
import math
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Manufacturing Stock Screener",
    page_icon="📊",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "universe.csv"


# ============================================================
# 2. DATA LOADING AND CLEANING
# ============================================================

@st.cache_data
def load_data(file_path: str, modified_time: float) -> pd.DataFrame:
    """Load and normalize the source dataset."""

    df = pd.read_csv(file_path)

    # Remove duplicate source columns.
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

    df["Display_Ticker"] = (
        df["Display_Ticker"].fillna("").astype(str).str.strip()
    )

    # Use company name as a fallback identifier if ticker data is absent.
    df["Name"] = df["Name"].fillna("Unknown company").astype(str)

    df["Display_Ticker"] = df["Display_Ticker"].where(
        df["Display_Ticker"].ne(""),
        df["Name"],
    )

    df["Market Cap"] = df["Market Cap"].fillna("").astype(str)

    # Convert market-cap strings such as $2.4T, $850B, and $45.2M
    # into numeric dollar values.
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

    # Deterministic secondary sorting key.
    df["Sort_Ticker"] = df["Display_Ticker"].astype(str).str.upper()

    return df


if not DATA_FILE.exists():
    st.error(
        "The file `universe.csv` was not found. "
        "Place it in the same repository directory as `app.py`."
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
# 3. METRIC SCORES AND DISPLAY HELPERS
# ============================================================

def add_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate favourable-direction percentile scores.

    Higher Altman Z is preferred.
    Lower STA is preferred.
    Lower SNOA is preferred.
    Higher gross profitability is preferred.

    The composite score requires all four metrics.
    """

    result = df.copy()

    result["Altman_Score"] = (
        result["Altman_Z"].rank(method="average", pct=True) * 100
    )

    result["STA_Score"] = (
        result["STA"].rank(
            method="average",
            pct=True,
            ascending=False,
        ) * 100
    )

    result["SNOA_Score"] = (
        result["SNOA"].rank(
            method="average",
            pct=True,
            ascending=False,
        ) * 100
    )

    result["GP_Score"] = (
        result["Gross_Profitability"].rank(
            method="average",
            pct=True,
        ) * 100
    )

    score_columns = [
        "Altman_Score",
        "STA_Score",
        "SNOA_Score",
        "GP_Score",
    ]

    metric_columns = [
        "Altman_Z",
        "STA",
        "SNOA",
        "Gross_Profitability",
    ]

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
    """Sort by the screening metric, using ticker as a tie-breaker."""

    return df.sort_values(
        by=[metric, "Sort_Ticker"],
        ascending=[ascending, True],
        kind="mergesort",
        na_position="last",
    ).copy()


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

    table = table.rename(
        columns={
            "Display_Ticker": "Ticker",
            "Name": "Company",
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
    )

    # Source ratios are stored as decimals. Convert only for display.
    for column in [
        "STA (%)",
        "SNOA (%)",
        "Gross Profitability (%)",
    ]:
        if column in table.columns:
            table[column] = table[column] * 100

    numeric_columns = [
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

    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(
                table[column], errors="coerce"
            ).round(2)

    # Final safeguard against duplicate display column names.
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
    The retained count is floor(valid observations × retention rate).
    """

    entering_count = len(entering_df)

    valid = entering_df[entering_df[metric].notna()].copy()
    missing_count = entering_count - len(valid)

    ordered = stable_sort(valid, metric, ascending=ascending)

    keep_count = math.floor(
        len(ordered) * retention_percent / 100
    )
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


# ============================================================
# 4. CUTOFF DETAILS — FIXED FOR BOTH THRESHOLD AND RANK FILTERS
# ============================================================

def show_cutoff_details(
    title: str,
    cutoff: dict,
    display_metric: str,
    unit: str = "",
):
    """
    Display cutoff information for both:
    - Altman Z's fixed numerical threshold.
    - STA, SNOA, and gross profitability's rank-based filters.

    The Altman Z dictionary does not contain valid_count,
    keep_count, or excluded_count. Those counts are derived from
    its survivor and excluded tables instead.
    """

    with st.expander(title):
        survivors = cutoff["survivors"]
        excluded = cutoff["excluded"]

        # Rank-based filters store these counts directly.
        if "valid_count" in cutoff:
            valid_count = cutoff["valid_count"]
            keep_count = cutoff["keep_count"]
            excluded_count = cutoff["excluded_count"]

        # Altman Z uses a fixed threshold and stores different fields.
        else:
            valid_count = len(survivors) + len(excluded)
            keep_count = len(survivors)
            excluded_count = len(excluded)

        st.write(
            f"**Valid observations:** {valid_count:,}  \n"
            f"**Retained:** {keep_count:,}  \n"
            f"**Excluded after screening:** {excluded_count:,}"
        )

        if "threshold" in cutoff:
            st.write(
                f"**Altman Z threshold:** {cutoff['threshold']:.2f}"
            )

            st.caption(
                "Companies below the selected Altman Z threshold are "
                "excluded. Companies with missing Altman Z values are "
                "reported separately in the funnel summary."
            )

        if valid_count == 0:
            st.info("No valid observations were available at this stage.")
            return

        if survivors.empty:
            st.warning(
                "No observations were retained at this stage. "
                "Review the current screening settings."
            )
            return

        if excluded.empty:
            st.info("All valid observations were retained.")
            return

        # For Altman Z, the closest excluded observation is the highest
        # excluded Z-score; the closest retained is the lowest passing
        # Z-score.
        if "threshold" in cutoff:
            excluded_row = excluded.sort_values(
                "Altman_Z",
                ascending=False,
                kind="mergesort",
            ).head(1)

            retained_row = survivors.sort_values(
                "Altman_Z",
                ascending=True,
                kind="mergesort",
            ).head(1)

        # For rank-based filters, the survivors and excluded rows are
        # already sorted in the order used for the retention decision.
        else:
            excluded_row = excluded.tail(1)
            retained_row = survivors.tail(1)

        excluded_value = float(
            excluded_row.iloc[0][cutoff["metric"]]
        )
        retained_value = float(
            retained_row.iloc[0][cutoff["metric"]]
        )

        if unit == "%":
            excluded_value *= 100
            retained_value *= 100

        left, right = st.columns(2)

        with left:
            st.markdown("**Closest excluded observation**")
            st.write(
                f"**{excluded_row.iloc[0]['Display_Ticker']}**"
            )
            st.write(
                f"{display_metric}: {excluded_value:,.2f}{unit}"
            )

        with right:
            st.markdown("**Closest retained observation**")
            st.write(
                f"**{retained_row.iloc[0]['Display_Ticker']}**"
            )
            st.write(
                f"{display_metric}: {retained_value:,.2f}{unit}"
            )

        st.caption(
            "For rank-based filters, ticker is the secondary sorting key "
            "when metric values tie at the cutoff. Altman Z uses a fixed "
            "numerical threshold instead."
        )


# ============================================================
# 5. CALCULATE PERCENTILE SCORES
# ============================================================

data = add_quality_scores(data)


# ============================================================
# 6. PAGE HEADER AND METHODOLOGY LINK
# ============================================================

st.title("Manufacturing Stock Screener")

st.markdown(
    """
    A quantitative screening dashboard for manufacturing companies.
    The funnel applies financial filters sequentially, while the
    composite score compares companies across four financial measures.
    """
)

st.markdown(
    "**Quick navigation:** [Methodology, formulas and screening rules ↓](#methodology)"
)

st.caption(
    "Data source: the repository's `universe.csv` file. "
    "This dashboard is a quantitative research tool, not investment advice."
)


# ============================================================
# 7. SIDEBAR: FUNNEL CONTROL
# ============================================================

st.sidebar.title("Funnel Control")

st.sidebar.markdown(
    """
    Adjust the settings to change which companies pass through the funnel.
    Each filter is applied to the companies that survived the preceding stage.
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
        "the financial filters. For example, 0.5 means $500 million."
    ),
)

altman_threshold = st.sidebar.number_input(
    "Minimum Altman Z-score",
    min_value=-20.0,
    max_value=50.0,
    value=1.81,
    step=0.01,
    help=(
        "Companies with Altman Z below this value are excluded. "
        "For manufacturing companies, 1.81 is a commonly cited "
        "distress-zone boundary."
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
        "Percentage of valid STA observations retained at this stage. "
        "Lower STA values are preferred by this screening method."
    ),
)

snoa_retention = st.sidebar.slider(
    "% kept after SNOA filter",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help=(
        "Percentage of valid SNOA observations retained at this stage. "
        "Lower SNOA values are preferred by this screening method."
    ),
)

gp_retention = st.sidebar.slider(
    "% kept after gross profitability filter",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help=(
        "Percentage of valid gross profitability observations retained. "
        "Higher gross profitability values are preferred."
    ),
)

st.sidebar.divider()

st.sidebar.caption(
    "Retention percentages apply to valid observations entering each stage. "
    "The retained count is rounded down to a whole company. Missing values "
    "are counted separately from companies excluded by the ranking cutoff."
)


# ============================================================
# 8. MARKET-CAP ELIGIBILITY
# ============================================================

minimum_market_cap = market_cap_min_billions * 1_000_000_000

eligible = data[
    data["Market_Cap_Value"].notna()
    & (data["Market_Cap_Value"] >= minimum_market_cap)
].copy()

eligible_count = len(eligible)


# ============================================================
# 9. SCREENING FUNNEL
# ============================================================

st.header("Screening Funnel")

st.markdown(
    """
    The screen begins with companies meeting the market-cap requirement.
    It then applies the Altman Z threshold, STA filter, SNOA filter, and
    gross profitability filter in that order.
    """
)

funnel_rows = []
cutoff_details = {}


# ------------------------------------------------------------
# Stage 1: Altman Z threshold
# ------------------------------------------------------------

stage_entering = len(eligible)

altman_valid = eligible[eligible["Altman_Z"].notna()].copy()
altman_missing = stage_entering - len(altman_valid)

altman_passed = altman_valid[
    altman_valid["Altman_Z"] >= altman_threshold
].copy()

altman_excluded = altman_valid[
    altman_valid["Altman_Z"] < altman_threshold
].copy()

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


# ------------------------------------------------------------
# Stage 2: STA
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Stage 3: SNOA
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Stage 4: Gross profitability
# ------------------------------------------------------------

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

final_survivors = final_survivors.sort_values(
    by=["Quality_Score", "Sort_Ticker"],
    ascending=[False, True],
    kind="mergesort",
)


# ============================================================
# 10. FUNNEL SUMMARY AND CHART
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
        f"stages**, from {eligible_count:,} companies meeting the market-cap "
        f"requirement."
    )
else:
    st.warning(
        "No companies passed all four stages with the current settings. "
        "Review the retention percentages and Altman Z threshold."
    )


# ============================================================
# 11. CUTOFF INSPECTION
# ============================================================

st.header("Cutoff Inspection")

st.markdown(
    """
    Review the companies closest to each filter's boundary. This shows
    the last excluded and last retained observations, along with the
    number of companies affected by the filter.
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
# 12. TOP 10 BUY CANDIDATES
# ============================================================

st.header("Top 10 BUY Candidates")

st.markdown(
    """
    These are the highest composite-scoring companies among the final
    funnel survivors with complete scores. The composite score measures
    relative financial characteristics; it is not a price target or a
    guarantee of future returns.
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
        "No BUY candidates are available because no companies passed "
        "the funnel with a complete composite score."
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
# 13. TOP 10 SELL CANDIDATES
# ============================================================

st.header("Top 10 SELL Candidates")

st.markdown(
    """
    These are the lowest composite-scoring companies among the market-cap
    eligible universe with all four metrics available. This list is not
    restricted to companies that passed the funnel. It is a relative
    comparison group, not a recommendation to short a stock.
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


# ============================================================
# ANIMATED BUY / SELL TICKER — COSMETIC ONLY
# Uses the existing candidates and composite scores.
# Does not change screening calculations or candidate ordering.
# ============================================================

def build_ticker_items(df, label, side):
    items = []

    for _, row in df.iterrows():
        ticker = html.escape(
            str(row.get("Display_Ticker", "")).strip()
        )
        company = html.escape(
            str(row.get("Name", "")).strip()
        )

        score = row.get("Quality_Score", np.nan)
        score_text = f"{score:.1f}" if pd.notna(score) else "N/A"

        items.append(
            f'<span class="stock-ticker-item {side}">'
            f'<span class="stock-ticker-label">{label}</span>'
            f'<strong>{ticker}</strong>'
            f'<span class="stock-ticker-company">{company}</span>'
            f'<span class="stock-ticker-score">Score {score_text}</span>'
            f'<span class="stock-ticker-divider">◆</span>'
            f'</span>'
        )

    return "".join(items)


buy_ticker_items = build_ticker_items(
    buy_candidates, "BUY", "buy"
)

sell_ticker_items = build_ticker_items(
    sell_candidates, "SELL", "sell"
)

ticker_group = buy_ticker_items + sell_ticker_items

if not ticker_group:
    ticker_group = (
        '<span class="stock-ticker-empty">'
        'No candidates available with current filters'
        '</span>'
    )

# Duplicate the content for a continuous scrolling loop.
ticker_content = (
    f'<div class="stock-ticker-group">{ticker_group}</div>'
    f'<div class="stock-ticker-group" aria-hidden="true">'
    f'{ticker_group}</div>'
)

components.html(
    f"""
    <script>
    (() => {{
        try {{
            const parentDoc = window.parent.document;

            // Remove previous ticker elements when Streamlit reruns.
            const oldBar = parentDoc.getElementById(
                "manufacturing-stock-ticker"
            );
            if (oldBar) oldBar.remove();

            const oldStyle = parentDoc.getElementById(
                "manufacturing-stock-ticker-style"
            );
            if (oldStyle) oldStyle.remove();

            const style = parentDoc.createElement("style");
            style.id = "manufacturing-stock-ticker-style";

            style.textContent = `
                #manufacturing-stock-ticker {{
                    position: fixed;
                    top: 3.25rem;
                    left: 0;
                    right: 0;
                    width: 100%;
                    height: 46px;
                    overflow: hidden;
                    display: flex;
                    align-items: center;
                    background: #101820;
                    border-top: 1px solid #293744;
                    border-bottom: 1px solid #293744;
                    z-index: 999999;
                    box-sizing: border-box;
                    font-family: Arial, sans-serif;
                }}

                #manufacturing-stock-ticker .stock-ticker-track {{
                    display: flex;
                    width: max-content;
                    flex-shrink: 0;
                    white-space: nowrap;
                    animation: manufacturing-stock-scroll 55s linear infinite;
                }}

                #manufacturing-stock-ticker:hover .stock-ticker-track {{
                    animation-play-state: paused;
                }}

                #manufacturing-stock-ticker .stock-ticker-group {{
                    display: flex;
                    flex-shrink: 0;
                    align-items: center;
                }}

                #manufacturing-stock-ticker .stock-ticker-item {{
                    display: inline-flex;
                    align-items: center;
                    gap: 9px;
                    padding-left: 18px;
                    color: #f1f5f9;
                    font-size: 12px;
                }}

                #manufacturing-stock-ticker .stock-ticker-label {{
                    font-weight: 800;
                    font-size: 10px;
                    letter-spacing: 0.8px;
                }}

                #manufacturing-stock-ticker .buy .stock-ticker-label,
                #manufacturing-stock-ticker .buy strong {{
                    color: #65d99a;
                }}

                #manufacturing-stock-ticker .sell .stock-ticker-label,
                #manufacturing-stock-ticker .sell strong {{
                    color: #ff8585;
                }}

                #manufacturing-stock-ticker .stock-ticker-company {{
                    color: #b4c0cc;
                }}

                #manufacturing-stock-ticker .stock-ticker-score {{
                    color: #d5dde5;
                    font-size: 11px;
                }}

                #manufacturing-stock-ticker .stock-ticker-divider {{
                    color: #617384;
                    padding: 0 14px 0 4px;
                    font-size: 9px;
                }}

                #manufacturing-stock-ticker .stock-ticker-empty {{
                    padding-left: 18px;
                    color: #b4c0cc;
                    font-size: 12px;
                }}

                @keyframes manufacturing-stock-scroll {{
                    from {{ transform: translateX(0); }}
                    to {{ transform: translateX(-50%); }}
                }}

                @media (prefers-reduced-motion: reduce) {{
                    #manufacturing-stock-ticker .stock-ticker-track {{
                        animation: none;
                    }}
                }}

                @media (max-width: 600px) {{
                    #manufacturing-stock-ticker {{
                        height: 42px;
                    }}

                    #manufacturing-stock-ticker .stock-ticker-item {{
                        gap: 6px;
                        padding-left: 12px;
                    }}
                }}
            `;

            parentDoc.head.appendChild(style);

            const bar = parentDoc.createElement("div");
            bar.id = "manufacturing-stock-ticker";
            bar.setAttribute(
                "aria-label",
                "Scrolling BUY and SELL candidate ticker"
            );

            bar.innerHTML = `
                <div class="stock-ticker-track">
                    {ticker_content}
                </div>
            `;

            parentDoc.body.appendChild(bar);

        }} catch (error) {{
            console.warn(
                "Stock ticker could not attach to page:",
                error
            );
        }}
    }})();
    </script>
    """,
    height=1,
    scrolling=False,
)


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
# 14. COMPOSITE QUALITY SCORE
# ============================================================

st.header("Composite Quality Score")

st.markdown(
    """
    The composite score is an equal-weight average of four percentile
    scores. Each component is oriented so that a higher score indicates
    a more favourable relative position.
    """
)

st.latex(
    r"\text{Composite Score} = "
    r"\frac{S_Z + S_{STA} + S_{SNOA} + S_{GP}}{4}"
)

st.markdown(
    """
    **Component interpretation**

    - **Altman Z percentile:** higher Altman Z receives a higher score.
    - **STA percentile:** lower STA receives a higher score.
    - **SNOA percentile:** lower SNOA receives a higher score.
    - **Gross profitability percentile:** higher gross profitability
      receives a higher score.
    - The composite score is calculated only when all four metrics are
      available.
    - Percentile scores are relative measures. They are not probabilities,
      expected returns, or absolute measures of investment quality.
    """
)


# ============================================================
# 15. INSPECT STAGE SURVIVORS
# ============================================================

st.header("Inspect Companies at Each Stage")

st.markdown(
    "Select a group to review the companies remaining at that point in the funnel."
)

stage_options = {
    "Market-cap eligible universe": eligible,
    "Passed Altman Z threshold": altman_passed,
    "Passed STA filter": sta_survivors,
    "Passed SNOA filter": snoa_survivors,
    "Passed gross profitability filter": final_survivors,
}

selected_stage = st.selectbox(
    "Select a group of companies",
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
# 16. DATA COVERAGE
# ============================================================

st.header("Data Coverage")

st.markdown(
    """
    Coverage shows how many companies have each financial metric available.
    Missing values are not replaced with zero; companies lacking a required
    metric are removed at the relevant screening stage.
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
                available_count / total_count * 100
                if total_count
                else np.nan
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
# 17. METHODOLOGY
# ============================================================

# This HTML anchor provides the destination for the methodology link
# near the top of the page.
st.markdown(
    '<div id="methodology"></div>',
    unsafe_allow_html=True,
)

st.header("Methodology")

st.markdown(
    """
    This section documents the financial formulas, screening sequence,
    ranking rules, composite score, and limitations used by the dashboard.
    """
)


# ------------------------------------------------------------
# 17.1 Universe and market capitalization
# ------------------------------------------------------------

with st.expander(
    "1. Universe and market capitalization",
    expanded=True,
):
    st.markdown(
        """
        The screen starts with the companies in `universe.csv`. The minimum
        market-cap control removes companies whose parsed market
        capitalization is below the selected value.

        Market capitalization is converted to numeric dollars for filtering.
        For example, `$850B` becomes 850 billion dollars and `$2.4T` becomes
        2.4 trillion dollars. Companies with missing or unparseable market
        capitalization cannot pass this initial eligibility step.
        """
    )


# ------------------------------------------------------------
# 17.2 Altman Z-score
# ------------------------------------------------------------

with st.expander("2. Altman Z-score", expanded=True):
    st.markdown(
        """
        The manufacturing version of the Altman Z-score combines five
        financial ratios. The screen uses the precomputed `Altman_Z`
        value supplied in the source dataset.
        """
    )

    st.latex(
        r"Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5"
    )

    st.markdown("The components are defined as:")

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
        r"X_4 = \frac{\text{Market Value of Equity}}"
        r"{\text{Total Liabilities}}"
    )

    st.latex(
        r"X_5 = \frac{\text{Sales}}{\text{Total Assets}}"
    )

    st.markdown(
        """
        **Commonly cited reference zones for the manufacturing model**

        - **Z below 1.81:** distress zone.
        - **Z from 1.81 to 2.99:** grey zone.
        - **Z above 2.99:** safe zone.

        These are commonly cited reference boundaries, not guarantees of
        bankruptcy or financial safety. The selected minimum is a screening
        parameter. At the default of 1.81, companies with a Z-score below
        1.81 are excluded.

        The app does not recalculate the Z-score from its component ratios.
        It uses the precomputed `Altman_Z` column in the dataset.
        """
    )


# ------------------------------------------------------------
# 17.3 STA
# ------------------------------------------------------------

with st.expander(
    "3. STA — total accruals scaled by assets",
    expanded=True,
):
    st.markdown(
        """
        STA is used as an accrual-quality measure. Under this screen,
        lower values are preferred. The calculation follows the variables
        and conventions represented in the source dataset.
        """
    )

    st.latex(
        r"STA = \frac{\text{Net Income} - \text{Operating Cash Flow}}"
        r"{\text{Total Assets}}"
    )

    st.markdown(
        """
        **Screening direction:** ascending STA, retaining the lowest
        valid values first.

        STA should not be interpreted in isolation. Its meaning can depend
        on the company's business model, accounting choices, and reporting
        period.
        """
    )


# ------------------------------------------------------------
# 17.4 SNOA
# ------------------------------------------------------------

with st.expander(
    "4. SNOA — scaled net operating assets",
    expanded=True,
):
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
        **Screening direction:** ascending SNOA, retaining the lowest
        valid values first.

        Operating-asset and operating-liability definitions can vary across
        data providers. This app uses the SNOA values already present in
        the source dataset.
        """
    )


# ------------------------------------------------------------
# 17.5 Gross profitability
# ------------------------------------------------------------

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
        **Screening direction:** descending gross profitability, retaining
        the highest valid values first.

        The app uses the precomputed `Gross_Profitability` field rather
        than recalculating it from the raw financial-statement columns.
        """
    )


# ------------------------------------------------------------
# 17.6 Funnel and retention percentages
# ------------------------------------------------------------

with st.expander(
    "6. Sequential funnel and retention percentages",
    expanded=True,
):
    st.markdown(
        """
        The screen applies the following stages in order:

        1. Minimum market capitalization.
        2. Altman Z-score threshold.
        3. STA rank-retention filter.
        4. SNOA rank-retention filter.
        5. Gross profitability rank-retention filter.

        Each stage operates only on the companies that survived the
        preceding stage. Therefore, changing an earlier filter can change
        the number and identity of companies entering later stages.
        """
    )

    st.markdown(
        "For rank-based filters, the target number retained is:"
    )

    st.latex(
        r"N_{\text{retained}} = "
        r"\left\lfloor N_{\text{valid}} \times "
        r"\frac{p}{100} \right\rfloor"
    )

    st.markdown(
        """
        Where:

        - \(NValid\) is the number of companies entering the
          stage with a non-missing value for the required metric.
        - \(p\) is the selected retention percentage.
        - The floor operation rounds down to the nearest whole company.

        Missing observations are counted separately from companies excluded
        by the rank cutoff. The retention percentage applies to valid
        observations, not the original universe size.

        When equal metric values fall across a retention boundary, ticker
        is used as a deterministic secondary sort key. This makes the
        selection reproducible, but companies with identical metric values
        can be split across the cutoff.
        """
    )


# ------------------------------------------------------------
# 17.7 Percentile and composite scores
# ------------------------------------------------------------

with st.expander(
    "7. Percentile scores and composite quality score",
    expanded=True,
):
    st.markdown(
        """
        The composite score compares companies on four percentile scores.
        Each component is oriented so that a higher score indicates a more
        favourable relative position.
        """
    )

    st.latex(
        r"\text{Composite Score} = "
        r"\frac{S_Z + S_{STA} + S_{SNOA} + S_{GP}}{4}"
    )

    st.markdown(
        """
        Where:

        - \(S_Z\) is the Altman Z percentile; higher Z is preferred.
        - \(S_{STA}\) is the STA percentile; lower STA is preferred.
        - \(S_{SNOA}\) is the SNOA percentile; lower SNOA is preferred.
        - \(S_{GP}\) is the gross profitability percentile; higher values
          are preferred.

        Each component is scaled from 0 to 100 and receives equal weight.
        The composite score is calculated only when all four underlying
        metrics are available.

        The percentile scores are relative to the dataset being ranked.
        They are not probabilities, expected returns, valuation targets,
        or absolute measures of investment quality.
        """
    )


# ------------------------------------------------------------
# 17.8 BUY and SELL lists
# ------------------------------------------------------------

with st.expander("8. BUY and SELL candidate lists", expanded=True):
    st.markdown(
        """
        **Top 10 BUY candidates:** the ten highest composite-scoring
        companies among the final funnel survivors with a complete score.

        **Top 10 SELL candidates:** the ten lowest composite-scoring
        companies among the market-cap eligible universe with a complete
        score. This group is not restricted to companies that passed the
        funnel.

        These labels describe the output of this quantitative screen.
        They do not constitute individualized investment advice and do
        not incorporate every factor relevant to an investment decision.
        """
    )


# ------------------------------------------------------------
# 17.9 Data coverage and limitations
# ------------------------------------------------------------

with st.expander("9. Data coverage and limitations", expanded=True):
    st.markdown(
        """
        - The app uses the local `universe.csv` file; it does not retrieve
          live market or financial-statement data.
        - Financial metrics are used as provided in the source file. The app
          does not independently audit company filings or provider formulas.
        - Missing values remain missing and are not replaced with zero.
        - Data providers can use different definitions, fiscal dates, and
          calculation conventions.
        - The manufacturing Altman Z model is not necessarily appropriate
          for every company, capital structure, or industry.
        - The funnel and composite score are screening tools, not substitutes
          for reviewing financial statements, valuation, debt, liquidity,
          competitive position, and company-specific risks.
        """
    )


# ============================================================
# 18. FOOTER
# ============================================================

st.caption(
    "Manufacturing Stock Screener • Quantitative screening only • "
    "Review underlying data and company filings before making "
    "investment decisions."
)
