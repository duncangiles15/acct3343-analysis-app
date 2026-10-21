
import re
import math
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

The funnel shows how companies progress through each screen. The final
rankings are relative screening results, not automatic buy or sell advice.
"""
)


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")

    # Remove duplicate source columns to prevent PyArrow errors.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required = [
        "Name", "Market Cap", "Ticker", "Ticker_guess",
        "Altman_Z", "STA", "SNOA", "Gross_Profitability",
    ]

    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    numeric_columns = [
        "Rank", "Price", "Today", "Price (30 days)",
        "Altman_Z", "STA", "SNOA", "Gross_Profitability",
        "Gross_Profit", "Total_Assets",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def parse_market_cap(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.number)):
            return float(value)

        text = (
            str(value).strip().upper()
            .replace(",", "").replace("$", "").replace(" ", "")
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

    for col in ["Ticker", "Ticker_guess"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["Display_Ticker"] = df["Ticker"]
    missing_ticker = df["Display_Ticker"].eq("")
    df.loc[missing_ticker, "Display_Ticker"] = df.loc[
        missing_ticker, "Ticker_guess"
    ]

    no_ticker = df["Display_Ticker"].eq("")
    df.loc[no_ticker, "Display_Ticker"] = [
        f"UNKNOWN-{i}" for i in df.index[no_ticker]
    ]

    df["Name"] = df["Name"].fillna("Unknown company").astype(str)
    return df


try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "universe.csv was not found. Place it in the same repository "
        "folder as app.py."
    )
    st.stop()

if data.empty:
    st.error("universe.csv contains no company records.")
    st.stop()


# ============================================================
# 3. SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("Screen Controls")
st.sidebar.caption("Each stage screens the survivors of the previous stage.")

max_cap = data["Market_Cap_Value"].max() / 1_000_000_000
if pd.isna(max_cap) or max_cap <= 0:
    max_cap = 100.0

market_cap_max = max(100.0, float(math.ceil(max_cap / 100) * 100))

min_market_cap_b = st.sidebar.number_input(
    "Minimum market cap ($ billions)",
    min_value=0.0,
    max_value=market_cap_max,
    value=0.0,
    step=1.0,
)

st.sidebar.subheader("1 · Financial Health")

min_altman_z = st.sidebar.number_input(
    "Minimum Altman Z-Score",
    min_value=-10.0,
    max_value=20.0,
    value=1.81,
    step=0.1,
    help=(
        "1.81 is a commonly cited manufacturing distress boundary. "
        "Passing this threshold does not guarantee financial safety."
    ),
)

st.sidebar.subheader("2 · Earnings Quality")

sta_retain_pct = st.sidebar.slider(
    "Retain best STA observations (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Lower STA values are preferred.",
)

st.sidebar.subheader("3 · Operating Asset Quality")

snoa_retain_pct = st.sidebar.slider(
    "Retain best SNOA observations (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Lower SNOA values are preferred.",
)

st.sidebar.subheader("4 · Gross Profitability")

gp_retain_pct = st.sidebar.slider(
    "Retain best gross profitability observations (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Higher gross profitability values are preferred.",
)


# ============================================================
# 4. INITIAL ELIGIBLE UNIVERSE
# ============================================================

eligible = data.loc[
    data["Market_Cap_Value"].notna()
    & (
        data["Market_Cap_Value"]
        >= min_market_cap_b * 1_000_000_000
    )
].copy()

if eligible.empty:
    st.warning("No companies meet the selected market-cap minimum.")
    st.stop()

# Ticker provides a deterministic tie-breaker for cutoff boundaries.
eligible["_sort_ticker"] = (
    eligible["Display_Ticker"].astype(str).str.upper()
)


# ============================================================
# 5. COMPOSITE QUALITY SCORE
# ============================================================

def add_quality_scores(df):
    out = df.copy()

    # Higher scores should always indicate a more favourable relative result.
    out["Altman_Pct"] = (
        out["Altman_Z"].rank(pct=True, method="average") * 100
    )

    out["STA_Pct"] = (
        1 - out["STA"].rank(pct=True, method="average")
    ) * 100

    out["SNOA_Pct"] = (
        1 - out["SNOA"].rank(pct=True, method="average")
    ) * 100

    out["GP_Pct"] = (
        out["Gross_Profitability"].rank(pct=True, method="average") * 100
    )

    raw_metrics = [
        "Altman_Z", "STA", "SNOA", "Gross_Profitability"
    ]
    percentile_metrics = [
        "Altman_Pct", "STA_Pct", "SNOA_Pct", "GP_Pct"
    ]

    complete = out[raw_metrics].notna().all(axis=1)

    out["Quality_Score"] = np.nan
    out.loc[complete, "Quality_Score"] = out.loc[
        complete, percentile_metrics
    ].mean(axis=1)

    return out


eligible = add_quality_scores(eligible)


# ============================================================
# 6. DISPLAY TABLE HELPER
# ============================================================

def prepare_display_table(df):
    """Construct a fresh table with unique column names."""
    if df is None or df.empty:
        return pd.DataFrame()

    source = df.loc[:, ~df.columns.duplicated()].copy()

    rename_map = {
        "Display_Ticker": "Ticker",
        "Name": "Company",
        "Market Cap": "Market Cap",
        "Price": "Price",
        "Quality_Score": "Composite Quality Score",
        "Altman_Z": "Altman Z-Score",
        "STA": "STA (%)",
        "SNOA": "SNOA (%)",
        "Gross_Profitability": "Gross Profitability (%)",
    }

    available = [col for col in rename_map if col in source.columns]
    output = source[available].copy().rename(columns=rename_map)

    # Convert ratio decimals to percentages for display.
    for col in ["STA (%)", "SNOA (%)", "Gross Profitability (%)"]:
        if col in output.columns:
            output[col] = output[col] * 100

    # Final protection against duplicate labels.
    output = output.loc[:, ~output.columns.duplicated()].copy()

    for col in ["Composite Quality Score", "Altman Z-Score"]:
        if col in output.columns:
            output[col] = output[col].round(3)

    for col in ["STA (%)", "SNOA (%)", "Gross Profitability (%)"]:
        if col in output.columns:
            output[col] = output[col].round(2)

    return output


def stable_sort(df, metric, ascending):
    """Sort by metric, then ticker for deterministic results."""
    return df.sort_values(
        [metric, "_sort_ticker"],
        ascending=[ascending, True],
        na_position="last",
        kind="mergesort",
    )


# ============================================================
# 7. SEQUENTIAL FUNNEL
# ============================================================

stage_records = []
stage_snapshots = {}
cutoff_details = {}


def record_stage(name, entering, missing, excluded, remaining, rule):
    stage_records.append({
        "Stage": name,
        "Rule": rule,
        "Entering": int(entering),
        "Missing required metrics": int(missing),
        "Excluded by cutoff": int(excluded),
        "Remaining": int(remaining),
    })


# ------------------------------------------------------------
# Stage 1: Altman Z threshold
# ------------------------------------------------------------

current = eligible.copy()
stage_name = "Financial Health — Altman Z"

entering_count = len(current)
missing_mask = current["Altman_Z"].isna()
missing_count = int(missing_mask.sum())

valid = current.loc[~missing_mask].copy()
ordered = stable_sort(valid, "Altman_Z", ascending=False)

passed = ordered.loc[ordered["Altman_Z"] >= min_altman_z].copy()
excluded = ordered.loc[ordered["Altman_Z"] < min_altman_z].copy()

cutoff_details[stage_name] = {
    "rule": f"Altman Z ≥ {min_altman_z:.2f}",
    "method": "Explicit metric-value threshold",
    "excluded": excluded.tail(1),
    "retained": passed.tail(1),
    "valid_count": len(valid),
    "target_count": len(passed),
    "excluded_count": len(excluded),
}

record_stage(
    stage_name,
    entering_count,
    missing_count,
    len(excluded),
    len(passed),
    f"Altman Z ≥ {min_altman_z:.2f}",
)

current = passed.copy()
stage_snapshots[stage_name] = current.copy()


# ------------------------------------------------------------
# Stages 2–4: rank-based retention
# ------------------------------------------------------------

def rank_retention_stage(
    entering_df,
    stage_name,
    metric,
    retain_pct,
    ascending,
    metric_label,
):
    entering_count = len(entering_df)

    missing_mask = entering_df[metric].isna()
    missing_count = int(missing_mask.sum())

    valid = entering_df.loc[~missing_mask].copy()
    ordered = stable_sort(valid, metric, ascending=ascending)

    # Exact target count from the valid observations entering this stage.
    target_count = int(math.floor(len(ordered) * retain_pct / 100))
    target_count = max(0, min(target_count, len(ordered)))

    passed = ordered.iloc[:target_count].copy()
    excluded = ordered.iloc[target_count:].copy()

    direction = "lowest" if ascending else "highest"

    rule = (
        f"Retain floor(valid observations × {retain_pct}%), "
        f"{direction} {metric_label} first"
    )

    cutoff_details[stage_name] = {
        "rule": rule,
        "method": "Rank-based; ticker breaks boundary ties",
        "excluded": excluded.tail(1),
        "retained": passed.tail(1),
        "valid_count": len(ordered),
        "target_count": target_count,
        "excluded_count": len(excluded),
        "ordered_data": ordered,
    }

    record_stage(
        stage_name,
        entering_count,
        missing_count,
        len(excluded),
        len(passed),
        rule,
    )

    stage_snapshots[stage_name] = passed.copy()
    return passed


current = rank_retention_stage(
    current,
    "Earnings Quality — STA",
    "STA",
    sta_retain_pct,
    ascending=True,
    metric_label="STA",
)

current = rank_retention_stage(
    current,
    "Operating Asset Quality — SNOA",
    "SNOA",
    snoa_retain_pct,
    ascending=True,
    metric_label="SNOA",
)

current = rank_retention_stage(
    current,
    "Gross Profitability",
    "Gross_Profitability",
    gp_retain_pct,
    ascending=False,
    metric_label="gross profitability",
)

final_candidates = current.copy()


# ============================================================
# 8. LIVE FUNNEL OVERVIEW
# ============================================================

st.header("Live Screening Funnel")
st.write(
    "Each stage filters the companies that survived the preceding stage. "
    "Missing data is counted separately from cutoff exclusions."
)

retained_pct = (
    len(final_candidates) / len(eligible) * 100
    if len(eligible) else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Eligible companies", f"{len(eligible):,}")
col2.metric("Final candidates", f"{len(final_candidates):,}")
col3.metric("Retained", f"{retained_pct:.1f}%")

funnel_df = pd.DataFrame([
    {
        "Stage": "Eligible universe",
        "Rule": "Meets minimum market cap",
        "Entering": len(eligible),
        "Missing required metrics": 0,
        "Excluded by cutoff": 0,
        "Remaining": len(eligible),
    },
    *stage_records,
])

st.dataframe(funnel_df, use_container_width=True, hide_index=True)

fig = px.bar(
    funnel_df,
    x="Stage",
    y="Remaining",
    text="Remaining",
    title="Companies Remaining After Each Stage",
)
fig.update_layout(
    xaxis_title="Screening stage",
    yaxis_title="Companies remaining",
    xaxis_tickangle=-25,
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 9. CUTOFF INSPECTION
# ============================================================

st.header("Cutoff Inspection")
st.write(
    "Review the last excluded and first retained observations at each "
    "stage. For rank-based screens, the target count is determined from "
    "valid observations entering that stage."
)

selected_cutoff = st.selectbox(
    "Choose a stage to inspect",
    list(cutoff_details.keys()),
)

detail = cutoff_details[selected_cutoff]
st.caption(f"Rule: {detail['rule']}")
st.caption(f"Method: {detail['method']}")

left, right = st.columns(2)

with left:
    st.subheader("Last excluded")
    if detail["excluded"].empty:
        st.info("No observations were excluded at this boundary.")
    else:
        st.dataframe(
            prepare_display_table(detail["excluded"]),
            use_container_width=True,
            hide_index=True,
        )

with right:
    st.subheader("First retained")
    if detail["retained"].empty:
        st.info("No observations were retained at this boundary.")
    else:
        st.dataframe(
            prepare_display_table(detail["retained"]),
            use_container_width=True,
            hide_index=True,
        )

a, b, c = st.columns(3)
a.metric("Valid observations", f"{detail['valid_count']:,}")
b.metric("Target survivors", f"{detail['target_count']:,}")
c.metric("Excluded by cutoff", f"{detail['excluded_count']:,}")


# ============================================================
# 10. TOP 10 BUY AND SELL
# ============================================================

st.header("Top 10 BUY and SELL Candidates")
st.markdown(
    """
- **Top 10 BUY:** highest composite quality scores among companies that
  passed every funnel stage.
- **Top 10 SELL:** lowest composite quality scores among the entire
  eligible universe with all four metrics available.

The SELL list is a relative low-quality ranking, not a short-sale
recommendation. Neither list establishes whether a stock is undervalued.
"""
)

buy_pool = final_candidates.loc[
    final_candidates["Quality_Score"].notna()
].copy()

top10_buy = buy_pool.sort_values(
    ["Quality_Score", "_sort_ticker"],
    ascending=[False, True],
    kind="mergesort",
).head(10).copy()

sell_pool = eligible.loc[
    eligible["Quality_Score"].notna()
].copy()

top10_sell = sell_pool.sort_values(
    ["Quality_Score", "_sort_ticker"],
    ascending=[True, True],
    kind="mergesort",
).head(10).copy()

buy_col, sell_col = st.columns(2)

with buy_col:
    st.subheader("Top 10 BUY")

    if top10_buy.empty:
        st.info(
            "No companies passed every funnel stage with all four "
            "metrics available."
        )
    else:
        st.dataframe(
            prepare_display_table(top10_buy),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Top 10 BUY CSV",
            data=top10_buy.drop(
                columns=["_sort_ticker"], errors="ignore"
            ).to_csv(index=False).encode("utf-8"),
            file_name="top10_buy.csv",
            mime="text/csv",
            key="download_top10_buy",
        )

with sell_col:
    st.subheader("Top 10 SELL")

    if top10_sell.empty:
        st.info(
            "Not enough companies have all four metrics for a composite ranking."
        )
    else:
        st.dataframe(
            prepare_display_table(top10_sell),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Top 10 SELL CSV",
            data=top10_sell.drop(
                columns=["_sort_ticker"], errors="ignore"
            ).to_csv(index=False).encode("utf-8"),
            file_name="top10_sell.csv",
            mime="text/csv",
            key="download_top10_sell",
        )


# ============================================================
# 11. FINAL CANDIDATES
# ============================================================

st.header("Final Funnel Candidates")
st.caption(
    "These companies passed every selected stage and are candidates for "
    "further research, not automatic buy recommendations."
)

if final_candidates.empty:
    st.warning(
        "No companies survived every stage. Review the settings and "
        "missing-data counts."
    )
else:
    ranked_final = final_candidates.sort_values(
        ["Quality_Score", "_sort_ticker"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    )

    st.dataframe(
        prepare_display_table(ranked_final),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download final candidates CSV",
        data=final_candidates.drop(
            columns=["_sort_ticker"], errors="ignore"
        ).to_csv(index=False).encode("utf-8"),
        file_name="final_funnel_candidates.csv",
        mime="text/csv",
        key="download_final_candidates",
    )


# ============================================================
# 12. INSPECT SURVIVORS BY STAGE
# ============================================================

st.header("Inspect Stage Survivors")

selected_stage = st.selectbox(
    "Choose a screening stage",
    list(stage_snapshots.keys()),
    key="survivor_stage",
)

survivors = stage_snapshots[selected_stage]
st.write(f"**{len(survivors):,} companies** survived this stage.")

survivors_ranked = survivors.sort_values(
    ["Quality_Score", "_sort_ticker"],
    ascending=[False, True],
    na_position="last",
    kind="mergesort",
)

st.dataframe(
    prepare_display_table(survivors_ranked),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download stage survivors CSV",
    data=survivors.drop(
        columns=["_sort_ticker"], errors="ignore"
    ).to_csv(index=False).encode("utf-8"),
    file_name="stage_survivors.csv",
    mime="text/csv",
    key="download_stage_survivors",
)


# ============================================================
# 13. DATA COVERAGE
# ============================================================

st.header("Data Quality and Coverage")

coverage_rows = []

for label, column in [
    ("Altman Z-Score", "Altman_Z"),
    ("STA", "STA"),
    ("SNOA", "SNOA"),
    ("Gross Profitability", "Gross_Profitability"),
]:
    available = int(eligible[column].notna().sum())
    missing = len(eligible) - available

    coverage_rows.append({
        "Metric": label,
        "Available observations": available,
        "Missing observations": missing,
        "Coverage (%)": round(
            available / len(eligible) * 100, 1
        ) if len(eligible) else 0,
    })

st.dataframe(
    pd.DataFrame(coverage_rows),
    use_container_width=True,
    hide_index=True,
)

complete_count = int(
    eligible[
        ["Altman_Z", "STA", "SNOA", "Gross_Profitability"]
    ].notna().all(axis=1).sum()
)

st.metric("Companies with all four metrics", f"{complete_count:,}")


# ============================================================
# 14. METHODOLOGY
# ============================================================

st.header("Methodology and Limitations")

with st.expander("How the funnel works", expanded=True):
    st.markdown(
        """
**Eligible universe:** companies with parseable market capitalization
meeting the selected minimum.

**Stage 1 — Altman Z:** apply the selected explicit minimum threshold.
The default is 1.81, a commonly cited manufacturing distress boundary.

**Stages 2–4 — rank-based retention:** among observations with valid data
entering each stage, sort companies in the preferred direction and retain
`floor(valid observations × retention percentage)`.

- STA: lowest values first.
- SNOA: lowest values first.
- Gross profitability: highest values first.

Ticker is the deterministic secondary sort key if tied metric values
cross the target boundary. Missing metrics are excluded at that stage and
reported separately; missing accounting values are not replaced with zero.
"""
    )

with st.expander("Composite quality score"):
    st.markdown(
        """
Each metric is converted to a percentile score from 0 to 100. The
direction is standardized so higher always means a more favourable
relative result under this screen.

| Metric | Preferred direction |
|---|---|
| Altman Z-Score | Higher |
| STA | Lower |
| SNOA | Lower |
| Gross Profitability | Higher |

The composite score is the arithmetic mean of the four percentile scores,
with equal weight. Companies must have all four metrics to receive a
composite score.

The score measures relative position within the eligible universe. It is
not a probability of investment success, a price target, or proof that a
stock is undervalued.
"""
    )

with st.expander("Formulas and interpretation"):
    st.markdown(
        """
- **STA** = (Net Income − Operating Cash Flow) / Total Assets.
- **SNOA** = (Operating Assets − Operating Liabilities) / Total Assets.
- **Gross Profitability** = Gross Profit / Total Assets.
- **Altman Z** combines working capital, retained earnings, EBIT, market
  equity relative to liabilities, and sales relative to assets.

The commonly cited manufacturing Altman Z reference zones are below 1.81
for distress and above 2.99 for the safe zone. These are model references,
not guarantees of future outcomes.
"""
    )

with st.expander("Important limitations"):
    st.markdown(
        """
- The screen does not calculate intrinsic value or directly determine
  whether a stock is undervalued.
- The order of the funnel affects the final survivor list.
- Retention percentages are assumptions that can be changed.
- Financial ratios can behave differently across business models.
- Missing, stale, or incorrectly matched data can affect results.
- The SELL list is a relative low-quality ranking, not a short-sale signal.
"""
    )

st.caption(
    "Calculated from universe.csv. Review current company filings, valuation, "
    "liquidity, and company-specific risks before making investment decisions."
)
