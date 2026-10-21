
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Manufacturing Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Manufacturing Stock Screening & Analysis")
st.write(
    "Screen manufacturing companies using Gross Profitability (GPA), "
    "STA, SNOA, and Altman Z-Score. Adjust the filters to update the "
    "individual screens and combined results."
)


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

def parse_market_cap(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace(",", "").replace("$", "")

    if not text or text.lower() in {"none", "nan", "n/a", "null", "-"}:
        return np.nan

    multiplier = 1.0
    suffixes = {
        "T": 1e12,
        "B": 1e9,
        "M": 1e6,
        "K": 1e3,
    }

    if text[-1].upper() in suffixes:
        multiplier = suffixes[text[-1].upper()]
        text = text[:-1].strip()

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def format_cap(value):
    if pd.isna(value):
        return "N/A"
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def fmt(value):
    return "N/A" if pd.isna(value) else f"{value:,.3f}"


@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")
    df.columns = df.columns.str.strip()

    if "Market Cap" not in df.columns:
        raise ValueError("universe.csv must contain a Market Cap column.")

    df["Market Cap"] = df["Market Cap"].apply(parse_market_cap)

    numeric_columns = [
        "Rank", "Price", "Today", "Price (30 days)",
        "STA", "SNOA", "Altman_Z",
        "Gross_Profit", "Total_Assets", "Gross_Profitability",
        "X1_Working_Capital_Pct", "X2_Retained_Earnings_Pct",
        "X3_EBIT_Pct", "X4_Market_Equity_to_Liabilities_Pct",
        "X5_Sales_to_Assets",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Ticker" not in df.columns:
        df["Ticker"] = df.get("Ticker_guess", "")

    if "Name" not in df.columns:
        df["Name"] = df["Ticker"]

    df["Ticker"] = df["Ticker"].fillna("").astype(str)
    df["Name"] = df["Name"].fillna("").astype(str)

    return df


try:
    universe = load_data()
except Exception as e:
    st.error(f"Could not load universe.csv: {e}")
    st.stop()


# ============================================================
# SETTINGS: SIDEBAR IS CREATED DIRECTLY HERE
# ============================================================

# Using direct st.sidebar controls avoids depending on helper
# functions or duplicated widget keys to create the sidebar.

st.sidebar.title("Screening Filters")
st.sidebar.caption("Change these values to update the app.")

cap_values = universe["Market Cap"].dropna()

if cap_values.empty:
    st.error("No valid market-cap values were found in universe.csv.")
    st.stop()

max_cap = float(cap_values.max())
cap_step = max(max_cap / 1000, 1.0)

min_cap = st.sidebar.slider(
    "Minimum market capitalization",
    min_value=0.0,
    max_value=max_cap,
    value=0.0,
    step=float(cap_step),
    format="%.0f",
    key="sidebar_min_cap",
)

st.sidebar.caption(f"Current minimum: {format_cap(min_cap)}")

sta_pct = st.sidebar.slider(
    "STA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    value=10,
    key="sidebar_sta_pct",
)

snoa_pct = st.sidebar.slider(
    "SNOA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    value=10,
    key="sidebar_snoa_pct",
)

gpa_pct = st.sidebar.slider(
    "GPA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    value=10,
    key="sidebar_gpa_pct",
)

safe_z = st.sidebar.slider(
    "Altman Z safe-zone threshold",
    min_value=2.0,
    max_value=4.0,
    value=2.99,
    step=0.01,
    key="sidebar_safe_z",
)

distress_z = st.sidebar.slider(
    "Altman Z distress threshold",
    min_value=0.5,
    max_value=2.5,
    value=1.81,
    step=0.01,
    key="sidebar_distress_z",
)

min_signals = st.sidebar.slider(
    "Minimum signals for combined classification",
    min_value=1,
    max_value=4,
    value=2,
    key="sidebar_min_signals",
)

if distress_z >= safe_z:
    st.sidebar.error("The distress threshold must be below the safe threshold.")
    st.error("Adjust the Altman Z thresholds in the sidebar.")
    st.stop()


# ============================================================
# FILTER UNIVERSE
# ============================================================

data = universe[
    universe["Market Cap"].notna()
    & (universe["Market Cap"] >= min_cap)
].copy()

if data.empty:
    st.warning("No companies match the current market-cap filter.")
    st.stop()


# ============================================================
# SIGNAL CALCULATIONS
# ============================================================

def percentile_flags(series, pct, higher_is_better):
    buy = pd.Series(False, index=series.index)
    sell = pd.Series(False, index=series.index)

    valid = series.dropna()

    if valid.empty:
        return buy, sell

    low_cutoff = valid.quantile(pct / 100)
    high_cutoff = valid.quantile(1 - pct / 100)
    observed = series.notna()

    if higher_is_better:
        buy.loc[observed] = series.loc[observed] >= high_cutoff
        sell.loc[observed] = series.loc[observed] <= low_cutoff
    else:
        buy.loc[observed] = series.loc[observed] <= low_cutoff
        sell.loc[observed] = series.loc[observed] >= high_cutoff

    return buy, sell


def score_companies(df):
    result = df.copy()

    for col in ["STA", "SNOA", "Altman_Z", "Gross_Profitability"]:
        if col not in result.columns:
            result[col] = np.nan

    result["GPA_BUY"], result["GPA_SELL"] = percentile_flags(
        result["Gross_Profitability"], gpa_pct, True
    )

    result["STA_BUY"], result["STA_SELL"] = percentile_flags(
        result["STA"], sta_pct, False
    )

    result["SNOA_BUY"], result["SNOA_SELL"] = percentile_flags(
        result["SNOA"], snoa_pct, False
    )

    result["ALTMAN_BUY"] = (
        result["Altman_Z"].notna() & (result["Altman_Z"] > safe_z)
    )

    result["ALTMAN_SELL"] = (
        result["Altman_Z"].notna() & (result["Altman_Z"] < distress_z)
    )

    buy_cols = ["GPA_BUY", "STA_BUY", "SNOA_BUY", "ALTMAN_BUY"]
    sell_cols = ["GPA_SELL", "STA_SELL", "SNOA_SELL", "ALTMAN_SELL"]

    result["Buy Signals"] = result[buy_cols].sum(axis=1)
    result["Sell Signals"] = result[sell_cols].sum(axis=1)

    result["Classification"] = "NEUTRAL"

    buy_mask = (
        (result["Buy Signals"] >= min_signals)
        & (result["Buy Signals"] > result["Sell Signals"])
    )

    sell_mask = (
        (result["Sell Signals"] >= min_signals)
        & (result["Sell Signals"] > result["Buy Signals"])
    )

    result.loc[buy_mask, "Classification"] = "BUY"
    result.loc[sell_mask, "Classification"] = "SELL"

    return result


# Recomputed each time Streamlit reruns after a filter change.
scored = score_companies(data)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def display_table(df, columns):
    columns = [c for c in columns if c in df.columns]
    out = df[columns].copy()

    out = out.rename(
        columns={
            "Rank": "Universe Rank",
            "Name": "Company",
            "Market Cap": "Market Cap (USD)",
            "Gross_Profitability": "GPA",
            "Altman_Z": "Altman Z-Score",
        }
    )

    if "Market Cap (USD)" in out.columns:
        out["Market Cap (USD)"] = out["Market Cap (USD)"].apply(format_cap)

    return out


def show_metric_table(df, metric, buy_col, sell_col, higher_is_better):
    valid = df[df[metric].notna()].copy()

    buy = valid[valid[buy_col]].copy()
    sell = valid[valid[sell_col]].copy()

    buy["Screen Result"] = "BUY signal"
    sell["Screen Result"] = "SELL signal"

    # Keep BUY and SELL groups separate so overlapping percentile ties
    # cannot accidentally hide a company's SELL signal.
    buy["_signal_order"] = 0
    sell["_signal_order"] = 1

    candidates = pd.concat([buy, sell], axis=0)
    candidates = candidates.drop_duplicates(subset=["Ticker"], keep="first")

    candidates["_metric_order"] = candidates[metric]

    candidates = candidates.sort_values(
        ["_signal_order", "_metric_order"],
        ascending=[True, not higher_is_better],
        na_position="last",
    ).drop(columns=["_signal_order", "_metric_order"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Valid observations", f"{len(valid):,}")
    c2.metric("BUY candidates", f"{int(valid[buy_col].sum()):,}")
    c3.metric("SELL candidates", f"{int(valid[sell_col].sum()):,}")

    st.caption(
        "The table below shows only companies that currently qualify "
        "for this metric's BUY or SELL signal."
    )

    if candidates.empty:
        st.info("No companies currently qualify for this screen.")
        return

    st.dataframe(
        display_table(
            candidates,
            ["Rank", "Ticker", "Name", "Market Cap", metric, "Screen Result"],
        ),
        use_container_width=True,
        hide_index=True,
    )


def make_chart(fig, height=350):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=55, b=15),
    )
    return fig


# ============================================================
# STEP 1 — UNIVERSE
# ============================================================

st.header("Step 1 — Investment Universe")

st.write(
    "The minimum market-cap filter in the sidebar determines which "
    "companies enter the screens below."
)

c1, c2, c3 = st.columns(3)
c1.metric("Source companies", f"{len(universe):,}")
c2.metric("Companies in current universe", f"{len(data):,}")
c3.metric("Excluded by size filter", f"{len(universe) - len(data):,}")

with st.expander("View companies in the current universe"):
    st.dataframe(
        display_table(
            data,
            ["Rank", "Ticker", "Name", "Market Cap", "Price", "Country"],
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STEP 2 — STA
# ============================================================

st.header("Step 2 — Earnings Quality (STA)")

st.latex(
    r"\mathrm{STA}="
    r"\frac{\mathrm{Net\ Income}-\mathrm{Operating\ Cash\ Flow}}"
    r"{\mathrm{Total\ Assets}}"
)

sta_main = st.slider(
    "STA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    value=sta_pct,
    key="main_sta_pct",
)

if sta_main != sta_pct:
    sta_pct = sta_main
    scored = score_companies(data)

show_metric_table(scored, "STA", "STA_BUY", "STA_SELL", False)


# ============================================================
# STEP 3 — SNOA
# ============================================================

st.header("Step 3 — Net Operating Assets (SNOA)")

st.latex(
    r"\mathrm{SNOA}="
    r"\frac{\mathrm{Operating\ Assets}-\mathrm{Operating\ Liabilities}}"
    r"{\mathrm{Total\ Assets}}"
)

snoa_main = st.slider(
    "SNOA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    value=snoa_pct,
    key="main_snoa_pct",
)

if snoa_main != snoa_pct:
    snoa_pct = snoa_main
    scored = score_companies(data)

show_metric_table(scored, "SNOA", "SNOA_BUY", "SNOA_SELL", False)


# ============================================================
# STEP 4 — ALTMAN Z
# ============================================================

st.header("Step 4 — Financial Distress (Altman Z-Score)")

st.latex(
    r"Z=1.2X_1+1.4X_2+3.3X_3+0.6X_4+1.0X_5"
)

z1, z2 = st.columns(2)

with z1:
    safe_main = st.slider(
        "Safe-zone threshold",
        min_value=2.0,
        max_value=4.0,
        value=safe_z,
        step=0.01,
        key="main_safe_z",
    )

with z2:
    distress_main = st.slider(
        "Distress threshold",
        min_value=0.5,
        max_value=2.5,
        value=distress_z,
        step=0.01,
        key="main_distress_z",
    )

if distress_main >= safe_main:
    st.warning("The distress threshold must be below the safe threshold.")
    st.stop()

if safe_main != safe_z or distress_main != distress_z:
    safe_z = safe_main
    distress_z = distress_main
    scored = score_companies(data)

show_metric_table(scored, "Altman_Z", "ALTMAN_BUY", "ALTMAN_SELL", True)

z_valid = scored["Altman_Z"].dropna()

if not z_valid.empty:
    fig_z = px.histogram(
        scored.dropna(subset=["Altman_Z"]),
        x="Altman_Z",
        nbins=25,
        title="Distribution of Altman Z-Scores",
    )

    fig_z.add_vline(
        x=distress_z,
        line_dash="dash",
        annotation_text="Distress threshold",
    )

    fig_z.add_vline(
        x=safe_z,
        line_dash="dash",
        annotation_text="Safe threshold",
    )

    st.plotly_chart(make_chart(fig_z, 380), use_container_width=True)


# ============================================================
# STEP 5 — GPA
# ============================================================

st.header("Step 5 — Gross Profitability (GPA)")

st.latex(
    r"\mathrm{GPA}="
    r"\frac{\mathrm{Gross\ Profit}}{\mathrm{Total\ Assets}}"
)

gpa_main = st.slider(
    "GPA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    value=gpa_pct,
    key="main_gpa_pct",
)

if gpa_main != gpa_pct:
    gpa_pct = gpa_main
    scored = score_companies(data)

show_metric_table(
    scored,
    "Gross_Profitability",
    "GPA_BUY",
    "GPA_SELL",
    True,
)


# ============================================================
# STEP 6 — COMBINED RESULTS
# ============================================================

st.header("Step 6 — Combined Screening Results")

signals_main = st.slider(
    "Minimum signals required",
    min_value=1,
    max_value=4,
    value=min_signals,
    key="main_min_signals",
)

if signals_main != min_signals:
    min_signals = signals_main
    scored = score_companies(data)

counts = (
    scored["Classification"]
    .value_counts()
    .reindex(["BUY", "NEUTRAL", "SELL"], fill_value=0)
)

m1, m2, m3 = st.columns(3)
m1.metric("BUY", f"{counts['BUY']:,}")
m2.metric("NEUTRAL", f"{counts['NEUTRAL']:,}")
m3.metric("SELL", f"{counts['SELL']:,}")

fig_class = px.bar(
    x=counts.index,
    y=counts.values,
    labels={"x": "Classification", "y": "Companies"},
    title="Companies by Classification",
)

st.plotly_chart(make_chart(fig_class), use_container_width=True)

signal_counts = pd.DataFrame(
    {
        "Screen": ["GPA", "STA", "SNOA", "Altman Z"],
        "BUY": [
            scored["GPA_BUY"].sum(),
            scored["STA_BUY"].sum(),
            scored["SNOA_BUY"].sum(),
            scored["ALTMAN_BUY"].sum(),
        ],
        "SELL": [
            scored["GPA_SELL"].sum(),
            scored["STA_SELL"].sum(),
            scored["SNOA_SELL"].sum(),
            scored["ALTMAN_SELL"].sum(),
        ],
    }
)

fig_signals = px.bar(
    signal_counts,
    x="Screen",
    y=["BUY", "SELL"],
    barmode="group",
    title="BUY and SELL Signals by Screen",
)

st.plotly_chart(make_chart(fig_signals), use_container_width=True)

st.subheader("All companies and combined results")

st.dataframe(
    display_table(
        scored.sort_values(
            ["Classification", "Buy Signals", "Sell Signals"],
            ascending=[True, False, True],
        ),
        [
            "Rank", "Ticker", "Name", "Market Cap",
            "Gross_Profitability", "STA", "SNOA", "Altman_Z",
            "Buy Signals", "Sell Signals", "Classification",
        ],
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download all screening results",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name="manufacturing_stock_screen_results.csv",
    mime="text/csv",
)


# ============================================================
# STEP 7 — TOP 10
# ============================================================

st.header("Step 7 — Top 10 BUY and SELL Candidates")

buys = scored[scored["Classification"] == "BUY"].copy()
sells = scored[scored["Classification"] == "SELL"].copy()

buys = buys.sort_values(
    ["Buy Signals", "Sell Signals", "Gross_Profitability"],
    ascending=[False, True, False],
    na_position="last",
).head(10)

sells = sells.sort_values(
    ["Sell Signals", "Buy Signals", "Gross_Profitability"],
    ascending=[False, True, True],
    na_position="last",
).head(10)

buy_tab, sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

with buy_tab:
    if buys.empty:
        st.info("No companies meet the current combined BUY criteria.")
    else:
        st.dataframe(
            display_table(
                buys,
                [
                    "Rank", "Ticker", "Name", "Market Cap",
                    "Gross_Profitability", "STA", "SNOA", "Altman_Z",
                    "Buy Signals", "Sell Signals", "Classification",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )

with sell_tab:
    if sells.empty:
        st.info("No companies meet the current combined SELL criteria.")
    else:
        st.dataframe(
            display_table(
                sells,
                [
                    "Rank", "Ticker", "Name", "Market Cap",
                    "Gross_Profitability", "STA", "SNOA", "Altman_Z",
                    "Buy Signals", "Sell Signals", "Classification",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# COMPANY EXPLORER
# ============================================================

st.header("Company Explorer")

query = st.text_input(
    "Search for a company or ticker",
    placeholder="Enter a company name or ticker...",
)

if query.strip():
    matches = scored[
        scored["Ticker"].str.contains(query, case=False, na=False, regex=False)
        | scored["Name"].str.contains(query, case=False, na=False, regex=False)
    ]

    if matches.empty:
        st.info("No matching companies found in the current universe.")
    else:
        st.dataframe(
            display_table(
                matches,
                [
                    "Rank", "Ticker", "Name", "Market Cap",
                    "Gross_Profitability", "STA", "SNOA", "Altman_Z",
                    "Buy Signals", "Sell Signals", "Classification",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# METHODOLOGY
# ============================================================

st.header("Methodology and Data Coverage")

st.markdown(
    """
- **GPA:** Gross Profit divided by Total Assets. Higher is preferred.
- **STA:** (Net Income − Operating Cash Flow) divided by Total Assets.
  Lower is preferred under this screen's convention.
- **SNOA:** Existing dataset value for net operating assets divided by
  total assets. Lower is preferred under this screen's convention.
- **Altman Z-Score:** Uses the provided score. Above the safe threshold
  is a BUY signal; below the distress threshold is a SELL signal.
- Missing metric values are excluded from that metric's percentile
  calculations and receive no signal.
- Percentile ties can cause the number of candidates to exceed the
  nominal cutoff.
- Combined classifications depend on the minimum signal setting and
  whether BUY or SELL signals outnumber the opposing signals.

This dashboard is a screening tool for research and education, not a
guarantee of performance or personalized investment advice.
"""
)

coverage = []

for metric, label in [
    ("Gross_Profitability", "GPA"),
    ("STA", "STA"),
    ("SNOA", "SNOA"),
    ("Altman_Z", "Altman Z-Score"),
]:
    available = int(data[metric].notna().sum())
    coverage.append(
        {
            "Metric": label,
            "Available": available,
            "Missing": len(data) - available,
            "Coverage (%)": round(100 * available / len(data), 1),
        }
    )

st.dataframe(
    pd.DataFrame(coverage),
    use_container_width=True,
    hide_index=True,
)
