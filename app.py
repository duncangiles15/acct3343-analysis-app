
import math
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Screening Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("Stock Screening Analysis")
st.caption(
    "An interactive screening tool using gross profitability, "
    "sales-to-assets, and Altman Z-score."
)


# --------------------------------------------------
# Data loading and preparation
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")

    # Use the main ticker column, with a fallback if necessary.
    if "Ticker" not in df.columns:
        if "Ticker_guess" in df.columns:
            df["Ticker"] = df["Ticker_guess"]
        else:
            df["Ticker"] = ""

    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip()

    # Convert relevant numeric columns without replacing missing values.
    numeric_columns = [
        "Market Cap",
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

    # Parse market-cap strings such as 1.2B or 850M if necessary.
    if "Market Cap" in df.columns:
        def parse_market_cap(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, (int, float, np.number)):
                return float(value)

            text = str(value).strip().replace(",", "").replace("$", "")
            multiplier = 1

            if text.endswith(("T", "t")):
                multiplier = 1e12
                text = text[:-1]
            elif text.endswith(("B", "b")):
                multiplier = 1e9
                text = text[:-1]
            elif text.endswith(("M", "m")):
                multiplier = 1e6
                text = text[:-1]
            elif text.endswith(("K", "k")):
                multiplier = 1e3
                text = text[:-1]

            try:
                return float(text) * multiplier
            except ValueError:
                return np.nan

        df["Market Cap"] = df["Market Cap"].apply(parse_market_cap)

    # Ensure expected columns exist while preserving missing values.
    for col in [
        "Name",
        "Altman_Status",
        "STA_Status",
        "SNOA_Status",
        "GP_Status",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["Name"] = df["Name"].fillna("Unknown company").astype(str)

    return df


try:
    original_df = load_data()
except Exception as exc:
    st.error(
        "The data file could not be loaded. Make sure universe.csv "
        "is in the same GitHub repository as app.py."
    )
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def percentile_flags(series, tickers, percentile, higher_is_better=True):
    """
    Flag the top or bottom percentile of valid observations.

    Missing observations are excluded from the eligible count.
    The cutoff uses floor(eligible firms × percentile).
    Ticker is used as a deterministic secondary sort key.
    """
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()

    result = pd.Series(False, index=series.index)

    eligible = pd.DataFrame({
        "value": values[valid],
        "ticker": tickers[valid].astype(str),
    })

    n_eligible = len(eligible)
    n_selected = math.floor(n_eligible * percentile)

    if n_selected <= 0 or n_eligible == 0:
        return result

    # For the best group, higher values sort first if higher is better.
    best = eligible.sort_values(
        ["value", "ticker"],
        ascending=[not higher_is_better, True],
        kind="mergesort",
    ).head(n_selected)

    # For the worst group, the direction is reversed.
    worst = eligible.sort_values(
        ["value", "ticker"],
        ascending=[higher_is_better, True],
        kind="mergesort",
    ).head(n_selected)

    result.loc[best.index] = True
    worst_flag = pd.Series(False, index=series.index)
    worst_flag.loc[worst.index] = True

    return result, worst_flag


def format_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def display_columns(df):
    desired = [
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
        "Decision",
    ]
    return [col for col in desired if col in df.columns]


def style_table(df):
    """
    Format a copy for display. The underlying numeric values are unchanged.
    """
    out = df.copy()

    if "Market Cap" in out.columns:
        out["Market Cap"] = out["Market Cap"].apply(
            lambda x: f"${x / 1e9:,.2f}B" if pd.notna(x) else "N/A"
        )

    if "Price" in out.columns:
        out["Price"] = out["Price"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
        )

    for col in ["Gross_Profitability", "STA", "SNOA", "Altman_Z"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: format_number(x, 3)
            )

    return out


# --------------------------------------------------
# Sidebar: interactive screening controls
# --------------------------------------------------

st.sidebar.header("Screening Controls")
st.sidebar.caption(
    "Change the thresholds to update the results, candidate lists, "
    "and charts."
)

min_market_cap_millions = st.sidebar.number_input(
    "Minimum market cap (USD millions)",
    min_value=0,
    max_value=10000000,
    value=0,
    step=100,
)

gpa_top_pct = st.sidebar.slider(
    "Gross profitability: top percentile for BUY signal",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
) / 100

gpa_bottom_pct = st.sidebar.slider(
    "Gross profitability: bottom percentile for SELL signal",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
) / 100

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

z_distress = st.sidebar.number_input(
    "Altman Z distress threshold",
    min_value=-10.0,
    max_value=10.0,
    value=1.81,
    step=0.01,
    format="%.2f",
)

z_safe = st.sidebar.number_input(
    "Altman Z safe-zone threshold",
    min_value=-10.0,
    max_value=10.0,
    value=2.99,
    step=0.01,
    format="%.2f",
)

min_buy_signals = st.sidebar.slider(
    "Minimum BUY signals for Potential LONG",
    min_value=1,
    max_value=2,
    value=2,
    step=1,
)

min_sell_signals = st.sidebar.slider(
    "Minimum SELL signals for Potential SHORT",
    min_value=1,
    max_value=3,
    value=2,
    step=1,
)

if z_safe <= z_distress:
    st.sidebar.error(
        "The safe-zone threshold should be greater than the distress threshold."
    )
    st.stop()

if gpa_top_pct + gpa_bottom_pct > 1:
    st.sidebar.error(
        "The gross profitability top and bottom percentiles overlap too much."
    )
    st.stop()

if sta_top_pct + sta_bottom_pct > 1:
    st.sidebar.error(
        "The STA top and bottom percentiles overlap too much."
    )
    st.stop()


# --------------------------------------------------
# Apply basic universe filters
# --------------------------------------------------

df = original_df.copy()

if "Market Cap" in df.columns:
    df = df[
        df["Market Cap"].isna()
        | (df["Market Cap"] >= min_market_cap_millions * 1e6)
    ].copy()

df = df.reset_index(drop=True)

if df.empty:
    st.warning("No companies remain after applying the current filters.")
    st.stop()


# --------------------------------------------------
# Screening signals
# --------------------------------------------------

# Gross profitability:
# Higher values are considered stronger.
gpa_buy, gpa_sell = percentile_flags(
    df["Gross_Profitability"],
    df["Ticker"],
    gpa_top_pct,
    higher_is_better=True,
)

# STA:
# Lower values are considered stronger for this screen.
sta_buy, sta_sell = percentile_flags(
    df["STA"],
    df["Ticker"],
    sta_bottom_pct,
    higher_is_better=False,
)

df["GPA_BUY"] = gpa_buy
df["GPA_SELL"] = gpa_sell
df["STA_BUY"] = sta_buy
df["STA_SELL"] = sta_sell

# Altman Z distress signal.
df["ALTMAN_SELL"] = (
    df["Altman_Z"].notna()
    & (df["Altman_Z"] < z_distress)
)

# A safe-zone result is displayed as context but is not an automatic BUY.
df["ALTMAN_SAFE"] = (
    df["Altman_Z"].notna()
    & (df["Altman_Z"] > z_safe)
)

df["Buy_Signals"] = (
    df["GPA_BUY"].astype(int)
    + df["STA_BUY"].astype(int)
)

df["Sell_Signals"] = (
    df["GPA_SELL"].astype(int)
    + df["STA_SELL"].astype(int)
    + df["ALTMAN_SELL"].astype(int)
)

required_metric_count = (
    df["Gross_Profitability"].notna().astype(int)
    + df["STA"].notna().astype(int)
    + df["Altman_Z"].notna().astype(int)
)

df["Available_Core_Metrics"] = required_metric_count

enough_data = required_metric_count >= 2

potential_long = (
    enough_data
    & (df["Buy_Signals"] >= min_buy_signals)
    & (df["Sell_Signals"] < min_sell_signals)
)

potential_short = (
    enough_data
    & (df["Sell_Signals"] >= min_sell_signals)
    & (df["Buy_Signals"] < min_buy_signals)
)

has_signals = (df["Buy_Signals"] > 0) | (df["Sell_Signals"] > 0)

df["Decision"] = "NEUTRAL"
df.loc[has_signals, "Decision"] = "MIXED"
df.loc[~enough_data, "Decision"] = "REVIEW"
df.loc[potential_long, "Decision"] = "POTENTIAL LONG"
df.loc[potential_short, "Decision"] = "POTENTIAL SHORT"


# --------------------------------------------------
# Rank candidates for the Top 10 lists
# --------------------------------------------------

# Longs: more BUY signals, fewer SELL signals, higher GPA,
# lower STA, and higher Altman Z are preferred as tie-breakers.
long_candidates = df[df["Decision"] == "POTENTIAL LONG"].copy()

long_candidates = long_candidates.sort_values(
    by=[
        "Buy_Signals",
        "Sell_Signals",
        "Gross_Profitability",
        "STA",
        "Altman_Z",
        "Ticker",
    ],
    ascending=[False, True, False, True, False, True],
    na_position="last",
)

long_candidates["Candidate Rank"] = range(1, len(long_candidates) + 1)

# Shorts: more SELL signals, fewer BUY signals, lower GPA,
# higher STA, and lower Altman Z are preferred as tie-breakers.
short_candidates = df[df["Decision"] == "POTENTIAL SHORT"].copy()

short_candidates = short_candidates.sort_values(
    by=[
        "Sell_Signals",
        "Buy_Signals",
        "Gross_Profitability",
        "STA",
        "Altman_Z",
        "Ticker",
    ],
    ascending=[False, True, True, False, True, True],
    na_position="last",
)

short_candidates["Candidate Rank"] = range(1, len(short_candidates) + 1)

top_10_buys = long_candidates.head(10).copy()
top_10_sells = short_candidates.head(10).copy()


# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

st.subheader("Screening Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Companies screened", f"{len(df):,}")
col2.metric("Potential LONG", f"{len(long_candidates):,}")
col3.metric("Potential SHORT", f"{len(short_candidates):,}")
col4.metric(
    "Need review",
    f"{(df['Decision'] == 'REVIEW').sum():,}",
)

st.caption(
    "Potential LONG and Potential SHORT are screening results based on "
    "the selected rules. They are candidates for further research, not "
    "guaranteed investment recommendations."
)


# --------------------------------------------------
# Simplified visual analysis
# --------------------------------------------------

st.subheader("Visual Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    decision_counts = (
        df["Decision"]
        .value_counts()
        .reindex(
            [
                "POTENTIAL LONG",
                "POTENTIAL SHORT",
                "MIXED",
                "NEUTRAL",
                "REVIEW",
            ],
            fill_value=0,
        )
        .rename_axis("Screening outcome")
        .reset_index(name="Companies")
    )

    fig_decisions = px.bar(
        decision_counts,
        x="Screening outcome",
        y="Companies",
        title="Companies by screening outcome",
        text="Companies",
    )

    fig_decisions.update_layout(
        xaxis_title="",
        yaxis_title="Number of companies",
        showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    fig_decisions.update_traces(textposition="outside")
    st.plotly_chart(fig_decisions, use_container_width=True)

with chart_col2:
    plot_df = df.dropna(
        subset=["Gross_Profitability", "STA"]
    ).copy()

    if not plot_df.empty:
        fig_scatter = px.scatter(
            plot_df,
            x="Gross_Profitability",
            y="STA",
            color="Decision",
            hover_name="Ticker",
            hover_data={
                "Name": True,
                "Altman_Z": ":.2f",
                "Buy_Signals": True,
                "Sell_Signals": True,
            },
            title="Gross profitability vs. STA",
            labels={
                "Gross_Profitability": "Gross profitability",
                "STA": "STA",
                "Decision": "Screening outcome",
            },
        )

        fig_scatter.update_layout(
            margin=dict(l=10, r=10, t=55, b=10),
            legend_title_text="Outcome",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info(
            "There are not enough complete GPA and STA observations "
            "to display this chart."
        )

st.caption(
    "The first chart shows how the current rules classify the universe. "
    "The second shows how gross profitability and STA compare for companies "
    "with both values available. Hover over a point for company details."
)


# --------------------------------------------------
# Top 10 candidate lists
# --------------------------------------------------

st.subheader("Top 10 BUY Candidates")
st.write(
    "Potential LONG companies ranked using the existing BUY/SELL signal "
    "rules and the displayed tie-breakers."
)

if top_10_buys.empty:
    st.info(
        "No Potential LONG candidates meet the current screening rules. "
        "Try adjusting the sidebar thresholds."
    )
else:
    buy_display = top_10_buys.copy()
    buy_display.insert(0, "Rank", buy_display["Candidate Rank"])
    cols = ["Rank"] + display_columns(buy_display)
    cols = list(dict.fromkeys(cols))
    st.dataframe(
        style_table(buy_display[cols]),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download Top 10 BUY candidates",
        data=top_10_buys.to_csv(index=False).encode("utf-8"),
        file_name="top_10_buy_candidates.csv",
        mime="text/csv",
    )

st.subheader("Top 10 SELL Candidates")
st.write(
    "Potential SHORT companies ranked using the existing SELL/BUY signal "
    "rules and the displayed tie-breakers."
)

if top_10_sells.empty:
    st.info(
        "No Potential SHORT candidates meet the current screening rules. "
        "Try adjusting the sidebar thresholds."
    )
else:
    sell_display = top_10_sells.copy()
    sell_display.insert(0, "Rank", sell_display["Candidate Rank"])
    cols = ["Rank"] + display_columns(sell_display)
    cols = list(dict.fromkeys(cols))
    st.dataframe(
        style_table(sell_display[cols]),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download Top 10 SELL candidates",
        data=top_10_sells.to_csv(index=False).encode("utf-8"),
        file_name="top_10_sell_candidates.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# Detailed screening results
# --------------------------------------------------

st.subheader("Detailed Results")

tab_long, tab_short, tab_mixed, tab_all = st.tabs(
    ["Potential LONG", "Potential SHORT", "Mixed / Review", "All Companies"]
)

with tab_long:
    st.write(
        "Companies meeting the minimum BUY-signal requirement without "
        "meeting the minimum SELL-signal requirement."
    )
    if long_candidates.empty:
        st.info("No companies currently meet the Potential LONG rules.")
    else:
        st.dataframe(
            style_table(long_candidates[display_columns(long_candidates)]),
            use_container_width=True,
            hide_index=True,
        )

with tab_short:
    st.write(
        "Companies meeting the minimum SELL-signal requirement without "
        "meeting the minimum BUY-signal requirement."
    )
    if short_candidates.empty:
        st.info("No companies currently meet the Potential SHORT rules.")
    else:
        st.dataframe(
            style_table(short_candidates[display_columns(short_candidates)]),
            use_container_width=True,
            hide_index=True,
        )

with tab_mixed:
    mixed_review = df[
        df["Decision"].isin(["MIXED", "REVIEW"])
    ].copy()

    st.write(
        "MIXED companies have signals that do not meet a directional "
        "threshold. REVIEW companies have fewer than two available core "
        "metrics."
    )

    if mixed_review.empty:
        st.info("There are no Mixed or Review companies under these settings.")
    else:
        st.dataframe(
            style_table(mixed_review[display_columns(mixed_review)]),
            use_container_width=True,
            hide_index=True,
        )

with tab_all:
    st.dataframe(
        style_table(df[display_columns(df)]),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download all screening results",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="all_screening_results.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# Company-level explanation
# --------------------------------------------------

st.subheader("Company-Level Explanation")

company_options = (
    df.assign(
        Company_Label=df["Ticker"].astype(str)
        + " — "
        + df["Name"].astype(str)
    )
    .sort_values("Company_Label")
)

selected_label = st.selectbox(
    "Choose a company",
    company_options["Company_Label"].tolist(),
)

selected_row = company_options[
    company_options["Company_Label"] == selected_label
].iloc[0]

left, right = st.columns(2)

with left:
    st.markdown(f"**{selected_row['Ticker']} — {selected_row['Name']}**")
    st.write(f"**Screening outcome:** {selected_row['Decision']}")
    st.write(f"**BUY signals:** {selected_row['Buy_Signals']}")
    st.write(f"**SELL signals:** {selected_row['Sell_Signals']}")
    st.write(
        f"**Available core metrics:** "
        f"{selected_row['Available_Core_Metrics']} of 3"
    )

with right:
    st.write(
        f"**Gross profitability:** "
        f"{format_number(selected_row['Gross_Profitability'], 4)}"
    )
    st.write(f"**STA:** {format_number(selected_row['STA'], 4)}")
    st.write(f"**SNOA:** {format_number(selected_row['SNOA'], 4)}")
    st.write(f"**Altman Z-score:** {format_number(selected_row['Altman_Z'], 3)}")


# Build a transparent explanation of the selected company's signals.
explanations = []

if pd.isna(selected_row["Gross_Profitability"]):
    explanations.append(
        "Gross profitability is missing, so the company was not ranked "
        "on that metric."
    )
else:
    if selected_row["GPA_BUY"]:
        explanations.append(
            "Gross profitability is within the selected top percentile "
            "and contributes a BUY signal."
        )
    elif selected_row["GPA_SELL"]:
        explanations.append(
            "Gross profitability is within the selected bottom percentile "
            "and contributes a SELL signal."
        )
    else:
        explanations.append(
            "Gross profitability is available but is outside the selected "
            "BUY and SELL percentile groups."
        )

if pd.isna(selected_row["STA"]):
    explanations.append(
        "STA is missing, so the company was not ranked on STA."
    )
else:
    if selected_row["STA_BUY"]:
        explanations.append(
            "STA is within the selected bottom percentile and contributes "
            "a BUY signal."
        )
    elif selected_row["STA_SELL"]:
        explanations.append(
            "STA is within the selected top percentile and contributes "
            "a SELL signal."
        )
    else:
        explanations.append(
            "STA is available but is outside the selected BUY and SELL "
            "percentile groups."
        )

if pd.isna(selected_row["Altman_Z"]):
    explanations.append(
        "Altman Z-score is missing, so no distress or safe-zone assessment "
        "is available."
    )
elif selected_row["ALTMAN_SELL"]:
    explanations.append(
        "Altman Z-score is below the selected distress threshold and "
        "contributes a SELL signal."
    )
elif selected_row["ALTMAN_SAFE"]:
    explanations.append(
        "Altman Z-score is above the selected safe-zone threshold. "
        "This is displayed as context and does not automatically add a "
        "BUY signal."
    )
else:
    explanations.append(
        "Altman Z-score is between the selected distress and safe-zone "
        "thresholds."
    )

for explanation in explanations:
    st.markdown(f"- {explanation}")

st.markdown("**Why this screening outcome?**")

if selected_row["Decision"] == "POTENTIAL LONG":
    st.success(
        "This company meets the current Potential LONG signal rules. "
        "The screen is a starting point for further research."
    )
elif selected_row["Decision"] == "POTENTIAL SHORT":
    st.warning(
        "This company meets the current Potential SHORT signal rules. "
        "The screen is a starting point for further research."
    )
elif selected_row["Decision"] == "REVIEW":
    st.info(
        "There are fewer than two available core metrics, so the app "
        "does not assign a directional screening outcome."
    )
elif selected_row["Decision"] == "MIXED":
    st.info(
        "The company has screening signals, but they do not satisfy the "
        "current rules for a Potential LONG or Potential SHORT."
    )
else:
    st.info(
        "The company does not currently meet the selected BUY or SELL "
        "signal thresholds."
    )


# --------------------------------------------------
# Methodology and limitations
# --------------------------------------------------

with st.expander("Screening Methodology"):
    st.markdown(
        """
        **1. Gross profitability**

        Gross profitability is calculated as gross profit divided by total
        assets. Higher values are treated as stronger in this screen.

        **2. STA**

        STA is calculated as net income minus operating cash flow, divided
        by total assets. Lower values are treated as stronger in this screen.

        **3. Altman Z-score**

        The manufacturing version of the Altman Z-score is used. A score
        below the selected distress threshold contributes a SELL signal.
        A score above the safe-zone threshold is displayed as context, but
        does not automatically create a BUY signal.

        **4. Percentile screening**

        The percentile controls determine how many eligible companies are
        selected from the top or bottom of a metric. Missing observations
        are excluded from that metric's eligible ranking; they are not
        replaced with zero. The selected count is the floor of the number
        of eligible firms multiplied by the selected percentile.

        **5. Directional outcomes**

        Potential LONG and Potential SHORT outcomes depend on the selected
        minimum signal counts. Companies with conflicting signals are
        labelled MIXED. Companies with fewer than two available core
        metrics are labelled REVIEW.

        **6. Candidate ranking**

        The Top 10 lists include only companies classified as Potential
        LONG or Potential SHORT under the current settings. Signal counts
        determine the primary ranking, with metric values and ticker used
        as tie-breakers.
        """
    )

with st.expander("Data Coverage and Limitations"):
    coverage_rows = []

    coverage_columns = {
        "Gross profitability": "Gross_Profitability",
        "STA": "STA",
        "SNOA": "SNOA",
        "Altman Z-score": "Altman_Z",
    }

    for label, column in coverage_columns.items():
        if column in df.columns:
            valid_count = int(df[column].notna().sum())
            missing_count = int(df[column].isna().sum())
            coverage_rows.append({
                "Metric": label,
                "Available": valid_count,
                "Missing": missing_count,
                "Coverage (%)": (
                    round(valid_count / len(df) * 100, 1)
                    if len(df)
                    else 0
                ),
            })

    if coverage_rows:
        st.dataframe(
            pd.DataFrame(coverage_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
        - The screening results depend on the data available in the
          supplied dataset and the selected thresholds.
        - Missing accounting data can affect which companies are eligible
          for each metric's percentile ranking.
        - The existing SNOA value is displayed for reference but is not
          used to generate BUY or SELL signals in this version.
        - This tool does not incorporate every factor relevant to an
          investment decision, such as valuation, management quality,
          industry-specific conditions, or future business prospects.
        - A screening result is not a guarantee of future performance and
          should not be treated as a stand-alone investment recommendation.
        """
    )

st.divider()
st.caption(
    "Interactive stock-screening tool for analytical and educational use."
)
