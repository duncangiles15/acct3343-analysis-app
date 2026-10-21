
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
)

st.title("Manufacturing Stock Screening & Analysis")
st.caption(
    "Quantitative screening of the manufacturing firms in the supplied "
    "universe.csv dataset, ranked by market capitalization."
)

st.markdown(
    """
This dashboard compares four financial measures to identify companies
that meet the selected BUY and SELL screening rules:

- **STA:** earnings quality relative to operating cash flow.
- **SNOA:** net operating assets relative to total assets.
- **Altman Z-Score:** an indicator of financial-distress risk.
- **GPA:** gross profit relative to total assets.

Screening results depend on the selected thresholds and the supplied data.
They are not guarantees of future returns or personalized investment advice.
"""
)


# ============================================================
# SHARED SETTINGS AND SLIDER SYNCHRONIZATION
# ============================================================

DEFAULTS = {
    "min_market_cap": 0.0,
    "gpa_pct": 10,
    "sta_pct": 10,
    "snoa_pct": 10,
    "safe_z": 2.99,
    "distress_z": 1.81,
    "min_signals": 2,
}

WIDGET_KEYS = {
    "min_market_cap": ["cap_sidebar", "cap_main"],
    "gpa_pct": ["gpa_sidebar", "gpa_main"],
    "sta_pct": ["sta_sidebar", "sta_main"],
    "snoa_pct": ["snoa_sidebar", "snoa_main"],
    "safe_z": ["safe_sidebar", "safe_main"],
    "distress_z": ["distress_sidebar", "distress_main"],
    "min_signals": ["signals_sidebar", "signals_main"],
}


def initialize_settings():
    for setting, default in DEFAULTS.items():
        if setting not in st.session_state:
            st.session_state[setting] = default

        for key in WIDGET_KEYS[setting]:
            if key not in st.session_state:
                st.session_state[key] = st.session_state[setting]


def sync_setting(setting, source_key):
    """Synchronize both widget copies before Streamlit reruns the app."""
    value = st.session_state[source_key]
    st.session_state[setting] = value

    for key in WIDGET_KEYS[setting]:
        st.session_state[key] = value


initialize_settings()


# ============================================================
# DATA CLEANING
# ============================================================

def parse_market_cap(value):
    """Convert values such as $2.4T, $850B and $45.2M to USD."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if not text or text.lower() in {
        "none", "nan", "n/a", "na", "null", "-", "--"
    }:
        return np.nan

    text = (
        text.replace(",", "")
        .replace("US$", "")
        .replace("$", "")
        .replace("USD", "")
        .strip()
    )

    multipliers = {
        "T": 1e12,
        "B": 1e9,
        "M": 1e6,
        "K": 1e3,
    }

    words = {
        "trillion": 1e12,
        "billion": 1e9,
        "million": 1e6,
        "thousand": 1e3,
    }

    multiplier = 1
    lowered = text.lower()

    for word, factor in words.items():
        if lowered.endswith(word):
            text = text[:-len(word)].strip()
            multiplier = factor
            break
    else:
        if text and text[-1].upper() in multipliers:
            multiplier = multipliers[text[-1].upper()]
            text = text[:-1].strip()

    try:
        return float(text) * multiplier
    except (ValueError, TypeError):
        return np.nan


def format_market_cap(value):
    if pd.isna(value):
        return "N/A"

    if abs(value) >= 1e12:
        return f"${value / 1e12:,.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"

    return f"${value:,.0f}"


def format_number(value, decimals=3):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


@st.cache_data
def load_data():
    frame = pd.read_csv("universe.csv")
    frame.columns = frame.columns.str.strip()

    if "Market Cap" not in frame.columns:
        raise ValueError("universe.csv must contain a Market Cap column.")

    frame["Market Cap"] = frame["Market Cap"].apply(parse_market_cap)

    numeric_columns = [
        "Rank",
        "Price",
        "Today",
        "Price (30 days)",
        "Altman_Z",
        "X1_Working_Capital_Pct",
        "X2_Retained_Earnings_Pct",
        "X3_EBIT_Pct",
        "X4_Market_Equity_to_Liabilities_Pct",
        "X5_Sales_to_Assets",
        "STA",
        "SNOA",
        "Gross_Profit",
        "Total_Assets",
        "Gross_Profitability",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            )

    if "Ticker" not in frame.columns:
        frame["Ticker"] = (
            frame["Ticker_guess"]
            if "Ticker_guess" in frame.columns
            else ""
        )

    if "Name" not in frame.columns:
        frame["Name"] = frame["Ticker"]

    frame["Ticker"] = frame["Ticker"].fillna("").astype(str)
    frame["Name"] = frame["Name"].fillna("").astype(str)

    return frame.drop_duplicates().reset_index(drop=True)


try:
    df_original = load_data()
except FileNotFoundError:
    st.error(
        "universe.csv was not found. Place it in the same repository "
        "folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# ============================================================
# DISPLAY TABLES
# ============================================================

def display_table(frame, columns=None):
    if columns is None:
        columns = [
            "Rank",
            "Ticker",
            "Name",
            "Market Cap",
            "Price",
            "Gross_Profitability",
            "STA",
            "SNOA",
            "Altman_Z",
            "Buy Signals",
            "Sell Signals",
            "Classification",
        ]

    columns = [column for column in columns if column in frame.columns]
    result = frame[columns].copy()

    result = result.rename(
        columns={
            "Rank": "Universe Rank",
            "Name": "Company",
            "Market Cap": "Market Cap (USD)",
            "Price": "Share Price",
            "Gross_Profitability": "GPA",
            "Altman_Z": "Altman Z-Score",
        }
    )

    if "Market Cap (USD)" in result.columns:
        result["Market Cap (USD)"] = (
            result["Market Cap (USD)"].apply(format_market_cap)
        )

    return result


# ============================================================
# CLEAN, CONSISTENT CHART STYLE
# ============================================================

CHART_TEMPLATE = "plotly_white"


def clean_chart(fig, height=360):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=12, r=12, t=58, b=18),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#263445",
        ),
        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(size=17),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        automargin=True,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E9EDF2",
        zeroline=False,
        automargin=True,
    )

    return fig


# ============================================================
# PERCENTILE SCREENS
# ============================================================

def percentile_flags(series, percentile, higher_is_better):
    """
    Return BUY and SELL flags based on the tails of valid observations.
    Missing metrics receive no signal.
    """
    buy = pd.Series(False, index=series.index)
    sell = pd.Series(False, index=series.index)

    valid = series.dropna()

    if valid.empty:
        return buy, sell, np.nan, np.nan

    low_cutoff = valid.quantile(percentile / 100)
    high_cutoff = valid.quantile(1 - percentile / 100)

    observed = series.notna()

    if higher_is_better:
        buy.loc[observed] = series.loc[observed] >= high_cutoff
        sell.loc[observed] = series.loc[observed] <= low_cutoff
    else:
        buy.loc[observed] = series.loc[observed] <= low_cutoff
        sell.loc[observed] = series.loc[observed] >= high_cutoff

    return buy, sell, low_cutoff, high_cutoff


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("Screening Controls")
st.sidebar.caption("Adjust settings here or inside each step.")

market_caps = df_original["Market Cap"].dropna()

if market_caps.empty:
    st.error("No usable market-cap values were found in universe.csv.")
    st.stop()

cap_min = 0.0
cap_max = float(market_caps.max())
cap_step = max(cap_max / 1000, 1_000_000.0)

if cap_max <= cap_min:
    st.error("The market-cap values do not support a usable slider.")
    st.stop()


def add_cap_slider(location):
    key = "cap_sidebar" if location == "sidebar" else "cap_main"
    st.slider(
        "Minimum market capitalization (USD)",
        min_value=cap_min,
        max_value=cap_max,
        step=float(cap_step),
        key=key,
        on_change=sync_setting,
        args=("min_market_cap", key),
        format="%.0f",
    )


def add_percent_slider(setting, location, label, maximum=25):
    key = f"{setting}_{location}"
    st.slider(
        label,
        min_value=1,
        max_value=maximum,
        key=key,
        on_change=sync_setting,
        args=(setting, key),
        format="%d%%",
    )


def add_z_slider(setting, location, label, low, high):
    key = f"{setting}_{location}"
    st.slider(
        label,
        min_value=low,
        max_value=high,
        step=0.01,
        key=key,
        on_change=sync_setting,
        args=(setting, key),
        format="%.2f",
    )


st.sidebar.subheader("1. Investment Universe")
add_cap_slider("sidebar")
st.sidebar.caption(
    "Current minimum: "
    + format_market_cap(st.session_state["min_market_cap"])
)

st.sidebar.subheader("2. STA")
add_percent_slider("sta_pct", "sidebar", "STA percentile cutoff")

st.sidebar.subheader("3. SNOA")
add_percent_slider("snoa_pct", "sidebar", "SNOA percentile cutoff")

st.sidebar.subheader("4. Altman Z-Score")
add_z_slider(
    "safe_z", "sidebar", "Safe-zone threshold", 2.0, 4.0
)
add_z_slider(
    "distress_z", "sidebar", "Distress threshold", 0.5, 2.5
)

st.sidebar.subheader("5. GPA")
add_percent_slider("gpa_pct", "sidebar", "GPA percentile cutoff")

st.sidebar.subheader("6. Combined Signals")
st.sidebar.slider(
    "Minimum signals required",
    min_value=1,
    max_value=4,
    key="signals_sidebar",
    on_change=sync_setting,
    args=("min_signals", "signals_sidebar"),
)


# ============================================================
# FILTER UNIVERSE
# ============================================================

data = df_original[
    df_original["Market Cap"].notna()
    & (
        df_original["Market Cap"]
        >= st.session_state["min_market_cap"]
    )
].copy()

if data.empty:
    st.warning(
        "No companies remain after applying the market-cap filter. "
        "Lower the minimum market capitalization."
    )
    st.stop()

if st.session_state["distress_z"] >= st.session_state["safe_z"]:
    st.error(
        "The distress threshold must be below the safe-zone threshold."
    )
    st.stop()


# ============================================================
# RECALCULATE ALL SIGNALS FROM CURRENT SETTINGS
# ============================================================

def score_companies(universe):
    result = universe.copy()

    for metric in [
        "STA",
        "SNOA",
        "Altman_Z",
        "Gross_Profitability",
    ]:
        if metric not in result.columns:
            result[metric] = np.nan

    result["GPA_BUY"], result["GPA_SELL"], _, _ = percentile_flags(
        result["Gross_Profitability"],
        st.session_state["gpa_pct"],
        higher_is_better=True,
    )

    result["STA_BUY"], result["STA_SELL"], _, _ = percentile_flags(
        result["STA"],
        st.session_state["sta_pct"],
        higher_is_better=False,
    )

    result["SNOA_BUY"], result["SNOA_SELL"], _, _ = percentile_flags(
        result["SNOA"],
        st.session_state["snoa_pct"],
        higher_is_better=False,
    )

    result["ALTMAN_BUY"] = (
        result["Altman_Z"].notna()
        & (result["Altman_Z"] > st.session_state["safe_z"])
    )

    result["ALTMAN_SELL"] = (
        result["Altman_Z"].notna()
        & (result["Altman_Z"] < st.session_state["distress_z"])
    )

    buy_columns = [
        "GPA_BUY", "STA_BUY", "SNOA_BUY", "ALTMAN_BUY"
    ]
    sell_columns = [
        "GPA_SELL", "STA_SELL", "SNOA_SELL", "ALTMAN_SELL"
    ]

    result["Buy Signals"] = result[buy_columns].sum(axis=1)
    result["Sell Signals"] = result[sell_columns].sum(axis=1)

    minimum = int(st.session_state["min_signals"])

    result["Classification"] = "NEUTRAL"

    buy_mask = (
        (result["Buy Signals"] >= minimum)
        & (result["Buy Signals"] > result["Sell Signals"])
    )
    sell_mask = (
        (result["Sell Signals"] >= minimum)
        & (result["Sell Signals"] > result["Buy Signals"])
    )

    result.loc[buy_mask, "Classification"] = "BUY"
    result.loc[sell_mask, "Classification"] = "SELL"

    return result


# Critical: score after reading the current widget settings.
# Every screen below uses this same dataframe.
scored_data = score_companies(data)


# ============================================================
# INDIVIDUAL SCREEN TABLE FUNCTION
# ============================================================

def show_metric_table(data_with_signals, metric, buy_col, sell_col):
    required = [metric, buy_col, sell_col]
    missing = [
        column for column in required
        if column not in data_with_signals.columns
    ]

    if missing:
        st.error(
            "Required screen columns are missing: "
            + ", ".join(missing)
        )
        return

    valid = data_with_signals[
        data_with_signals[metric].notna()
    ].copy()

    missing_count = int(data_with_signals[metric].isna().sum())

    col1, col2 = st.columns(2)
    col1.metric("Valid observations", f"{len(valid):,}")
    col2.metric("Missing observations", f"{missing_count:,}")

    if valid.empty:
        st.info("No valid observations are available for this screen.")
        return

    valid["Screen Result"] = np.select(
        [valid[buy_col], valid[sell_col]],
        ["BUY signal", "SELL signal"],
        default="No signal",
    )

    ascending = metric != "Gross_Profitability"
    valid = valid.sort_values(
        metric,
        ascending=ascending,
        na_position="last",
    )

    columns = [
        column for column in [
            "Rank",
            "Ticker",
            "Name",
            "Market Cap",
            metric,
            "Screen Result",
        ]
        if column in valid.columns
    ]

    st.dataframe(
        display_table(valid, columns),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STEP 1 — UNIVERSE
# ============================================================

st.header("Step 1 — Investment Universe")
st.markdown(
    "Filter the supplied manufacturing-firm universe by market "
    "capitalization. The size filter defines which firms enter the "
    "subsequent screens."
)

add_cap_slider("main")

u1, u2, u3 = st.columns(3)
u1.metric("Firms in source dataset", f"{len(df_original):,}")
u2.metric("Firms in current universe", f"{len(data):,}")
u3.metric(
    "Excluded by size filter",
    f"{len(df_original) - len(data):,}",
)

with st.expander("View firms in the current universe"):
    st.dataframe(
        display_table(data),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STEP 2 — STA
# ============================================================

st.header("Step 2 — Earnings Quality (STA)")

st.markdown(
    """
STA compares net income with operating cash flow relative to total assets.
Under this screen's convention, lower STA is preferred.
"""
)

st.latex(
    r"\mathrm{STA}="
    r"\frac{\mathrm{Net\ Income}-\mathrm{Operating\ Cash\ Flow}}"
    r"{\mathrm{Total\ Assets}}"
)

add_percent_slider("sta_pct", "main", "STA percentile cutoff")

st.caption(
    f"Current cutoff: {st.session_state['sta_pct']}% at each end."
)

show_metric_table(scored_data, "STA", "STA_BUY", "STA_SELL")


# ============================================================
# STEP 3 — SNOA
# ============================================================

st.header("Step 3 — Net Operating Assets (SNOA)")

st.markdown(
    """
SNOA measures net operating assets relative to total assets. Under the
screening convention used here, lower values are preferred. Existing SNOA
values from the dataset are used directly.
"""
)

st.latex(
    r"\mathrm{SNOA}="
    r"\frac{\mathrm{Operating\ Assets}-\mathrm{Operating\ Liabilities}}"
    r"{\mathrm{Total\ Assets}}"
)

add_percent_slider("snoa_pct", "main", "SNOA percentile cutoff")

st.caption(
    f"Current cutoff: {st.session_state['snoa_pct']}% at each end."
)

show_metric_table(scored_data, "SNOA", "SNOA_BUY", "SNOA_SELL")


# ============================================================
# STEP 4 — ALTMAN Z-SCORE
# ============================================================

st.header("Step 4 — Financial Distress (Altman Z-Score)")

st.markdown(
    """
The supplied Altman Z-Scores are used directly. Scores above the safe-zone
threshold receive a BUY signal; scores below the distress threshold receive
a SELL signal. Scores between the thresholds are neutral.
"""
)

st.latex(
    r"Z=1.2X_1+1.4X_2+3.3X_3+0.6X_4+1.0X_5"
)

zc1, zc2 = st.columns(2)

with zc1:
    add_z_slider(
        "safe_z", "main", "Safe-zone threshold", 2.0, 4.0
    )

with zc2:
    add_z_slider(
        "distress_z", "main", "Distress threshold", 0.5, 2.5
    )

if st.session_state["distress_z"] >= st.session_state["safe_z"]:
    st.error("The distress threshold must be below the safe-zone threshold.")
    st.stop()

z_valid = scored_data[scored_data["Altman_Z"].notna()].copy()

z1, z2, z3 = st.columns(3)
z1.metric("Valid Z-Scores", f"{len(z_valid):,}")
z2.metric("BUY signals", f"{int(z_valid['ALTMAN_BUY'].sum()):,}")
z3.metric("SELL signals", f"{int(z_valid['ALTMAN_SELL'].sum()):,}")

if not z_valid.empty:
    z_valid["Screen Result"] = np.select(
        [z_valid["ALTMAN_BUY"], z_valid["ALTMAN_SELL"]],
        ["BUY signal", "SELL signal"],
        default="Neutral",
    )

    zcols = [
        column for column in [
            "Rank", "Ticker", "Name", "Market Cap",
            "Altman_Z", "Screen Result",
        ]
        if column in z_valid.columns
    ]

    st.dataframe(
        display_table(z_valid, zcols),
        use_container_width=True,
        hide_index=True,
    )

    fig_z = px.histogram(
        z_valid,
        x="Altman_Z",
        nbins=25,
        title="Distribution of Altman Z-Scores",
        labels={"Altman_Z": "Altman Z-Score"},
    )

    fig_z.add_vline(
        x=st.session_state["distress_z"],
        line_dash="dash",
        line_color="#C0392B",
        annotation_text="Distress",
        annotation_position="top left",
    )
    fig_z.add_vline(
        x=st.session_state["safe_z"],
        line_dash="dash",
        line_color="#2471A3",
        annotation_text="Safe zone",
        annotation_position="top right",
    )

    fig_z.update_layout(
        showlegend=False,
        xaxis_title="Altman Z-Score",
        yaxis_title="Number of firms",
    )

    st.plotly_chart(
        clean_chart(fig_z, height=390),
        use_container_width=True,
    )


# ============================================================
# STEP 5 — GPA
# ============================================================

st.header("Step 5 — Gross Profitability (GPA)")

st.markdown(
    """
GPA measures gross profit relative to total assets. Higher values are
preferred under this screening convention.
"""
)

st.latex(
    r"\mathrm{GPA}="
    r"\frac{\mathrm{Gross\ Profit}}{\mathrm{Total\ Assets}}"
)

add_percent_slider("gpa_pct", "main", "GPA percentile cutoff")

st.caption(
    f"Current cutoff: {st.session_state['gpa_pct']}% at each end."
)

show_metric_table(
    scored_data,
    "Gross_Profitability",
    "GPA_BUY",
    "GPA_SELL",
)


# ============================================================
# STEP 6 — COMBINED RESULTS
# ============================================================

st.header("Step 6 — Combined Screening Results")

st.markdown(
    """
Each company can receive up to four BUY signals and four SELL signals.
A company is classified BUY or SELL when it reaches the minimum signal
count in that direction and has more signals in that direction than the
other. Other companies are classified NEUTRAL.
"""
)

st.slider(
    "Minimum matching signals required",
    min_value=1,
    max_value=4,
    key="signals_main",
    on_change=sync_setting,
    args=("min_signals", "signals_main"),
)

counts = (
    scored_data["Classification"]
    .value_counts()
    .reindex(["BUY", "NEUTRAL", "SELL"], fill_value=0)
)

cc1, cc2, cc3 = st.columns(3)
cc1.metric("BUY", f"{counts['BUY']:,}")
cc2.metric("NEUTRAL", f"{counts['NEUTRAL']:,}")
cc3.metric("SELL", f"{counts['SELL']:,}")

class_chart_data = pd.DataFrame(
    {
        "Classification": ["BUY", "NEUTRAL", "SELL"],
        "Companies": [
            int(counts["BUY"]),
            int(counts["NEUTRAL"]),
            int(counts["SELL"]),
        ],
    }
)

fig_class = px.bar(
    class_chart_data,
    x="Classification",
    y="Companies",
    text="Companies",
    title="Companies by Classification",
    category_orders={
        "Classification": ["BUY", "NEUTRAL", "SELL"]
    },
)

fig_class.update_traces(
    textposition="outside",
    cliponaxis=False,
    showlegend=False,
)

fig_class.update_layout(
    yaxis_title="Number of firms",
    xaxis_title="",
)

st.plotly_chart(
    clean_chart(fig_class, height=330),
    use_container_width=True,
)

signal_chart_data = pd.DataFrame(
    {
        "Screen": ["GPA", "STA", "SNOA", "Altman Z"],
        "BUY": [
            int(scored_data["GPA_BUY"].sum()),
            int(scored_data["STA_BUY"].sum()),
            int(scored_data["SNOA_BUY"].sum()),
            int(scored_data["ALTMAN_BUY"].sum()),
        ],
        "SELL": [
            int(scored_data["GPA_SELL"].sum()),
            int(scored_data["STA_SELL"].sum()),
            int(scored_data["SNOA_SELL"].sum()),
            int(scored_data["ALTMAN_SELL"].sum()),
        ],
    }
)

signal_long = signal_chart_data.melt(
    id_vars="Screen",
    value_vars=["BUY", "SELL"],
    var_name="Signal",
    value_name="Companies",
)

fig_signals = px.bar(
    signal_long,
    x="Companies",
    y="Screen",
    color="Signal",
    orientation="h",
    barmode="group",
    text="Companies",
    title="BUY and SELL Signals by Screen",
    category_orders={
        "Screen": ["GPA", "STA", "SNOA", "Altman Z"]
    },
)

fig_signals.update_traces(
    textposition="outside",
    cliponaxis=False,
)

fig_signals.update_layout(
    xaxis_title="Number of firms",
    yaxis_title="",
    legend_title_text="",
)

st.plotly_chart(
    clean_chart(fig_signals, height=350),
    use_container_width=True,
)

st.subheader("All screening results")

st.dataframe(
    display_table(
        scored_data.sort_values(
            ["Classification", "Buy Signals", "Sell Signals"],
            ascending=[True, False, True],
        )
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download all screening results",
    data=scored_data.to_csv(index=False).encode("utf-8"),
    file_name="manufacturing_stock_screen_results.csv",
    mime="text/csv",
)


# ============================================================
# STEP 7 — TOP 10 BUY / SELL
# ============================================================

st.header("Step 7 — Top 10 BUY and SELL Candidates")

buy_candidates = scored_data[
    scored_data["Classification"] == "BUY"
].copy()

sell_candidates = scored_data[
    scored_data["Classification"] == "SELL"
].copy()

if not buy_candidates.empty:
    buy_candidates = buy_candidates.sort_values(
        [
            "Buy Signals",
            "Sell Signals",
            "Gross_Profitability",
            "STA",
            "SNOA",
        ],
        ascending=[False, True, False, True, True],
        na_position="last",
    )

if not sell_candidates.empty:
    sell_candidates = sell_candidates.sort_values(
        [
            "Sell Signals",
            "Buy Signals",
            "Gross_Profitability",
            "STA",
            "SNOA",
        ],
        ascending=[False, True, True, False, False],
        na_position="last",
    )

buy_top10 = buy_candidates.head(10)
sell_top10 = sell_candidates.head(10)

buy_tab, sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

with buy_tab:
    if buy_top10.empty:
        st.info("No companies meet the current BUY criteria.")
    else:
        st.dataframe(
            display_table(buy_top10),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Top 10 BUY",
            data=buy_top10.to_csv(index=False).encode("utf-8"),
            file_name="top_10_buy.csv",
            mime="text/csv",
            key="download_buy",
        )

with sell_tab:
    if sell_top10.empty:
        st.info("No companies meet the current SELL criteria.")
    else:
        st.dataframe(
            display_table(sell_top10),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Top 10 SELL",
            data=sell_top10.to_csv(index=False).encode("utf-8"),
            file_name="top_10_sell.csv",
            mime="text/csv",
            key="download_sell",
        )


# ============================================================
# COMPANY EXPLORER
# ============================================================

st.header("Company Explorer")

search_text = st.text_input(
    "Search by company name or ticker",
    placeholder="Type a company name or ticker...",
)

if search_text.strip():
    match_mask = (
        scored_data["Ticker"].str.contains(
            search_text, case=False, na=False, regex=False
        )
        | scored_data["Name"].str.contains(
            search_text, case=False, na=False, regex=False
        )
    )

    matches = scored_data[match_mask].copy()

    if matches.empty:
        st.info("No matching companies found in the current universe.")
    else:
        st.dataframe(
            display_table(matches),
            use_container_width=True,
            hide_index=True,
        )

        tickers = matches["Ticker"].drop_duplicates().tolist()

        selected_ticker = st.selectbox(
            "Select a company",
            options=tickers,
            format_func=lambda ticker: (
                f"{ticker} — "
                f"{matches.loc[matches['Ticker'] == ticker, 'Name'].iloc[0]}"
            ),
        )

        selected_rows = matches[
            matches["Ticker"] == selected_ticker
        ]

        if not selected_rows.empty:
            company = selected_rows.iloc[0]

            st.subheader(f"{company['Name']} ({company['Ticker']})")

            details = st.columns(4)
            details[0].metric(
                "Market Cap",
                format_market_cap(company["Market Cap"]),
            )
            details[1].metric(
                "GPA",
                format_number(company["Gross_Profitability"]),
            )
            details[2].metric("STA", format_number(company["STA"]))
            details[3].metric("SNOA", format_number(company["SNOA"]))

            st.write(
                f"**Altman Z-Score:** "
                f"{format_number(company['Altman_Z'])}"
            )
            st.write(
                f"**BUY signals:** {int(company['Buy Signals'])} | "
                f"**SELL signals:** {int(company['Sell Signals'])} | "
                f"**Classification:** {company['Classification']}"
            )

            detail_table = pd.DataFrame(
                {
                    "Screen": ["GPA", "STA", "SNOA", "Altman Z"],
                    "Metric value": [
                        company["Gross_Profitability"],
                        company["STA"],
                        company["SNOA"],
                        company["Altman_Z"],
                    ],
                    "BUY signal": [
                        bool(company["GPA_BUY"]),
                        bool(company["STA_BUY"]),
                        bool(company["SNOA_BUY"]),
                        bool(company["ALTMAN_BUY"]),
                    ],
                    "SELL signal": [
                        bool(company["GPA_SELL"]),
                        bool(company["STA_SELL"]),
                        bool(company["SNOA_SELL"]),
                        bool(company["ALTMAN_SELL"]),
                    ],
                }
            )

            detail_table["Metric value"] = detail_table[
                "Metric value"
            ].apply(
                lambda value: (
                    "N/A" if pd.isna(value)
                    else format_number(value)
                )
            )

            st.dataframe(
                detail_table,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# METHODOLOGY AND COVERAGE
# ============================================================

st.header("Methodology and Data Coverage")

st.markdown(
    """
| Screen | BUY signal | SELL signal |
|---|---|---|
| GPA | Highest selected percentile | Lowest selected percentile |
| STA | Lowest selected percentile | Highest selected percentile |
| SNOA | Lowest selected percentile | Highest selected percentile |
| Altman Z-Score | Above the safe-zone threshold | Below the distress threshold |

Missing values are not converted to zero. They receive no signal and are
excluded from the relevant percentile calculation.

The percentile rules can flag more firms than the selected percentage when
multiple companies tie at a boundary. The Altman thresholds apply strict
greater-than and less-than comparisons.

These results depend on the dataset, reporting dates, definitions, and
selected thresholds. The app uses the supplied Altman Z-Score and SNOA
values rather than silently recalculating them.
"""
)

coverage_rows = []

for metric, label in [
    ("Gross_Profitability", "Gross Profitability (GPA)"),
    ("STA", "Earnings Quality (STA)"),
    ("SNOA", "Operating Assets (SNOA)"),
    ("Altman_Z", "Altman Z-Score"),
]:
    available = int(data[metric].notna().sum())
    missing = int(data[metric].isna().sum())

    coverage_rows.append(
        {
            "Metric": label,
            "Available observations": available,
            "Missing observations": missing,
            "Coverage (%)": round(100 * available / len(data), 1),
        }
    )

st.dataframe(
    pd.DataFrame(coverage_rows),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "This tool provides quantitative screening outputs for research "
    "and education. It does not guarantee performance or replace "
    "independent financial analysis."
)
