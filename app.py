
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Manufacturing Stock Screening App",
    page_icon="📊",
    layout="wide",
)

st.title("Manufacturing Stock Screening & Analysis")

st.markdown(
    """
## Investment universe: Top manufacturing firms

This analysis begins with the manufacturing firms in the supplied
`universe.csv`, which is ranked by market capitalization in the source
dataset. The app screens this defined group rather than the entire stock
market.

Four financial measures are used:

- **STA:** earnings quality and the relationship between net income and
  operating cash flow.
- **SNOA:** net operating assets relative to total assets.
- **Altman Z-Score:** an indicator of financial-distress risk.
- **GPA:** gross profitability relative to total assets.

The individual screens are combined to identify the Top 10 BUY and Top 10
SELL candidates under the selected rules.

These are quantitative screening results, not guarantees of future returns
or personalized investment recommendations.
"""
)


# ============================================================
# SESSION STATE AND SYNCHRONIZED SLIDERS
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

SLIDER_WIDGETS = {
    "min_market_cap": ["cap_sidebar", "cap_main"],
    "gpa_pct": ["gpa_sidebar", "gpa_main"],
    "sta_pct": ["sta_sidebar", "sta_main"],
    "snoa_pct": ["snoa_sidebar", "snoa_main"],
    "safe_z": ["safe_sidebar", "safe_main"],
    "distress_z": ["distress_sidebar", "distress_main"],
    "min_signals": ["signals_sidebar", "signals_main"],
}


def initialise_state():
    for setting, default in DEFAULTS.items():
        if setting not in st.session_state:
            st.session_state[setting] = default

        for widget_key in SLIDER_WIDGETS[setting]:
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state[setting]


def sync_slider(setting, source_key):
    value = st.session_state[source_key]
    st.session_state[setting] = value

    for widget_key in SLIDER_WIDGETS[setting]:
        st.session_state[widget_key] = value


initialise_state()


# ============================================================
# MARKET CAPITALIZATION PARSING
# ============================================================

def parse_market_cap(value):
    """Convert values such as $2.4T, $850B, and $45.2M to numbers."""
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
        "T": 1_000_000_000_000,
        "B": 1_000_000_000,
        "M": 1_000_000,
        "K": 1_000,
    }

    word_multipliers = {
        "trillion": 1_000_000_000_000,
        "billion": 1_000_000_000,
        "million": 1_000_000,
        "thousand": 1_000,
    }

    multiplier = 1
    lowered = text.lower()

    for word, factor in word_multipliers.items():
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


def fmt_market_cap(value):
    if pd.isna(value):
        return "N/A"

    absolute = abs(value)

    if absolute >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:,.2f}T"
    if absolute >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"

    return f"${value:,.0f}"


def fmt_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv("universe.csv")
    data.columns = data.columns.str.strip()

    if "Market Cap" in data.columns:
        data["Market Cap"] = data["Market Cap"].apply(parse_market_cap)

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
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "Ticker" not in data.columns:
        if "Ticker_guess" in data.columns:
            data["Ticker"] = data["Ticker_guess"]
        else:
            data["Ticker"] = ""

    if "Name" not in data.columns:
        data["Name"] = data["Ticker"]

    data["Ticker"] = data["Ticker"].fillna("").astype(str)
    data["Name"] = data["Name"].fillna(data["Ticker"]).astype(str)

    return data.drop_duplicates().reset_index(drop=True)


try:
    df_original = load_data()
except FileNotFoundError:
    st.error(
        "Could not find universe.csv. Ensure it is in the same repository "
        "folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load universe.csv: {error}")
    st.stop()


# ============================================================
# TABLE DISPLAY HELPERS
# ============================================================

def make_display_table(data, columns=None):
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

    columns = [column for column in columns if column in data.columns]
    result = data[columns].copy()

    rename = {
        "Rank": "Universe Rank",
        "Name": "Company",
        "Market Cap": "Market Cap (USD)",
        "Price": "Share Price",
        "Gross_Profitability": "GPA",
        "Altman_Z": "Altman Z-Score",
    }

    result = result.rename(columns=rename)

    if "Market Cap (USD)" in result.columns:
        result["Market Cap (USD)"] = result["Market Cap (USD)"].apply(
            fmt_market_cap
        )

    return result


def metric_coverage(data, metric):
    if metric not in data.columns:
        return 0, len(data)

    return int(data[metric].notna().sum()), int(data[metric].isna().sum())


# ============================================================
# PERCENTILE SIGNALS
# ============================================================

def percentile_flags(series, percentile, higher_is_better):
    """
    Calculate BUY and SELL signals using valid observations only.
    Missing observations never receive a signal.
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
st.sidebar.caption(
    "Change settings here or within each step. Corresponding sliders "
    "are synchronized."
)

if "Market Cap" not in df_original.columns:
    st.error("The dataset does not contain a Market Cap column.")
    st.stop()

market_caps = df_original["Market Cap"].dropna()

if market_caps.empty:
    st.error(
        "No market-cap values could be read from universe.csv. "
        "Check the market-cap format in the source file."
    )
    st.stop()

cap_min = 0.0
cap_max = float(market_caps.max())
cap_step = max(cap_max / 1000, 1_000_000.0)

if cap_max <= cap_min:
    st.error("The dataset does not contain usable market-cap values.")
    st.stop()

st.sidebar.subheader("1. Investment Universe")
st.sidebar.slider(
    "Minimum market capitalization (USD)",
    min_value=cap_min,
    max_value=cap_max,
    step=float(cap_step),
    key="cap_sidebar",
    on_change=sync_slider,
    args=("min_market_cap", "cap_sidebar"),
)
st.sidebar.caption(
    "Current minimum: "
    + fmt_market_cap(st.session_state["min_market_cap"])
)

st.sidebar.subheader("2. Earnings Quality — STA")
st.sidebar.slider(
    "STA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    key="sta_sidebar",
    on_change=sync_slider,
    args=("sta_pct", "sta_sidebar"),
)

st.sidebar.subheader("3. Operating Assets — SNOA")
st.sidebar.slider(
    "SNOA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    key="snoa_sidebar",
    on_change=sync_slider,
    args=("snoa_pct", "snoa_sidebar"),
)

st.sidebar.subheader("4. Altman Z-Score")
st.sidebar.slider(
    "Safe-zone threshold",
    min_value=2.0,
    max_value=4.0,
    step=0.01,
    key="safe_sidebar",
    on_change=sync_slider,
    args=("safe_z", "safe_sidebar"),
)
st.sidebar.slider(
    "Distress threshold",
    min_value=0.5,
    max_value=2.5,
    step=0.01,
    key="distress_sidebar",
    on_change=sync_slider,
    args=("distress_z", "distress_sidebar"),
)

st.sidebar.subheader("5. Gross Profitability — GPA")
st.sidebar.slider(
    "GPA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    key="gpa_sidebar",
    on_change=sync_slider,
    args=("gpa_pct", "gpa_sidebar"),
)

st.sidebar.subheader("6. Combined Classification")
st.sidebar.slider(
    "Minimum signals required",
    min_value=1,
    max_value=4,
    key="signals_sidebar",
    on_change=sync_slider,
    args=("min_signals", "signals_sidebar"),
)


# ============================================================
# FILTER THE INVESTMENT UNIVERSE
# ============================================================

data = df_original[
    df_original["Market Cap"].notna()
    & (df_original["Market Cap"] >= st.session_state["min_market_cap"])
].copy()

if data.empty:
    st.warning(
        "No companies remain under this market-cap setting. "
        "Lower the minimum market capitalization."
    )
    st.stop()

if st.session_state["distress_z"] >= st.session_state["safe_z"]:
    st.error(
        "The distress threshold must be below the safe-zone threshold. "
        "Adjust the Altman sliders."
    )
    st.stop()


# ============================================================
# SCORE COMPANIES BEFORE DISPLAYING ANY SCREEN
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

    # GPA: higher is better.
    result["GPA_BUY"], result["GPA_SELL"], _, _ = percentile_flags(
        result["Gross_Profitability"],
        st.session_state["gpa_pct"],
        higher_is_better=True,
    )

    # STA: lower is better.
    result["STA_BUY"], result["STA_SELL"], _, _ = percentile_flags(
        result["STA"],
        st.session_state["sta_pct"],
        higher_is_better=False,
    )

    # SNOA: lower is better.
    result["SNOA_BUY"], result["SNOA_SELL"], _, _ = percentile_flags(
        result["SNOA"],
        st.session_state["snoa_pct"],
        higher_is_better=False,
    )

    # Altman Z-Score: safe scores receive BUY, distress scores receive SELL.
    result["ALTMAN_BUY"] = (
        result["Altman_Z"].notna()
        & (result["Altman_Z"] > st.session_state["safe_z"])
    )

    result["ALTMAN_SELL"] = (
        result["Altman_Z"].notna()
        & (result["Altman_Z"] < st.session_state["distress_z"])
    )

    buy_columns = [
        "GPA_BUY",
        "STA_BUY",
        "SNOA_BUY",
        "ALTMAN_BUY",
    ]

    sell_columns = [
        "GPA_SELL",
        "STA_SELL",
        "SNOA_SELL",
        "ALTMAN_SELL",
    ]

    result["Buy Signals"] = result[buy_columns].sum(axis=1)
    result["Sell Signals"] = result[sell_columns].sum(axis=1)
    result["Classification"] = "NEUTRAL"

    minimum = int(st.session_state["min_signals"])

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


# IMPORTANT: all screen tables use scored_data, which has signal columns.
scored_data = score_companies(data)


# ============================================================
# STEP 1 — INVESTMENT UNIVERSE
# ============================================================

st.header("Step 1 — Define the Investment Universe")

st.markdown(
    """
The source dataset contains the manufacturing firms included in this
analysis. The minimum market-cap slider determines which companies enter
the later screens. It is a universe filter, not a BUY or SELL signal.
"""
)

st.slider(
    "Minimum market capitalization for this step (USD)",
    min_value=cap_min,
    max_value=cap_max,
    step=float(cap_step),
    key="cap_main",
    on_change=sync_slider,
    args=("min_market_cap", "cap_main"),
)

col1, col2, col3 = st.columns(3)
col1.metric("Firms in source dataset", f"{len(df_original):,}")
col2.metric("Firms in current universe", f"{len(data):,}")
col3.metric(
    "Excluded by size filter",
    f"{len(df_original) - len(data):,}",
)

st.dataframe(
    make_display_table(data).head(20),
    use_container_width=True,
    hide_index=True,
)

with st.expander("View all firms in the current universe"):
    st.dataframe(
        make_display_table(data),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# GENERIC METRIC SCREEN — FIXED ARGUMENTS AND SIGNAL COLUMNS
# ============================================================

def show_metric_table(data_with_signals, metric, buy_col, sell_col):
    """
    Display a screen using the scored dataframe, not the unscored dataframe.
    This avoids the KeyError for STA_BUY, STA_SELL, etc.
    """
    required = [metric, buy_col, sell_col]
    missing_columns = [
        column for column in required
        if column not in data_with_signals.columns
    ]

    if missing_columns:
        st.error(
            "This screen is missing required columns: "
            + ", ".join(missing_columns)
            + ". Confirm that the latest app.py has been committed."
        )
        return

    valid = data_with_signals[
        data_with_signals[metric].notna()
    ].copy()

    missing_count = int(data_with_signals[metric].isna().sum())

    col1, col2 = st.columns(2)
    col1.metric("Firms evaluated in this screen", f"{len(valid):,}")
    col2.metric("Missing metric — excluded", f"{missing_count:,}")

    if valid.empty:
        st.warning("There are no valid observations for this metric.")
        return

    valid["Screen Result"] = np.select(
        [valid[buy_col], valid[sell_col]],
        ["BUY signal", "SELL signal"],
        default="No signal",
    )

    # Sort GPA high-to-low; STA and SNOA low-to-high.
    valid = valid.sort_values(
        metric,
        ascending=(metric != "Gross_Profitability"),
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
        make_display_table(valid, columns),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STEP 2 — STA
# ============================================================

st.header("Step 2 — Screen Earnings Quality Using STA")

st.markdown(
    """
STA measures the difference between net income and operating cash flow
relative to total assets. A large positive difference can indicate that
reported earnings are less strongly supported by operating cash flow.

Under this screening convention, lower STA is preferred. The lowest selected
percentile receives BUY signals, and the highest receives SELL signals.
Missing values receive no signal.
"""
)

st.latex(
    r"\mathrm{STA}="
    r"\frac{\mathrm{Net\ Income}-\mathrm{Operating\ Cash\ Flow}}"
    r"{\mathrm{Total\ Assets}}"
)

st.slider(
    "STA cutoff: percentage of valid firms flagged at each end",
    min_value=1,
    max_value=25,
    key="sta_main",
    on_change=sync_slider,
    args=("sta_pct", "sta_main"),
)

st.caption(
    f"Current cutoff: {st.session_state['sta_pct']}% at each end. "
    "Ties at the boundary can result in more firms being flagged."
)

show_metric_table(scored_data, "STA", "STA_BUY", "STA_SELL")


# ============================================================
# STEP 3 — SNOA
# ============================================================

st.header("Step 3 — Screen Operating Assets Using SNOA")

st.markdown(
    """
SNOA measures net operating assets relative to total assets. Under the
screening convention used here, lower SNOA is preferred and higher SNOA is
less favourable. The app uses the existing SNOA values in the dataset.
Missing values are excluded from the percentile calculation and receive no
signal.
"""
)

st.latex(
    r"\mathrm{SNOA}="
    r"\frac{\mathrm{Operating\ Assets}-\mathrm{Operating\ Liabilities}}"
    r"{\mathrm{Total\ Assets}}"
)

st.slider(
    "SNOA cutoff: percentage of valid firms flagged at each end",
    min_value=1,
    max_value=25,
    key="snoa_main",
    on_change=sync_slider,
    args=("snoa_pct", "snoa_main"),
)

st.caption(
    f"Current cutoff: {st.session_state['snoa_pct']}% at each end."
)

show_metric_table(scored_data, "SNOA", "SNOA_BUY", "SNOA_SELL")


# ============================================================
# STEP 4 — ALTMAN Z-SCORE
# ============================================================

st.header("Step 4 — Assess Financial Distress Using Altman Z-Score")

st.markdown(
    """
The Altman Z-Score combines financial ratios to help assess financial-distress
risk. This app uses the scores already supplied in the dataset.

- Score above the safe-zone threshold: BUY signal from this screen.
- Score below the distress threshold: SELL signal from this screen.
- Score between the thresholds or exactly on either threshold: neutral.
- Missing score: no signal.
"""
)

st.latex(r"Z=1.2X_1+1.4X_2+3.3X_3+0.6X_4+1.0X_5")

st.caption(
    "Existing Altman_Z values are used directly; the app does not silently "
    "recalculate them using potentially different ratio scaling."
)

zcol1, zcol2 = st.columns(2)

with zcol1:
    st.slider(
        "Safe-zone threshold",
        min_value=2.0,
        max_value=4.0,
        step=0.01,
        key="safe_main",
        on_change=sync_slider,
        args=("safe_z", "safe_main"),
    )

with zcol2:
    st.slider(
        "Distress threshold",
        min_value=0.5,
        max_value=2.5,
        step=0.01,
        key="distress_main",
        on_change=sync_slider,
        args=("distress_z", "distress_main"),
    )

if st.session_state["distress_z"] >= st.session_state["safe_z"]:
    st.error("The distress threshold must be below the safe-zone threshold.")
    st.stop()

st.caption(
    f"Current rule: Z > {st.session_state['safe_z']:.2f} is BUY; "
    f"Z < {st.session_state['distress_z']:.2f} is SELL."
)

z_valid = scored_data[scored_data["Altman_Z"].notna()].copy()
z_missing = int(scored_data["Altman_Z"].isna().sum())

z1, z2, z3 = st.columns(3)
z1.metric("Valid scores", f"{len(z_valid):,}")
z2.metric("Missing scores", f"{z_missing:,}")
z3.metric("Distress signals", f"{int(z_valid['ALTMAN_SELL'].sum()):,}")

if not z_valid.empty:
    z_valid["Screen Result"] = np.select(
        [z_valid["ALTMAN_BUY"], z_valid["ALTMAN_SELL"]],
        ["BUY signal", "SELL signal"],
        default="Neutral",
    )

    zcols = [
        column for column in [
            "Rank",
            "Ticker",
            "Name",
            "Market Cap",
            "Altman_Z",
            "Screen Result",
        ]
        if column in z_valid.columns
    ]

    st.dataframe(
        make_display_table(z_valid, zcols),
        use_container_width=True,
        hide_index=True,
    )

    fig_z = px.histogram(
        z_valid,
        x="Altman_Z",
        nbins=35,
        title="Distribution of Altman Z-Scores",
        labels={"Altman_Z": "Altman Z-Score"},
    )

    fig_z.add_vline(
        x=st.session_state["distress_z"],
        line_dash="dash",
        annotation_text="Distress threshold",
    )

    fig_z.add_vline(
        x=st.session_state["safe_z"],
        line_dash="dash",
        annotation_text="Safe-zone threshold",
    )

    st.plotly_chart(fig_z, use_container_width=True)


# ============================================================
# STEP 5 — GPA
# ============================================================

st.header("Step 5 — Screen for Gross Profitability (GPA)")

st.markdown(
    """
GPA measures gross profit relative to total assets. Higher values indicate
greater gross profit relative to the asset base under this definition.

For this screen, higher GPA is preferred. The highest selected percentile
receives a BUY signal, and the lowest selected percentile receives a SELL
signal.
"""
)

st.latex(
    r"\mathrm{GPA}="
    r"\frac{\mathrm{Gross\ Profit}}{\mathrm{Total\ Assets}}"
)

st.slider(
    "GPA cutoff: percentage of valid firms flagged at each end",
    min_value=1,
    max_value=25,
    key="gpa_main",
    on_change=sync_slider,
    args=("gpa_pct", "gpa_main"),
)

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
# STEP 6 — COMBINE SIGNALS
# ============================================================

st.header("Step 6 — Combine the Screening Signals")

st.markdown(
    """
Each company can receive up to four BUY signals and four SELL signals:
one from GPA, STA, SNOA, and Altman Z-Score.

- Missing metrics contribute neither a BUY nor a SELL signal.
- BUY requires the minimum number of BUY signals and more BUY than SELL.
- SELL requires the minimum number of SELL signals and more SELL than BUY.
- Remaining companies are classified NEUTRAL.
"""
)

st.slider(
    "Minimum matching signals required for BUY or SELL",
    min_value=1,
    max_value=4,
    key="signals_main",
    on_change=sync_slider,
    args=("min_signals", "signals_main"),
)

st.caption(
    f"Current rule: at least {st.session_state['min_signals']} signals "
    "in one direction, and more signals in that direction than the opposite."
)

counts = (
    scored_data["Classification"]
    .value_counts()
    .reindex(["BUY", "NEUTRAL", "SELL"], fill_value=0)
)

c1, c2, c3 = st.columns(3)
c1.metric("BUY", f"{counts['BUY']:,}")
c2.metric("NEUTRAL", f"{counts['NEUTRAL']:,}")
c3.metric("SELL", f"{counts['SELL']:,}")

class_chart_data = counts.rename_axis("Classification").reset_index(
    name="Companies"
)

fig_class = px.bar(
    class_chart_data,
    x="Classification",
    y="Companies",
    text="Companies",
    title="Companies by Final Classification",
)

st.plotly_chart(fig_class, use_container_width=True)

signal_chart_data = pd.DataFrame(
    {
        "Screen": ["GPA", "STA", "SNOA", "Altman Z"],
        "BUY signals": [
            int(scored_data["GPA_BUY"].sum()),
            int(scored_data["STA_BUY"].sum()),
            int(scored_data["SNOA_BUY"].sum()),
            int(scored_data["ALTMAN_BUY"].sum()),
        ],
        "SELL signals": [
            int(scored_data["GPA_SELL"].sum()),
            int(scored_data["STA_SELL"].sum()),
            int(scored_data["SNOA_SELL"].sum()),
            int(scored_data["ALTMAN_SELL"].sum()),
        ],
    }
)

fig_signals = px.bar(
    signal_chart_data,
    x="Screen",
    y=["BUY signals", "SELL signals"],
    barmode="group",
    title="BUY and SELL Signals by Screen",
)

st.plotly_chart(fig_signals, use_container_width=True)

st.subheader("Results for all firms")

st.dataframe(
    make_display_table(
        scored_data.sort_values(
            ["Classification", "Buy Signals", "Sell Signals"],
            ascending=[True, False, True],
        )
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download all screening results as CSV",
    data=scored_data.to_csv(index=False).encode("utf-8"),
    file_name="manufacturing_stock_screen_results.csv",
    mime="text/csv",
)


# ============================================================
# STEP 7 — TOP 10 BUY AND TOP 10 SELL
# ============================================================

st.header("Step 7 — Top 10 BUY and Top 10 SELL Candidates")

st.markdown(
    """
These tables rank companies classified by the current screening rules.
They are quantitative screening outputs, not guarantees of future performance.
"""
)

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
    st.caption(
        "Ranked by number of BUY signals, then fewer SELL signals and "
        "supporting financial metrics."
    )

    if buy_top10.empty:
        st.info(
            "No companies currently meet the BUY rule. Review the cutoffs "
            "and minimum-signal setting if appropriate."
        )
    else:
        st.dataframe(
            make_display_table(buy_top10),
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
    st.caption(
        "Ranked by number of SELL signals, then fewer BUY signals and "
        "supporting financial metrics."
    )

    if sell_top10.empty:
        st.info(
            "No companies currently meet the SELL rule. Review the cutoffs "
            "and minimum-signal setting if appropriate."
        )
    else:
        st.dataframe(
            make_display_table(sell_top10),
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

st.markdown(
    "Search by company name or ticker to see its financial metrics, "
    "individual screen signals, and final classification."
)

search_text = st.text_input(
    "Search company name or ticker",
    placeholder="Enter a company name or ticker...",
)

if search_text.strip():
    mask = (
        scored_data["Ticker"].str.contains(
            search_text,
            case=False,
            na=False,
            regex=False,
        )
        | scored_data["Name"].str.contains(
            search_text,
            case=False,
            na=False,
            regex=False,
        )
    )

    matches = scored_data[mask].copy()

    if matches.empty:
        st.info("No companies matched that search in the current universe.")
    else:
        st.dataframe(
            make_display_table(matches),
            use_container_width=True,
            hide_index=True,
        )

        ticker_options = matches["Ticker"].drop_duplicates().tolist()

        selected_ticker = st.selectbox(
            "Choose a company for detailed signal information",
            options=ticker_options,
            format_func=lambda ticker: (
                f"{ticker} — "
                f"{matches.loc[matches['Ticker'] == ticker, 'Name'].iloc[0]}"
            ),
        )

        selected_rows = matches[matches["Ticker"] == selected_ticker]

        if not selected_rows.empty:
            company = selected_rows.iloc[0]

            st.subheader(f"{company['Name']} ({company['Ticker']})")

            detail_cols = st.columns(4)
            detail_cols[0].metric(
                "Market Cap",
                fmt_market_cap(company["Market Cap"]),
            )
            detail_cols[1].metric(
                "GPA",
                fmt_number(company["Gross_Profitability"], 4),
            )
            detail_cols[2].metric("STA", fmt_number(company["STA"], 4))
            detail_cols[3].metric("SNOA", fmt_number(company["SNOA"], 4))

            st.write(
                f"**Altman Z-Score:** {fmt_number(company['Altman_Z'])}"
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

            detail_table["Metric value"] = detail_table["Metric value"].apply(
                lambda value: (
                    "N/A" if pd.isna(value) else fmt_number(value, 4)
                )
            )

            st.dataframe(
                detail_table,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# METHODOLOGY AND DATA COVERAGE
# ============================================================

st.header("Methodology and Data Coverage")

st.markdown(
    """
### Screening conventions

| Screen | BUY signal | SELL signal |
|---|---|---|
| GPA | Highest selected percentile | Lowest selected percentile |
| STA | Lowest selected percentile | Highest selected percentile |
| SNOA | Lowest selected percentile | Highest selected percentile |
| Altman Z-Score | Above safe-zone threshold | Below distress threshold |

A score exactly on either Altman threshold is neutral. Percentile cutoffs can
flag more than the selected share of firms when multiple companies have
identical values at the boundary.

### Missing data

Missing metrics are not converted to zero. A missing value cannot generate a
BUY or SELL signal and is excluded from that metric's percentile calculation.

### Limitations

Results depend on the supplied dataset, metric definitions, reporting dates,
and selected thresholds. Companies may report financial information on
different dates. These rankings are a screening tool and are not guarantees
of future performance.
"""
)

coverage_rows = []

for metric, label in [
    ("Gross_Profitability", "Gross Profitability (GPA)"),
    ("STA", "Earnings Quality (STA)"),
    ("SNOA", "Operating Assets (SNOA)"),
    ("Altman_Z", "Altman Z-Score"),
]:
    available, missing = metric_coverage(data, metric)

    coverage_rows.append(
        {
            "Metric": label,
            "Available observations": available,
            "Missing observations": missing,
            "Coverage (%)": (
                round(100 * available / len(data), 1)
                if len(data)
                else 0
            ),
        }
    )

st.dataframe(
    pd.DataFrame(coverage_rows),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Market capitalization supports suffixes such as K, M, B, and T. "
    "Unrecognized values remain missing rather than being treated as zero."
)
