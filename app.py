
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE SETUP
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

    This analysis begins with the **top manufacturing firms included in
    the supplied investment universe**, ranked by market capitalization
    in the source dataset. The purpose is to screen this defined group
    systematically rather than search the entire stock market.

    Each company is assessed through several separate financial screens:
    earnings quality (STA), operating assets (SNOA), financial distress
    (Altman Z-Score), and gross profitability (GPA). The results are then
    combined into a transparent set of BUY, SELL, and NEUTRAL signals.

    **Important:** A screen is a quantitative filter, not a guarantee of
    future returns or a personalized investment recommendation.
    """
)

st.caption(
    "Use the controls in the sidebar or the matching controls within each "
    "step. Both sets stay synchronized."
)


# ============================================================
# CONSTANTS AND SETTINGS
# ============================================================

DATA_FILE = "universe.csv"

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
    """Set initial values for synchronized slider widgets."""
    for setting, default in DEFAULTS.items():
        if setting not in st.session_state:
            st.session_state[setting] = default

        for widget_key in SLIDER_WIDGETS[setting]:
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state[setting]


def sync_slider(setting, source_key):
    """Synchronize both slider widgets through a shared setting."""
    value = st.session_state[source_key]
    st.session_state[setting] = value

    for widget_key in SLIDER_WIDGETS[setting]:
        st.session_state[widget_key] = value


initialise_state()


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

def parse_market_cap(value):
    """
    Convert market-cap strings such as:
      $2.4T -> 2,400,000,000,000
      $850B -> 850,000,000,000
      $45.2M -> 45,200,000

    Already-numeric values are kept as numeric values.
    Missing or unrecognized values remain missing.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    text = text.replace(",", "").replace("$", "").replace(" ", "")

    if text.lower() in {"none", "nan", "n/a", "na", "null", "-", "--"}:
        return np.nan

    multipliers = {
        "T": 1_000_000_000_000,
        "B": 1_000_000_000,
        "M": 1_000_000,
        "K": 1_000,
    }

    suffix = text[-1].upper()
    multiplier = multipliers.get(suffix, 1)

    if suffix in multipliers:
        text = text[:-1]

    # Also tolerate values such as "2.4 trillion" or "850 billion".
    lowered = text.lower()
    word_multipliers = {
        "trillion": 1_000_000_000_000,
        "billion": 1_000_000_000,
        "million": 1_000_000,
        "thousand": 1_000,
    }

    for word, word_multiplier in word_multipliers.items():
        if lowered.endswith(word):
            text = text[:-len(word)]
            multiplier = word_multiplier
            break

    try:
        return float(text) * multiplier
    except (TypeError, ValueError):
        return np.nan


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data.columns = data.columns.str.strip()

    # Parse market capitalization separately. Applying pd.to_numeric()
    # directly to "$2.4T" or "$850B" would turn those values into NaN.
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

    # Use ticker/name fields that exist in the dataset.
    if "Ticker" not in data.columns:
        if "Ticker_guess" in data.columns:
            data["Ticker"] = data["Ticker_guess"]
        else:
            data["Ticker"] = ""

    if "Name" not in data.columns:
        data["Name"] = data["Ticker"]

    data["Ticker"] = data["Ticker"].fillna("").astype(str)
    data["Name"] = data["Name"].fillna(data["Ticker"]).astype(str)

    # Remove exact duplicate rows, but do not remove companies simply
    # because one of their financial metrics is missing.
    data = data.drop_duplicates().reset_index(drop=True)

    return data


try:
    df_original = load_data()
except FileNotFoundError:
    st.error(
        "Could not find universe.csv. Make sure it is in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load universe.csv: {error}")
    st.stop()


# ============================================================
# GENERAL HELPERS
# ============================================================

def fmt_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


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


def find_column(data, possibilities):
    for column in possibilities:
        if column in data.columns:
            return column
    return None


def metric_coverage(data, column):
    if column not in data.columns:
        return 0, len(data)
    return int(data[column].notna().sum()), int(data[column].isna().sum())


def percentile_flags(series, percentile, higher_is_better):
    """
    Return BUY and SELL boolean Series.

    Missing values never trigger a signal. Quantile cutoffs are calculated
    only from nonmissing values. Ties can cause more than exactly the
    requested percentage of firms to be flagged.
    """
    buy = pd.Series(False, index=series.index)
    sell = pd.Series(False, index=series.index)

    valid = series.dropna()
    if valid.empty:
        return buy, sell, np.nan, np.nan

    low_cutoff = valid.quantile(percentile / 100)
    high_cutoff = valid.quantile(1 - percentile / 100)

    if higher_is_better:
        buy.loc[series.notna()] = series.loc[series.notna()] >= high_cutoff
        sell.loc[series.notna()] = series.loc[series.notna()] <= low_cutoff
    else:
        buy.loc[series.notna()] = series.loc[series.notna()] <= low_cutoff
        sell.loc[series.notna()] = series.loc[series.notna()] >= high_cutoff

    return buy, sell, low_cutoff, high_cutoff


def add_signal_columns(data):
    """Create signal columns while keeping missing metrics unflagged."""
    result = data.copy()

    for column in ["STA", "SNOA", "Gross_Profitability", "Altman_Z"]:
        if column not in result.columns:
            result[column] = np.nan

    # Independent percentile cutoffs for each metric.
    gpa_buy, gpa_sell, gpa_low, gpa_high = percentile_flags(
        result["Gross_Profitability"],
        st.session_state["gpa_pct"],
        higher_is_better=True,
    )

    sta_buy, sta_sell, sta_low, sta_high = percentile_flags(
        result["STA"],
        st.session_state["sta_pct"],
        higher_is_better=False,
    )

    snoa_buy, snoa_sell, snoa_low, snoa_high = percentile_flags(
        result["SNOA"],
        st.session_state["snoa_pct"],
        higher_is_better=False,
    )

    result["GPA_BUY"] = gpa_buy
    result["GPA_SELL"] = gpa_sell
    result["STA_BUY"] = sta_buy
    result["STA_SELL"] = sta_sell
    result["SNOA_BUY"] = snoa_buy
    result["SNOA_SELL"] = snoa_sell

    # Altman Z-Score:
    # Above the safe threshold = BUY signal.
    # Below the distress threshold = SELL signal.
    # Between thresholds, including the boundaries = neutral.
    safe_threshold = st.session_state["safe_z"]
    distress_threshold = st.session_state["distress_z"]

    result["ALTMAN_BUY"] = (
        result["Altman_Z"].notna()
        & (result["Altman_Z"] > safe_threshold)
    )
    result["ALTMAN_SELL"] = (
        result["Altman_Z"].notna()
        & (result["Altman_Z"] < distress_threshold)
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

    minimum_signals = int(st.session_state["min_signals"])

    result["Classification"] = "NEUTRAL"

    buy_mask = (
        (result["Buy Signals"] >= minimum_signals)
        & (result["Buy Signals"] > result["Sell Signals"])
    )
    sell_mask = (
        (result["Sell Signals"] >= minimum_signals)
        & (result["Sell Signals"] > result["Buy Signals"])
    )

    result.loc[buy_mask, "Classification"] = "BUY"
    result.loc[sell_mask, "Classification"] = "SELL"

    return result, {
        "gpa_low": gpa_low,
        "gpa_high": gpa_high,
        "sta_low": sta_low,
        "sta_high": sta_high,
        "snoa_low": snoa_low,
        "snoa_high": snoa_high,
    }


def display_columns(data):
    """Choose available columns for readable company tables."""
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
        "Buy Signals",
        "Sell Signals",
        "Classification",
    ]
    return [column for column in preferred if column in data.columns]


def prettify_table(data):
    """Rename financial columns for display without changing calculations."""
    rename = {
        "Rank": "Universe Rank",
        "Ticker": "Ticker",
        "Name": "Company",
        "Market Cap": "Market Cap (USD)",
        "Price": "Share Price",
        "Gross_Profitability": "GPA",
        "STA": "STA",
        "SNOA": "SNOA",
        "Altman_Z": "Altman Z-Score",
    }
    return data.rename(columns=rename)


def show_metric_table(data, metric, buy_col, sell_col, label):
    if metric not in data.columns:
        st.warning(f"The dataset does not contain the {label} column.")
        return

    valid = data[data[metric].notna()].copy()
    missing = int(data[metric].isna().sum())

    left, right = st.columns(2)
    left.metric("Firms evaluated in this screen", f"{len(valid):,}")
    right.metric("Firms excluded: metric missing", f"{missing:,}")

    if valid.empty:
        st.warning("No companies have valid data for this metric.")
        return

    valid["Screen Result"] = np.select(
        [
            valid[buy_col],
            valid[sell_col],
        ],
        [
            "BUY signal",
            "SELL signal",
        ],
        default="No signal",
    )

    valid = valid.sort_values(
        metric,
        ascending=(buy_col.endswith("BUY") is False),
        na_position="last",
    )

    show = [
        column for column in
        ["Ticker", "Name", "Market Cap", metric, "Screen Result"]
        if column in valid.columns
    ]

    st.dataframe(
        prettify_table(valid[show]),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("Screening Controls")
st.sidebar.caption(
    "Adjust the rules here or in the matching step below. "
    "The two sets of sliders are synchronized."
)

cap_values = df_original["Market Cap"].dropna() if "Market Cap" in df_original else pd.Series(dtype=float)

if len(cap_values):
    cap_max = float(cap_values.max())
    cap_min = float(cap_values.min())
    cap_step = max(cap_max / 1000, 1_000_000)
else:
    cap_min = 0.0
    cap_max = 1_000_000_000_000
    cap_step = 1_000_000_000

st.sidebar.subheader("1. Investment Universe")
st.sidebar.slider(
    "Minimum market capitalization (USD)",
    min_value=float(max(0, cap_min)),
    max_value=float(max(cap_min, cap_max)),
    step=float(cap_step),
    key="cap_sidebar",
    on_change=sync_slider,
    args=("min_market_cap", "cap_sidebar"),
    help="Firms below this market cap are removed from the screening universe.",
)
st.sidebar.caption(
    f"Current minimum: {fmt_market_cap(st.session_state['min_market_cap'])}"
)

st.sidebar.subheader("2. Earnings Quality — STA")
st.sidebar.slider(
    "STA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    step=1,
    key="sta_sidebar",
    on_change=sync_slider,
    args=("sta_pct", "sta_sidebar"),
    help="Low STA is treated as a positive signal; high STA as a negative signal.",
)

st.sidebar.subheader("3. Operating Assets — SNOA")
st.sidebar.slider(
    "SNOA percentile cutoff (%)",
    min_value=1,
    max_value=25,
    step=1,
    key="snoa_sidebar",
    on_change=sync_slider,
    args=("snoa_pct", "snoa_sidebar"),
    help="Low SNOA is treated as a positive signal; high SNOA as a negative signal.",
)

st.sidebar.subheader("4. Financial Distress — Altman Z")
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
    step=1,
    key="gpa_sidebar",
    on_change=sync_slider,
    args=("gpa_pct", "gpa_sidebar"),
    help="High GPA is treated as a positive signal; low GPA as a negative signal.",
)

st.sidebar.subheader("6. Combined Classification")
st.sidebar.slider(
    "Minimum signals required",
    min_value=1,
    max_value=4,
    step=1,
    key="signals_sidebar",
    on_change=sync_slider,
    args=("min_signals", "signals_sidebar"),
)

if st.session_state["distress_z"] >= st.session_state["safe_z"]:
    st.sidebar.error(
        "The distress threshold must be below the safe-zone threshold."
    )


# ============================================================
# APPLY THE INVESTMENT UNIVERSE FILTER
# ============================================================

data = df_original.copy()

if "Market Cap" not in data.columns:
    st.error("The dataset does not contain a Market Cap column.")
    st.stop()

# Missing market cap cannot satisfy a minimum market-cap filter.
data = data[
    data["Market Cap"].notna()
    & (data["Market Cap"] >= st.session_state["min_market_cap"])
].copy()

if data.empty:
    st.warning(
        "No companies remain under the current market-cap setting. "
        "Lower the minimum market capitalization."
    )
    st.stop()

# Avoid classifying companies using contradictory Altman thresholds.
if st.session_state["distress_z"] >= st.session_state["safe_z"]:
    st.error(
        "Please set the distress threshold below the safe-zone threshold "
        "using the sidebar sliders."
    )
    st.stop()

scored_data, cutoffs = add_signal_columns(data)


# ============================================================
# STEP 1 — INVESTMENT UNIVERSE
# ============================================================

st.header("Step 1 — Define the Investment Universe")

st.markdown(
    """
    **What this step does**

    The starting universe is the group of top manufacturing firms supplied
    in `universe.csv`. Market capitalization is used to exclude firms below
    a chosen size threshold.

    This is a universe-selection rule, not a BUY or SELL signal. A company
    that fails this filter is not evaluated in the later screens.
    """
)

st.slider(
    "Minimum market capitalization for this step (USD)",
    min_value=float(max(0, cap_min)),
    max_value=float(max(cap_min, cap_max)),
    step=float(cap_step),
    key="cap_main",
    on_change=sync_slider,
    args=("min_market_cap", "cap_main"),
    help="Only firms at or above this market cap enter the later screening steps.",
)

m1, m2, m3 = st.columns(3)
m1.metric("Starting firms in dataset", f"{len(df_original):,}")
m2.metric("Firms in current universe", f"{len(data):,}")
m3.metric("Firms excluded by universe filter", f"{len(df_original) - len(data):,}")

st.dataframe(
    prettify_table(
        data[display_columns(data)].head(20)
    ),
    use_container_width=True,
    hide_index=True,
)

with st.expander("View all companies in the current universe"):
    st.dataframe(
        prettify_table(data[display_columns(data)]),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STEP 2 — STA
# ============================================================

st.header("Step 2 — Screen Earnings Quality Using STA")

st.markdown(
    """
    **What is STA?**

    STA is used here as an earnings-quality measure based on the difference
    between net income and operating cash flow, scaled by total assets.

    A larger positive difference can indicate that reported earnings are
    less strongly supported by operating cash flow. Under this screening
    convention, **lower STA is preferred**.

    The dataset's existing STA values are used. Missing STA values do not
    receive either signal and are excluded from this screen's cutoff
    calculation.
    """
)

st.latex(r"\mathrm{STA}=\frac{\mathrm{Net\ Income}-\mathrm{Operating\ Cash\ Flow}}{\mathrm{Total\ Assets}}")

st.slider(
    "STA cutoff: flag the lowest and highest percentiles",
    min_value=1,
    max_value=25,
    step=1,
    key="sta_main",
    on_change=sync_slider,
    args=("sta_pct", "sta_main"),
    help="Lower STA values receive BUY signals; higher STA values receive SELL signals.",
)

st.write(
    f"Current rule: the lowest {st.session_state['sta_pct']}% of valid STA "
    f"values are BUY signals, and the highest {st.session_state['sta_pct']}% "
    "are SELL signals. Ties at the cutoff can cause the actual number flagged "
    "to exceed the nominal percentile."
)

show_metric_table(data, "STA", "STA_BUY", "STA_SELL", "STA")


# ============================================================
# STEP 3 — SNOA
# ============================================================

st.header("Step 3 — Screen Operating Assets Using SNOA")

st.markdown(
    """
    **What is SNOA?**

    SNOA is the operating-assets measure already supplied in the dataset.
    For this screen, lower values are treated as more favourable and higher
    values as less favourable.

    Only nonmissing SNOA observations determine the percentile cutoffs.
    Missing observations are counted separately and do not generate signals.
    """
)

st.latex(
    r"\mathrm{SNOA}=\frac{\mathrm{Operating\ Assets}-\mathrm{Operating\ Liabilities}}{\mathrm{Total\ Assets}}"
)

st.slider(
    "SNOA cutoff: flag the lowest and highest percentiles",
    min_value=1,
    max_value=25,
    step=1,
    key="snoa_main",
    on_change=sync_slider,
    args=("snoa_pct", "snoa_main"),
    help="Lower SNOA values receive BUY signals; higher SNOA values receive SELL signals.",
)

st.write(
    f"Current rule: the lowest {st.session_state['snoa_pct']}% of valid SNOA "
    f"values are BUY signals, and the highest {st.session_state['snoa_pct']}% "
    "are SELL signals."
)

show_metric_table(data, "SNOA", "SNOA_BUY", "SNOA_SELL", "SNOA")


# ============================================================
# STEP 4 — ALTMAN Z-SCORE
# ============================================================

st.header("Step 4 — Assess Financial Distress Using Altman Z-Score")

st.markdown(
    """
    **What does the Altman Z-Score do?**

    The Altman Z-Score combines financial ratios to assess a firm's
    financial-distress risk. The dataset supplies the calculated score.

    For this manufacturing-firm screen:
    - **Above the safe-zone threshold:** BUY signal from this screen.
    - **Below the distress threshold:** SELL signal from this screen.
    - **Between the thresholds, including the boundaries:** no signal.
    - **Missing score:** no signal; the company is counted as missing for
      this screen.
    """
)

st.latex(
    r"Z=1.2X_1+1.4X_2+3.3X_3+0.6X_4+1.0X_5"
)

st.caption(
    "The app uses the existing Altman_Z values from the dataset; it does not "
    "recalculate component ratios with potentially different scaling."
)

z1, z2 = st.columns(2)

with z1:
    st.slider(
        "Safe-zone threshold",
        min_value=2.0,
        max_value=4.0,
        step=0.01,
        key="safe_main",
        on_change=sync_slider,
        args=("safe_z", "safe_main"),
        help="A score strictly above this value gets an Altman BUY signal.",
    )

with z2:
    st.slider(
        "Distress threshold",
        min_value=0.5,
        max_value=2.5,
        step=0.01,
        key="distress_main",
        on_change=sync_slider,
        args=("distress_z", "distress_main"),
        help="A score strictly below this value gets an Altman SELL signal.",
    )

if st.session_state["distress_z"] >= st.session_state["safe_z"]:
    st.error(
        "The distress threshold must be lower than the safe-zone threshold."
    )
    st.stop()

st.write(
    f"Current rule: Z > {st.session_state['safe_z']:.2f} gives a BUY signal; "
    f"Z < {st.session_state['distress_z']:.2f} gives a SELL signal; "
    "the remaining scores are neutral."
)

z_valid = data[data["Altman_Z"].notna()].copy()
z_missing = int(data["Altman_Z"].isna().sum())

za, zb, zc = st.columns(3)
za.metric("Valid Altman scores", f"{len(z_valid):,}")
zb.metric("Missing Altman scores", f"{z_missing:,}")
zc.metric(
    "Distress-flagged firms",
    f"{int(z_valid['ALTMAN_SELL'].sum()):,}",
)

if not z_valid.empty:
    z_valid["Screen Result"] = np.select(
        [z_valid["ALTMAN_BUY"], z_valid["ALTMAN_SELL"]],
        ["BUY signal", "SELL signal"],
        default="Neutral",
    )

    st.dataframe(
        prettify_table(
            z_valid[
                [
                    column for column in
                    ["Ticker", "Name", "Market Cap", "Altman_Z", "Screen Result"]
                    if column in z_valid.columns
                ]
            ]
        ),
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
        annotation_text="Safe threshold",
    )
    st.plotly_chart(fig_z, use_container_width=True)


# ============================================================
# STEP 5 — GROSS PROFITABILITY
# ============================================================

st.header("Step 5 — Screen for Gross Profitability (GPA)")

st.markdown(
    """
    **What is GPA?**

    Gross profitability measures gross profit relative to the firm's asset
    base. Higher values indicate more gross profit generated per dollar of
    assets under this definition.

    In this screen, **higher GPA is preferred**. The highest selected
    percentile receives a BUY signal and the lowest selected percentile
    receives a SELL signal.
    """
)

st.latex(
    r"\mathrm{GPA}=\frac{\mathrm{Gross\ Profit}}{\mathrm{Total\ Assets}}"
)

st.slider(
    "GPA cutoff: flag the highest and lowest percentiles",
    min_value=1,
    max_value=25,
    step=1,
    key="gpa_main",
    on_change=sync_slider,
    args=("gpa_pct", "gpa_main"),
    help="Higher GPA values receive BUY signals; lower GPA values receive SELL signals.",
)

st.write(
    f"Current rule: the highest {st.session_state['gpa_pct']}% of valid GPA "
    f"values are BUY signals, and the lowest {st.session_state['gpa_pct']}% "
    "are SELL signals."
)

show_metric_table(
    data,
    "Gross_Profitability",
    "GPA_BUY",
    "GPA_SELL",
    "GPA",
)


# ============================================================
# STEP 6 — COMBINE SIGNALS
# ============================================================

st.header("Step 6 — Combine the Screening Signals")

st.markdown(
    """
    **How the combined classification works**

    Each firm can receive up to four BUY signals and four SELL signals:
    one from each of GPA, STA, SNOA, and Altman Z-Score.

    - A BUY signal is counted as +1 in the BUY total.
    - A SELL signal is counted as +1 in the SELL total.
    - A missing metric contributes neither a BUY nor a SELL signal.
    - A firm is classified BUY when its BUY total reaches the required
      minimum and exceeds its SELL total.
    - A firm is classified SELL when its SELL total reaches the required
      minimum and exceeds its BUY total.
    - All other firms are classified NEUTRAL.

    This means a firm is not automatically penalized simply because one
    of its metrics is missing.
    """
)

st.slider(
    "Minimum number of matching signals needed for BUY or SELL",
    min_value=1,
    max_value=4,
    step=1,
    key="signals_main",
    on_change=sync_slider,
    args=("min_signals", "signals_main"),
    help="The classification must meet this minimum and have more signals in that direction than the opposite direction.",
)

st.write(
    f"Current rule: at least {st.session_state['min_signals']} signals in one "
    "direction, with more signals in that direction than the opposite direction."
)

classification_counts = (
    scored_data["Classification"]
    .value_counts()
    .reindex(["BUY", "NEUTRAL", "SELL"], fill_value=0)
)

c1, c2, c3 = st.columns(3)
c1.metric("BUY", f"{classification_counts['BUY']:,}")
c2.metric("NEUTRAL", f"{classification_counts['NEUTRAL']:,}")
c3.metric("SELL", f"{classification_counts['SELL']:,}")

chart_data = classification_counts.rename_axis("Classification").reset_index(name="Companies")
fig_class = px.bar(
    chart_data,
    x="Classification",
    y="Companies",
    title="Companies by Final Classification",
    text="Companies",
)
st.plotly_chart(fig_class, use_container_width=True)

signal_data = pd.DataFrame(
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
    signal_data,
    x="Screen",
    y=["BUY signals", "SELL signals"],
    barmode="group",
    title="BUY and SELL Signals by Screen",
)
st.plotly_chart(fig_signals, use_container_width=True)

st.subheader("Final results for all companies")
st.dataframe(
    prettify_table(
        scored_data[display_columns(scored_data)].sort_values(
            ["Classification", "Buy Signals", "Sell Signals"],
            ascending=[True, False, True],
        )
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download all screened companies as CSV",
    data=scored_data.to_csv(index=False).encode("utf-8"),
    file_name="manufacturing_stock_screen_results.csv",
    mime="text/csv",
)


# ============================================================
# STEP 7 — TOP 10 BUY AND SELL
# ============================================================

st.header("Step 7 — Top 10 BUY and Top 10 SELL Candidates")

st.markdown(
    """
    These lists show the strongest classifications under the current
    screening rules. They are rankings of the companies that passed the
    quantitative process, not predictions or guarantees.
    """
)

buy_candidates = scored_data[
    scored_data["Classification"] == "BUY"
].copy()

sell_candidates = scored_data[
    scored_data["Classification"] == "SELL"
].copy()

# Rank BUY firms by more BUY signals, fewer SELL signals, then GPA descending.
buy_sort_columns = [
    column for column in
    ["Buy Signals", "Sell Signals", "Gross_Profitability", "STA", "SNOA"]
    if column in buy_candidates.columns
]
buy_ascending = [False, True, False, True, True][:len(buy_sort_columns)]

if not buy_candidates.empty:
    buy_candidates = buy_candidates.sort_values(
        buy_sort_columns,
        ascending=buy_ascending,
        na_position="last",
    )

# Rank SELL firms by more SELL signals, fewer BUY signals, then GPA ascending.
sell_sort_columns = [
    column for column in
    ["Sell Signals", "Buy Signals", "Gross_Profitability", "STA", "SNOA"]
    if column in sell_candidates.columns
]
sell_ascending = [False, True, True, False, False][:len(sell_sort_columns)]

if not sell_candidates.empty:
    sell_candidates = sell_candidates.sort_values(
        sell_sort_columns,
        ascending=sell_ascending,
        na_position="last",
    )

buy_top10 = buy_candidates.head(10)
sell_top10 = sell_candidates.head(10)

buy_tab, sell_tab = st.tabs(["Top 10 BUY", "Top 10 SELL"])

with buy_tab:
    st.caption(
        "Ranked by BUY signal count, then fewer SELL signals and "
        "supporting financial metrics."
    )
    if buy_top10.empty:
        st.info(
            "No firms currently meet the BUY classification rule. "
            "Review the cutoff sliders and minimum-signal setting."
        )
    else:
        st.dataframe(
            prettify_table(
                buy_top10[display_columns(buy_top10)]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Top 10 BUY as CSV",
            data=buy_top10.to_csv(index=False).encode("utf-8"),
            file_name="top_10_buy.csv",
            mime="text/csv",
            key="download_buy",
        )

with sell_tab:
    st.caption(
        "Ranked by SELL signal count, then fewer BUY signals and "
        "supporting financial metrics."
    )
    if sell_top10.empty:
        st.info(
            "No firms currently meet the SELL classification rule. "
            "Review the cutoff sliders and minimum-signal setting."
        )
    else:
        st.dataframe(
            prettify_table(
                sell_top10[display_columns(sell_top10)]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Top 10 SELL as CSV",
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
    "Search for a company to review its available metrics, individual "
    "signals, and final classification."
)

search_text = st.text_input(
    "Search by company name or ticker",
    placeholder="Enter a company name or ticker...",
)

if search_text.strip():
    search_mask = (
        scored_data["Ticker"].str.contains(
            search_text, case=False, na=False, regex=False
        )
        | scored_data["Name"].str.contains(
            search_text, case=False, na=False, regex=False
        )
    )
    matches = scored_data[search_mask].copy()

    if matches.empty:
        st.info("No companies matched that search in the current universe.")
    else:
        st.dataframe(
            prettify_table(matches[display_columns(matches)]),
            use_container_width=True,
            hide_index=True,
        )

        selected_ticker = st.selectbox(
            "Choose a matching company for signal details",
            options=matches["Ticker"].tolist(),
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
                fmt_market_cap(company.get("Market Cap", np.nan)),
            )
            detail_cols[1].metric(
                "GPA",
                fmt_number(company.get("Gross_Profitability", np.nan), 4),
            )
            detail_cols[2].metric(
                "STA",
                fmt_number(company.get("STA", np.nan), 4),
            )
            detail_cols[3].metric(
                "SNOA",
                fmt_number(company.get("SNOA", np.nan), 4),
            )

            st.write(
                f"**Altman Z-Score:** {fmt_number(company.get('Altman_Z', np.nan))}"
            )
            st.write(
                f"**BUY signals:** {int(company['Buy Signals'])}  |  "
                f"**SELL signals:** {int(company['Sell Signals'])}  |  "
                f"**Classification:** {company['Classification']}"
            )

            signal_details = pd.DataFrame(
                {
                    "Screen": ["GPA", "STA", "SNOA", "Altman Z"],
                    "Metric value": [
                        company.get("Gross_Profitability", np.nan),
                        company.get("STA", np.nan),
                        company.get("SNOA", np.nan),
                        company.get("Altman_Z", np.nan),
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

            # Display missing values as N/A and do not imply they were zero.
            signal_details["Metric value"] = signal_details["Metric value"].apply(
                lambda value: "N/A" if pd.isna(value) else fmt_number(value, 4)
            )
            st.dataframe(
                signal_details,
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
    | Altman Z-Score | Above the safe-zone threshold | Below the distress threshold |

    A value exactly on an Altman threshold is neutral. For percentile-based
    screens, companies tied at a cutoff may cause the share flagged to differ
    from the exact nominal percentile.

    ### Missing data

    Missing metrics are not converted to zero. A missing value cannot
    generate a BUY or SELL signal, and missing values are excluded from
    that metric's percentile calculation. Each screen reports the number of
    firms with valid observations and the number excluded for missing data.

    ### Interpretation and limitations

    The output depends on the supplied dataset, its definitions, reporting
    dates, and the selected thresholds. Different firms may report at
    different times. A classification is a screening result, not a
    substitute for checking source statements, valuation, industry context,
    and other risks.
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
                round(100 * available / len(data), 1) if len(data) else 0
            ),
        }
    )

st.dataframe(
    pd.DataFrame(coverage_rows),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Market capitalization is parsed from numeric values and common "
    "suffixes such as K, M, B, and T. Values that cannot be interpreted "
    "remain missing and are not treated as zero."
)
