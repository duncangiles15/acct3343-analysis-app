
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Stock Screening App", layout="wide")

st.title("Manufacturing Stock Screening App")
st.write("Explore 851 companies using Altman Z, STA, SNOA, and gross profitability.")

@st.cache_data
def load_data():
    df = pd.read_csv("universe.csv")
    df["Ticker"] = df["Ticker"].astype("string").str.strip().str.upper()
    return df

df = load_data()

st.sidebar.header("Screening Settings")

min_gp = st.sidebar.slider("Minimum Gross Profitability", -1.0, 1.0, 0.10, 0.01)
min_altman = st.sidebar.slider("Minimum Altman Z", -5.0, 20.0, 1.81, 0.1)
sta_limit = st.sidebar.slider("STA SELL threshold", -1.0, 1.0, 0.0, 0.01)
snoa_limit = st.sidebar.slider("SNOA SELL threshold", -1.0, 1.0, 0.5, 0.01)
sell_needed = st.sidebar.slider("SELL signals required", 1, 4, 2)

df["STA_SELL"] = df["STA"].notna() & (df["STA"] > sta_limit)
df["SNOA_SELL"] = df["SNOA"].notna() & (df["SNOA"] > snoa_limit)
df["GP_SELL"] = df["Gross_Profitability"].notna() & (df["Gross_Profitability"] < min_gp)
df["Altman_SELL"] = df["Altman_Z"].notna() & (df["Altman_Z"] < min_altman)

flags = ["STA_SELL", "SNOA_SELL", "GP_SELL", "Altman_SELL"]
df["SELL_Signals"] = df[flags].sum(axis=1)
df["Available_Metrics"] = df[["STA", "SNOA", "Gross_Profitability", "Altman_Z"]].notna().sum(axis=1)

df["Decision"] = np.where(
    df["SELL_Signals"] >= sell_needed,
    "SELL",
    np.where(df["Available_Metrics"] < sell_needed, "REVIEW", "BUY")
)

buy = df[df["Decision"] == "BUY"]
sell = df[df["Decision"] == "SELL"]
review = df[df["Decision"] == "REVIEW"]

a, b, c, d = st.columns(4)
a.metric("Companies", len(df))
b.metric("BUY", len(buy))
c.metric("SELL", len(sell))
d.metric("REVIEW", len(review))

st.subheader("BUY Candidates")
st.dataframe(
    buy[["Ticker", "Name", "Country", "Market Cap",
         "Gross_Profitability", "Altman_Z", "STA", "SNOA", "SELL_Signals"]]
    .sort_values("Gross_Profitability", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.subheader("SELL Candidates")
st.dataframe(
    sell[["Ticker", "Name", "Country", "Market Cap",
          "Gross_Profitability", "Altman_Z", "STA", "SNOA", "SELL_Signals"]],
    use_container_width=True,
    hide_index=True
)

st.subheader("Gross Profitability vs. Altman Z")
chart_data = df.dropna(subset=["Gross_Profitability", "Altman_Z"])

if not chart_data.empty:
    fig = px.scatter(
        chart_data,
        x="Gross_Profitability",
        y="Altman_Z",
        color="Decision",
        hover_name="Ticker",
        hover_data=["Name", "STA", "SNOA"],
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Company Details")
ticker = st.selectbox("Select a company", sorted(df["Ticker"].dropna().unique()))
company = df[df["Ticker"] == ticker].iloc[0]

st.write("**Company:**", company["Name"])
st.write("**Country:**", company["Country"])
st.write("**Market cap:**", company["Market Cap"])
st.write("**Altman Z:**", company["Altman_Z"])
st.write("**STA:**", company["STA"])
st.write("**SNOA:**", company["SNOA"])
st.write("**Gross profitability:**", company["Gross_Profitability"])
st.write("**Screening decision:**", company["Decision"])

st.subheader("Your Analysis")
st.text_area(
    "Write your conclusions and limitations here",
    placeholder="Discuss the patterns you observe and limitations of the financial data.",
    height=150
)

st.caption("Screening results are for educational analysis, not investment advice.")

