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
