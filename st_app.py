# ECO 5012B - Section 3 - Streamlit Nowcasting App
# Interactive nowcasting app for German GDP growth

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# page config
st.set_page_config(page_title="Germany GDP Nowcaster", layout="wide")

# title and description
st.title("Germany GDP Nowcasting App")
st.markdown("This app uses the OECD Business Confidence Index (BCI) to nowcast German GDP growth. Based on Ashwin et al. (2024).")

# regression coefficients from our real estimated model (Section 2.3)
BETA_0    = -48.5858
BETA_1    = -0.3084   # lagged GDP coefficient
BETA_S    =  0.4871   # sentiment coefficient - normal times

# sidebar controls
st.sidebar.header("Model Controls")

# radio button - economic state
state = st.sidebar.radio(
    "Select Economic State:",
    ["Normal Times", "Supply Shock"]
)
st.sidebar.caption("The 50% reduction is a stylised assumption illustrating weaker sentiment transmission during supply shocks, consistent with the regime-dependence finding in Ashwin et al. (2024).")

# adjusting sentiment coefficient based on state
if state == "Normal Times":
    beta_s_adjusted = BETA_S
    st.sidebar.info("Normal times: standard sentiment impact")
else:
    beta_s_adjusted = BETA_S * 0.5
    st.sidebar.warning("Supply Shock: sentiment impact reduced by 50%")

# slider - adjust sentiment coefficient by percentage
pct_change = st.sidebar.slider(
    "Adjust Sentiment Coefficient βS (%):",
    min_value=-50,
    max_value=50,
    value=0,
    step=5
)
st.sidebar.caption("Slider adjusts βS by a fixed percentage on top of the state adjustment.")

# applying the percentage adjustment on top of the state adjustment
beta_s_final = beta_s_adjusted * (1 + pct_change / 100)

# load latest real values for nowcast calculation
try:
    df_latest = pd.read_csv("germany_nowcast_data.csv", index_col="date", parse_dates=True)
    bci_value = df_latest["BCI"].iloc[-1]
    gdp_lag_value = df_latest["GDP_growth"].iloc[-1]
except FileNotFoundError:
    bci_value = 100.0
    gdp_lag_value = 0.0

# computing nowcast
nowcast = BETA_0 + BETA_1 * gdp_lag_value + beta_s_final * bci_value

# displaying nowcast metric
st.header("Live Nowcast")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Predicted GDP Growth (%)",
        value=f"{nowcast:.3f}%"
    )

with col2:
    st.metric(
        label="Sentiment Coefficient βS",
        value=f"{beta_s_final:.4f}"
    )

with col3:
    st.metric(
        label="Economic State",
        value=state
    )

# loading real data for the chart
st.header("Historical GDP Growth vs Live Nowcast")

try:
    df = pd.read_csv("germany_nowcast_data.csv", index_col="date", parse_dates=True)
    df["GDP_lag"] = df["GDP_growth"].shift(1)
    df = df.dropna()

    # computing fitted values using the slider-adjusted coefficient
    df["fitted"] = BETA_0 + BETA_1 * df["GDP_lag"] + beta_s_final * df["BCI"]

    # creating the plotly express chart
    df_plot = df[["GDP_growth", "fitted"]].reset_index()
    df_plot_long = df_plot.melt(id_vars="date",
                                 value_vars=["GDP_growth", "fitted"],
                                 var_name="Series",
                                 value_name="GDP Growth (%)")

    df_plot_long["Series"] = df_plot_long["Series"].map({
        "GDP_growth": "Actual GDP Growth",
        "fitted": "Model Fitted Values"
    })

    fig = px.line(
        df_plot_long,
        x="date",
        y="GDP Growth (%)",
        color="Series",
        color_discrete_map={
            "Actual GDP Growth": "steelblue",
            "Model Fitted Values": "red"
        },
        title=f"Germany GDP Nowcast - {state} (βS adjusted by {pct_change}%)",
        template="plotly_white",
        height=500,
        labels={"date": "Date"}
    )

    # make fitted values dashed
    fig.update_traces(line=dict(dash="dash"), selector=dict(name="Model Fitted Values"))
    fig.update_traces(line=dict(dash="solid"), selector=dict(name="Actual GDP Growth"))

    # live nowcast point
    last_date = df.index[-1] + pd.DateOffset(months=3)
    fig.add_scatter(
        x=[last_date],
        y=[nowcast],
        mode="markers",
        name="Live Nowcast",
        marker=dict(color="green", size=12, symbol="star")
    )

    # crisis shading
    fig.add_vrect(
        x0="2008-07-01", x1="2009-06-30",
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="GFC"
    )
    fig.add_vrect(
        x0="2020-01-01", x1="2020-09-30",
        fillcolor="orange", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="COVID"
    )

    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("germany_nowcast_data.csv not found. Please make sure it is in the same folder as this file.")

# research extension section
st.header("Research Extension")
st.markdown("""
This app illustrates a key finding from Ashwin et al. (2024) — that sentiment informativeness 
varies across economic regimes. The **Supply Shock** state reduces the sentiment coefficient by 50%, 
reflecting the finding that during supply-driven crises, business confidence becomes a weaker 
predictor of GDP growth as output is constrained by factors beyond demand expectations.

The slider allows users to further adjust the sentiment coefficient βS by up to ±50%, 
exploring how the strength of the sentiment-GDP relationship affects the nowcast. 
Combined with the state selector, this demonstrates both the level and regime-dependence 
of sentiment informativeness.

**Research question:** Does the sentiment-GDP relationship differ between demand-driven crises 
(like the GFC) and supply-driven shocks (like COVID-19)? This app allows users to explore 
this state-dependent relationship interactively.
""")
