Germany GDP Nowcasting App

A sentiment-based GDP nowcasting model for Germany, built in Python and deployed as a live interactive Streamlit application.

Live app: https://eco5012b-app-fsgzkgzyregsskwnwkpdpi.streamlit.app


Overview

This project replicates and extends the nowcasting methodology of Ashwin, Kalamara & Saiz (2024), 'Nowcasting Euro area GDP with news sentiment: A tale of two crises', Journal of Applied Econometrics, 39(5), pp. 887–905.

The model uses the OECD Business Confidence Index (BCI) as a real-time sentiment proxy to nowcast German quarterly GDP growth, addressing the problem that official GDP figures are released with a significant lag. The application allows users to interact with the model in real time, adjusting the sentiment coefficient and toggling between economic states to observe how model predictions respond.


Methodology

Model specification:

ΔGDPt = β0 + β1·ΔGDPt-1 + βS·St + εt

Where:


ΔGDPt — quarter-on-quarter log growth rate of German real GDP
ΔGDPt-1 — lagged GDP growth (controls for output persistence)
St — OECD Business Confidence Index (sentiment proxy)
εt — error term


Data sources:


German real GDP: Eurostat via FRED (CLVMNACSCAB1GQDE), chained 2015 prices, seasonally adjusted
Business Confidence Index: OECD via FRED (BSCICP03DEM665S)
Sample: Q2 2000 – Q1 2024 (95 observations)


Data preparation:


GDP converted to log quarter-on-quarter growth rate
Monthly BCI aggregated to quarterly frequency by averaging three monthly observations within each quarter, following Ashwin et al. (2024)
Series merged on an inner join



Key Results

StatisticValueSentiment coefficient βS0.487 (p < 0.001)GDP lag coefficient β1-0.308 (p = 0.003)R-squared0.194Durbin-Watson1.982Observations95

A one-point increase in BCI is associated with a 0.487 percentage point increase in quarterly GDP growth. The BCI is statistically significant at the 1% level, confirming it contains information useful for nowcasting beyond past GDP growth alone.

The Durbin-Watson statistic of 1.982 confirms no first-order autocorrelation in residuals. The Jarque-Bera test rejects normality (driven by extreme kurtosis of 21.8 from GFC 2008 and COVID-19 2020 outliers) — consistent with Ashwin et al.'s finding that linear sentiment models underpredict the magnitude of structural shocks.

Model extension: A Random Forest regression (500 trees, max depth 3) was estimated for comparison. In-sample R² of 0.664, but cross-validated R² of -0.627, confirming overfitting with 95 observations and supporting the simpler OLS specification.


Streamlit Application

The interactive app includes:


Live nowcast — predicted GDP growth updated with the current BCI value
Sentiment slider — adjust βS by up to ±50% to explore sensitivity
Economic state toggle — switch between Normal Times (βS = 0.487) and Supply Shock (βS halved to 0.244), illustrating how the sentiment-GDP transmission weakens during supply-driven contractions (e.g. COVID-19) vs demand-driven crises (e.g. GFC 2008)
Interactive chart — actual GDP growth, model fitted values, and live nowcast projection with annotated crisis periods



Installation

bashgit clone https://github.com/edwardjvn-art/ECO5012B-App.git
cd ECO5012B-App
pip install -r requirements.txt
streamlit run app.py

Dependencies: pandas, pandas_datareader, numpy, statsmodels, scikit-learn, plotly, streamlit


Research Extension

The application tests a hypothesis not examined in Ashwin et al. (2024): whether the sentiment-GDP relationship differs between demand-driven and supply-driven crises. During demand shocks (GFC 2008), falling BCI directly reflects firms cutting spending — sentiment and output move together. During supply shocks (COVID-19), output is constrained by exogenous factors (lockdowns, supply chain disruptions), weakening the sentiment transmission channel. The state-dependent toggle illustrates this distinction interactively.


Reference

Ashwin, J., Kalamara, E. and Saiz, L. (2024) 'Nowcasting Euro area GDP with news sentiment: A tale of two crises', Journal of Applied Econometrics, 39(5), pp. 887–905. https://doi.org/10.1002/jae.3057
