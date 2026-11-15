# ☕ Starbucks Revenue Forecasting

**🔗 Live dashboard:** https://sbux-revenue-forecast.streamlit.app/

An end-to-end data analytics project that downloads Starbucks's official quarterly revenue from SEC filings, compares four forecasting methods using an honest backtest, and presents the results in an interactive dashboard.

---

## Business problem

Finance and planning teams need to estimate future revenue to set budgets, plan hiring and judge performance. A forecast is only useful if you know **how accurate it has been in the past** and **how uncertain it is**.

This project answers three questions:

1. What will Starbucks's revenue be over the next 4 quarters?
2. Which forecasting method has been the most accurate historically?
3. How much uncertainty surrounds the forecast?

## Data

| Item | Detail |
|---|---|
| Source | [SEC EDGAR XBRL API](https://www.sec.gov/edgar/sec-api-documentation) (`companyfacts` endpoint) |
| Company | Starbucks Corp (CIK 0000829224) |
| Metric | Quarterly total revenue, in $ billions |
| Period | 2012 to latest filing |
| Update | Re-run `forecast.py` to pull the newest filing |

**Data preparation steps**

- Pulled several revenue tags, because companies change the tag they use over time, and kept the most recently filed value for each period.
- Kept quarter-length periods (about 90 days) from the 10-Q filings.
- **Q4 is not filed separately**, so it was derived as `Annual revenue - (Q1 + Q2 + Q3)`.
- Mapped each fiscal quarter-end to the nearest calendar quarter. Starbucks's fiscal year ends around late September, so period labels such as `2024Q4` are **approximate calendar quarters**, not Starbucks's own fiscal quarter names.
- Checked for missing quarters and interpolated any gaps (the script prints a warning if it does).

## Methodology

### Models compared

| Model | What it does |
|---|---|
| **Seasonal naive** (baseline) | Predicts that a quarter will equal the same quarter last year. Any serious model must beat this. |
| **ETS (Holt-Winters)** | Smooths the level, trend and seasonal pattern, giving recent data more weight. |
| **SARIMA** | A statistical model that learns from past values and past errors, with a yearly seasonal component. Fitted on log revenue. Also produces the 80% prediction interval. |
| **Gradient Boosting** | A machine learning model that predicts year-over-year growth from recent growth rates and the calendar quarter. |
| **Ensemble** | The average of ETS, SARIMA and Gradient Boosting. |

### Validation: walk-forward backtest

A random train/test split would let the model "peek" at the future, which is called **data leakage**. Instead, the project does this for each of the last 12 quarters:

1. Train using only data available **before** that quarter.
2. Predict that quarter.
3. Compare with the real reported figure.
4. Move forward one quarter and repeat.

### Metrics

- **MAPE**: average percentage error. Easy to explain to non-technical readers.
- **MAE**: average error in $ billions.
- **RMSE**: like MAE, but penalises large misses more.

### SQL analysis

`sql_analysis.py` loads the history into SQLite and uses window functions (`LAG`) to calculate year-over-year and quarter-over-quarter growth, plus a `GROUP BY` to measure seasonality.

## Dashboard

The Streamlit dashboard has five tabs:

- **Forecast:** history, model forecasts and the 80% interval.
- **Model showdown:** accuracy of every model compared with the baseline.
- **Backtest:** predicted vs actual, plus the error for each quarter.
- **Growth & seasons:** year-over-year growth and seasonality by quarter.
- **Data:** tables and a CSV download.

The sidebar lets you choose which models to display, how much history to show and whether to display the interval.

## Limitations

Being clear about limits is part of good analysis.

- **Small sample:** only about 50 quarters of data, so the machine learning model has little to learn from.
- **COVID-19 (2020):** an extreme outlier that distorts seasonality and growth patterns. It was not removed or adjusted for.
- **History only:** models do not use pricing, store counts, consumer spending, inflation, company guidance or analyst expectations.
- **Derived Q4:** Q4 revenue is calculated, not separately reported.
- **Approximate quarters:** period labels are approximate calendar quarters, not Starbucks's fiscal quarter names.
- **Prediction interval:** the interval comes from SARIMA only and assumes the model is correctly specified.
- **Not financial advice:** this is a learning project and should not be used for investment decisions.

## How to run it

**1. Clone the repository**
```bash
git clone https://github.com/[YOUR-USERNAME]/sbux-revenue-forecast.git
cd sbux-revenue-forecast
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the pipeline**
```bash
python forecast.py
python sql_analysis.py
```

**5. Launch the dashboard**
```bash
streamlit run app.py
```

## Tech stack

- **Python:** pandas, NumPy, requests
- **Modelling:** statsmodels (ETS, SARIMA), scikit-learn (Gradient Boosting)
- **Visualisation:** Matplotlib, Plotly
- **Dashboard:** Streamlit
- **SQL:** SQLite (window functions)
- **Data:** SEC EDGAR XBRL API


*Data comes from public SEC filings. This project is not affiliated with Starbucks.*
