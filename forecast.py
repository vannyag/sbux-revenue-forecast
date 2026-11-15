import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
CIK = "0000829224"   # Starbucks. Try Apple 0000320193, Nike 0000320187, Costco 0000909832
HEADERS = {"User-Agent": "Vannya vannyagarg1785@gmail.com"}
START_YEAR = 2012
N_TEST = 12          # quarters used for backtesting
HORIZON = 4          # quarters to forecast
TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]


# ---------------- 1. DATA ----------------
def fetch_raw():
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    facts = r.json()["facts"]["us-gaap"]
    rows = []
    for tag in TAGS:
        if tag in facts:
            for u in facts[tag]["units"]["USD"]:
                rows.append({**u, "tag": tag})
    df = pd.DataFrame(rows)
    for c in ["start", "end", "filed"]:
        df[c] = pd.to_datetime(df[c])
    df["days"] = (df["end"] - df["start"]).dt.days
    df = df.sort_values("filed").drop_duplicates(["start", "end"], keep="last")
    return df


def to_quarterly(df):
    """10-Qs give Q1-Q3 (about 90 days). Q4 is not filed separately,
    so derive it: Q4 = Annual - (Q1+Q2+Q3)."""
    q = df[df.days.between(80, 100)][["start", "end", "val"]]
    a = df[df.days.between(350, 380)][["start", "end", "val"]]
    derived = []
    for _, row in a.iterrows():
        inside = q[(q.start >= row.start) & (q.end < row.end)]
        if len(inside) == 3:
            derived.append({
                "start": inside.end.max() + pd.Timedelta(days=1),
                "end": row.end,
                "val": row.val - inside.val.sum(),
            })
    out = pd.concat([q, pd.DataFrame(derived)]).sort_values("end").drop_duplicates("end")
    return (out.set_index("end")["val"] / 1e9).rename("revenue_bn")


def load_series():
    s = to_quarterly(fetch_raw())
    s.index = (s.index - pd.Timedelta(days=15)).to_period("Q")
    s = s[~s.index.duplicated(keep="last")]
    s = s[s.index.year >= START_YEAR]
    full = pd.period_range(s.index.min(), s.index.max(), freq="Q")
    missing = full.difference(s.index)
    if len(missing):
        print("Missing quarters (interpolated):", list(missing))
        s = s.reindex(full).interpolate()
    return s


# ---------------- 2. MODELS ----------------
def naive_fc(train, h):
    """Baseline: same quarter last year."""
    v = train.values
    return np.array([v[-4 + k] for k in range(h)])


def ets_fc(train, h):
    m = ExponentialSmoothing(train.values, trend="add", damped_trend=True,
                             seasonal="mul", seasonal_periods=4).fit()
    return m.forecast(h)


def sarimax_full(train, h):
    m = SARIMAX(np.log(train.values), order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 4)).fit(disp=False)
    fc = m.get_forecast(h)
    ci = fc.conf_int(alpha=0.20)          # 80% prediction interval
    return np.exp(fc.predicted_mean), np.exp(ci[:, 0]), np.exp(ci[:, 1])


def sarimax_fc(train, h):
    return sarimax_full(train, h)[0]


def _feat(y, q, t):
    return [y[t-1] - y[t-5], y[t-2] - y[t-6], y[t-1] - y[t-2], q[t]]


def gbm_fc(train, h):
    """ML model: predicts year-over-year growth, converts back to revenue."""
    y = list(np.log(train.values))
    q = list(train.index.quarter)
    X, target = [], []
    for t in range(6, len(y)):
        X.append(_feat(y, q, t))
        target.append(y[t] - y[t-4])
    model = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X, target)
    last = train.index[-1]
    for k in range(1, h + 1):
        q.append((last + k).quarter)
        t = len(y)
        g = model.predict([_feat(y, q, t)])[0]
        y.append(y[t-4] + g)
    return np.exp(np.array(y[-h:]))


MODELS = {"naive": naive_fc, "ets": ets_fc, "sarima": sarimax_fc, "gbm": gbm_fc}


# ---------------- 3. WALK-FORWARD BACKTEST ----------------
def backtest(s):
    """Pretend it's the past: train on data up to each quarter, predict it, compare."""
    rows = []
    for i in range(len(s) - N_TEST, len(s)):
        train, actual = s.iloc[:i], s.iloc[i]
        row = {"period": str(s.index[i]), "actual": actual}
        for name, fn in MODELS.items():
            try:
                row[name] = float(fn(train, 1)[0])
            except Exception as e:
                print(f"{name} failed at {s.index[i]}: {e}")
                row[name] = np.nan
        row["ensemble"] = np.nanmean([row["ets"], row["sarima"], row["gbm"]])
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(bt):
    out = []
    for m in ["naive", "ets", "sarima", "gbm", "ensemble"]:
        err = bt[m] - bt["actual"]
        out.append({
            "model": m,
            "MAPE_%": round((err.abs() / bt["actual"]).mean() * 100, 2),
            "MAE_bn": round(err.abs().mean(), 3),
            "RMSE_bn": round(np.sqrt((err ** 2).mean()), 3),
        })
    return pd.DataFrame(out).sort_values("MAPE_%")


# ---------------- 4. FINAL FORECAST ----------------
def final_forecast(s):
    future = pd.period_range(s.index[-1] + 1, periods=HORIZON, freq="Q")
    mean, lo, hi = sarimax_full(s, HORIZON)
    fc = pd.DataFrame({"period": future.astype(str)})
    fc["naive"] = naive_fc(s, HORIZON)
    fc["ets"] = ets_fc(s, HORIZON)
    fc["sarima"] = mean
    fc["gbm"] = gbm_fc(s, HORIZON)
    fc["ensemble"] = fc[["ets", "sarima", "gbm"]].mean(axis=1)
    fc["lower_80"], fc["upper_80"] = lo, hi
    return fc


if __name__ == "__main__":
    s = load_series()
    print(f"Loaded {len(s)} quarters: {s.index[0]} to {s.index[-1]}")
    s.rename("revenue_bn").reset_index().rename(columns={"index": "period"}).astype(str) \
        .to_csv("outputs/history.csv", index=False)

    bt = backtest(s)
    bt.to_csv("outputs/backtest.csv", index=False)
    scores = metrics(bt)
    scores.to_csv("outputs/metrics.csv", index=False)
    print("\nBACKTEST RESULTS (lower is better)")
    print(scores.to_string(index=False))

    fc = final_forecast(s)
    fc.to_csv("outputs/forecast.csv", index=False)
    print("\nFORECAST ($bn)")
    print(fc.round(2).to_string(index=False))

    plt.figure(figsize=(12, 5))
    hist_x = s.index.astype(str)
    plt.plot(hist_x, s.values, label="Actual", color="black")
    plt.plot(fc["period"], fc["ensemble"], "o--", label="Ensemble forecast", color="tab:green")
    plt.fill_between(fc["period"], fc["lower_80"], fc["upper_80"], alpha=0.2, label="80% interval")
    plt.xticks(list(hist_x)[::4] + list(fc["period"]), rotation=45)
    plt.ylabel("Revenue ($bn)")
    plt.title("Starbucks Quarterly Revenue Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/forecast.png", dpi=150)
    print("Saved outputs/forecast.png")