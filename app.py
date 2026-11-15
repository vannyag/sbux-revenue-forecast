import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Starbucks Revenue Forecast", page_icon="☕", layout="wide")

# ---------- PALETTE (taken from the Starbucks slide design) ----------
DEEP = "#1E5C42"      # forest green (slide backgrounds)
FOREST = "#14382A"    # darkest green (text)
MID = "#3F8A67"
SOFT = "#6FB08F"
MINT = "#D6E8E0"      # pale mint (light slides)
PAPER = "#EEF5F1"
GOLD = "#C9A24D"      # accent
CLAY = "#C8553D"      # accent for negatives
GREY = "#8A9A93"

MODEL_COLORS = {"naive": GREY, "ets": SOFT, "sarima": GOLD, "gbm": CLAY, "ensemble": DEEP}
QUARTER_COLORS = [DEEP, SOFT, GOLD, "#A9CDBB"]
GOOD, BAD = MID, CLAY

# ---------- STYLE ----------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: {MINT}; }}

/* readable text everywhere */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li,
.stApp label, .stApp span, [data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] {{ color: {FOREST}; }}
.stApp h2, .stApp h3 {{ color: {DEEP}; font-weight: 800; }}
[data-testid="stCaptionContainer"] p {{ color: {MID} !important; }}

/* hero banner: dark green slide with a mint circle */
.hero {{
  position: relative; overflow: hidden; background: {DEEP}; border-radius: 28px;
  padding: 30px 40px; display: flex; align-items: center; gap: 28px; margin-bottom: 22px;
}}
.hero::after {{
  content: ""; position: absolute; right: -90px; bottom: -120px; width: 320px; height: 320px;
  border-radius: 50%; background: rgba(214,232,224,.12);
}}
.hero .badge {{
  flex: none; width: 96px; height: 96px; border-radius: 50%; background: {MINT};
  display: flex; align-items: center; justify-content: center; font-size: 3rem;
  box-shadow: 0 0 0 6px rgba(214,232,224,.25);
}}
.stApp .hero h1 {{ color: #FFFFFF; margin: 0; font-weight: 800; font-size: 2.4rem; letter-spacing: 1px; }}
.stApp .hero p {{ color: {MINT}; margin: 6px 0 0 0; font-size: 1rem; }}

/* KPI cards: alternate dark green and white, like the slides */
.kpi {{ border-radius: 22px; padding: 20px 22px; height: 100%; }}
.kpi.dark {{ background: {DEEP}; }}
.kpi.light {{ background: #FFFFFF; border: 2px solid {DEEP}; }}
.kpi .dot {{
  width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 1.1rem; margin-bottom: 8px;
}}
.kpi.dark .dot {{ background: {MINT}; }}
.kpi.light .dot {{ background: {DEEP}; }}
.stApp .kpi .label {{ font-size: .82rem; font-weight: 600; }}
.stApp .kpi .value {{ font-size: 2rem; font-weight: 800; line-height: 1.2; }}
.stApp .kpi .sub {{ font-size: .8rem; }}
.stApp .kpi.dark .label, .stApp .kpi.dark .sub {{ color: {MINT}; }}
.stApp .kpi.dark .value {{ color: #FFFFFF; }}
.stApp .kpi.light .label, .stApp .kpi.light .sub {{ color: {MID}; }}
.stApp .kpi.light .value {{ color: {DEEP}; }}

/* note box */
.note {{
  background: {PAPER}; border-left: 8px solid {DEEP}; border-radius: 14px;
  padding: 14px 18px; margin: 10px 0 18px 0;
}}
.stApp .note, .stApp .note b {{ color: {FOREST}; }}

/* tabs */
button[data-baseweb="tab"] {{ font-size: 1rem; }}
button[data-baseweb="tab"] p {{ color: {MID} !important; font-weight: 600; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color: {DEEP} !important; font-weight: 800; }}
[data-baseweb="tab-highlight"] {{ background-color: {DEEP} !important; height: 4px; }}

/* sidebar and widgets */
[data-testid="stSidebar"] {{ background: #C5DDD1; }}
[data-testid="stSidebar"] * {{ color: {FOREST}; }}
span[data-baseweb="tag"] {{ background: {DEEP} !important; }}
span[data-baseweb="tag"] span {{ color: #FFFFFF !important; }}
[data-testid="stDataFrame"] {{ background: #FFFFFF; border-radius: 14px; }}
.stDownloadButton button {{ background: {DEEP}; border: 0; border-radius: 999px; }}
.stDownloadButton button p {{ color: #FFFFFF !important; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)


def style(fig, h=450, **kw):
    fig.update_layout(
        template="plotly_white", height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.88)",
        font=dict(family="Poppins, sans-serif", color=FOREST),
        margin=dict(l=10, r=10, t=40, b=10), **kw,
    )
    return fig


# ---------- DATA ----------
@st.cache_data
def load():
    hist = pd.read_csv("outputs/history.csv")
    hist.columns = ["period", "revenue_bn"]
    fc = pd.read_csv("outputs/forecast.csv")
    bt = pd.read_csv("outputs/backtest.csv")
    scores = pd.read_csv("outputs/metrics.csv")
    return hist, fc, bt, scores


try:
    hist, fc, bt, scores = load()
except FileNotFoundError:
    st.error("Data files not found. Run `python forecast.py` first, then start the dashboard from the project folder.")
    st.stop()

hist["yoy"] = (hist.revenue_bn / hist.revenue_bn.shift(4) - 1) * 100
hist["quarter"] = "Q" + hist.period.str[-1]

# ---------- SIDEBAR ----------
st.sidebar.markdown("## ☕ Controls")
models = st.sidebar.multiselect("Models to show", list(MODEL_COLORS), default=["ensemble", "sarima"])
n_hist = st.sidebar.slider("Quarters of history", 8, len(hist), min(24, len(hist)))
show_band = st.sidebar.checkbox("Show 80% interval", value=True)
st.sidebar.caption("Data: SEC EDGAR. Revenue in $ billions.")

# ---------- HERO ----------
st.markdown(
    """
<div class="hero">
<div class="badge">☕</div>
<div>
<h1>STARBUCKS REVENUE FORECAST</h1>
<p>Live SEC filings, four forecasting models and an honest backtest.</p>
</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- KPI CARDS ----------
latest = hist.iloc[-1]
next_q = fc.iloc[0]
last4 = hist.revenue_bn.tail(4).sum()
next4 = fc.ensemble.sum()
growth4 = (next4 / last4 - 1) * 100
best = scores.iloc[0]


def card(col, kind, icon, label, value, sub):
    col.markdown(
        f'<div class="kpi {kind}"><div class="dot">{icon}</div><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


k1, k2, k3, k4 = st.columns(4)
card(k1, "dark", "💵", "Latest quarter", f"${latest.revenue_bn:.2f}B", f"{latest.period}  |  YoY {latest.yoy:+.1f}%")
card(k2, "light", "🔮", "Next quarter forecast", f"${next_q.ensemble:.2f}B", f"{next_q.period}  |  ensemble")
card(k3, "dark", "📅", "Next 4 quarters", f"${next4:.1f}B", f"{growth4:+.1f}% vs last 4 quarters")
card(k4, "light", "🏆", "Most accurate model", str(best["model"]).upper(), f"MAPE {best['MAPE_%']}% in backtest")

st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Forecast", "🏆 Model showdown", "🔍 Backtest", "📊 Growth & seasons", "🗂 Data"]
)

# ---------- TAB 1: FORECAST ----------
with tab1:
    h = hist.tail(n_hist)
    fig = go.Figure()
    fig.add_scatter(x=h.period, y=h.revenue_bn, name="Actual", mode="lines+markers",
                    line=dict(color=FOREST, width=3), marker=dict(size=6))
    if show_band:
        fig.add_scatter(
            x=[latest.period] + list(fc.period) + list(fc.period[::-1]) + [latest.period],
            y=[latest.revenue_bn] + list(fc.upper_80) + list(fc.lower_80[::-1]) + [latest.revenue_bn],
            fill="toself", fillcolor="rgba(30,92,66,0.15)", line=dict(width=0),
            name="80% interval", hoverinfo="skip",
        )
    for m in models:
        fig.add_scatter(
            x=[latest.period] + list(fc.period), y=[latest.revenue_bn] + list(fc[m]),
            name=m, mode="lines+markers",
            line=dict(color=MODEL_COLORS[m], width=4 if m == "ensemble" else 2.5, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
        )
    style(fig, 520, yaxis_title="Revenue ($bn)", legend=dict(orientation="h", y=1.1), hovermode="x unified")
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div class="note">The ensemble expects <b>${next_q.ensemble:.2f}B</b> in {next_q.period}. '
        f'The shaded band is the range SARIMA expects 80% of the time, and it widens the further ahead we look.</div>',
        unsafe_allow_html=True,
    )

# ---------- TAB 2: MODEL SHOWDOWN ----------
with tab2:
    s = scores.sort_values("MAPE_%")
    fig = go.Figure(go.Bar(
        x=s["model"], y=s["MAPE_%"],
        marker_color=[MODEL_COLORS.get(m, GREY) for m in s["model"]],
        text=[f"{v}%" for v in s["MAPE_%"]], textposition="outside",
    ))
    style(fig, 430, yaxis_title="MAPE % (lower is better)")
    st.plotly_chart(fig, use_container_width=True)

    naive_mape = float(scores.loc[scores["model"] == "naive", "MAPE_%"].iloc[0])
    if best["model"] == "naive":
        msg = "The simple seasonal baseline won. That is a real finding: more complex does not always mean better."
    else:
        gain = (1 - best["MAPE_%"] / naive_mape) * 100
        msg = (f"<b>{str(best['model']).upper()}</b> beat the naive baseline by {gain:.0f}% "
               f"({best['MAPE_%']}% vs {naive_mape}% error).")
    st.markdown(f'<div class="note">{msg}</div>', unsafe_allow_html=True)
    st.dataframe(scores, use_container_width=True, hide_index=True)

# ---------- TAB 3: BACKTEST ----------
with tab3:
    st.caption("Each point is a quarter the models predicted using only data from before it.")
    fig = go.Figure()
    fig.add_scatter(x=bt.period, y=bt.actual, name="Actual", line=dict(color=FOREST, width=4))
    for m in models:
        fig.add_scatter(x=bt.period, y=bt[m], name=m, mode="lines+markers",
                        line=dict(color=MODEL_COLORS[m], width=2.5))
    style(fig, 430, yaxis_title="Revenue ($bn)", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    pick = st.selectbox("Error for model", list(MODEL_COLORS), index=4)
    err = (bt[pick] - bt.actual) / bt.actual * 100
    fig2 = go.Figure(go.Bar(
        x=bt.period, y=err, marker_color=[GOOD if e >= 0 else BAD for e in err],
        text=[f"{e:+.1f}%" for e in err], textposition="outside",
    ))
    style(fig2, 330, yaxis_title="Forecast error %")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Green means the model over-predicted. Clay red means it under-predicted.")

# ---------- TAB 4: GROWTH & SEASONS ----------
with tab4:
    c1, c2 = st.columns([3, 2])
    g = hist.dropna(subset=["yoy"]).tail(n_hist)
    fig = go.Figure(go.Bar(x=g.period, y=g.yoy, marker_color=[GOOD if v >= 0 else BAD for v in g.yoy]))
    style(fig, 400, title="Year-over-year growth (%)")
    fig.update_xaxes(tickangle=-45)
    c1.plotly_chart(fig, use_container_width=True)

    hist["year"] = hist.period.str[:4]
    hist["index"] = hist.revenue_bn / hist.groupby("year").revenue_bn.transform("mean") * 100
    seas = hist.groupby("quarter")["index"].mean().reindex(["Q1", "Q2", "Q3", "Q4"])
    fig = go.Figure(go.Bar(x=seas.index, y=seas.values, marker_color=QUARTER_COLORS,
                           text=[f"{v:.0f}" for v in seas.values], textposition="outside"))
    style(fig, 400, title="Seasonality (100 = yearly average)")
    fig.update_yaxes(range=[min(80, seas.min() - 5), seas.max() + 8])
    c2.plotly_chart(fig, use_container_width=True)
    st.caption("Note: the 2020 COVID quarters pull the growth chart sharply down, then back up.")

# ---------- TAB 5: DATA ----------
with tab5:
    st.subheader("Forecast table")
    st.dataframe(fc.round(2), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download forecast CSV", fc.to_csv(index=False), "forecast.csv", "text/csv")
    st.subheader("History")
    st.dataframe(hist[["period", "revenue_bn", "yoy"]].round(2), use_container_width=True, hide_index=True)