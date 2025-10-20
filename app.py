# app.py
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -------------------- Page Config & Light Styles --------------------
st.set_page_config(page_title="Lockheed Martin Dashboard", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {background:#fff;border:1px solid #eee;border-radius:12px;padding:16px 20px;box-shadow:0 1px 2px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] {
    color: #333 !important;          /* dark gray text */
    font-weight: 600 !important;     /* bold labels */
    font-size: 1rem !important;      /* readable size */
}
.stTabs [aria-selected="true"] {
    color: #000 !important;          /* black for the active tab */
}
</style>
""", unsafe_allow_html=True)

# -------------------- Defaults --------------------
DEFAULT_TICKER = "LMT"
DEFAULT_PEERS  = ["NOC", "RTX", "GD", "BA"]  # Northrop, Raytheon, General Dynamics, Boeing

# -------------------- Helpers --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def price_history(tickers, period="3y", interval="1d"):
    """
    Wrap yfinance download and normalize columns:
    - Multiple tickers -> MultiIndex columns (['Open','High','Low','Close','Volume'], ticker)
      Access like data['Close'] -> DataFrame of closes by ticker.
    - Single ticker    -> drop ticker level so columns are simple: Open, High, Low, Close, Volume
    """
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)

    # If user passed a single ticker as a list/tuple, drop the ticker level for single-level columns
    if isinstance(tickers, (list, tuple, set)) and len(tickers) == 1:
        if isinstance(data.columns, pd.MultiIndex) and data.columns.nlevels == 2:
            data = data.droplevel(1, axis=1)

    return data

@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker: str):
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
        }
    except Exception:
        return {"market_cap": None, "pe": None, "eps": None}

def fmt_money(n):
    if not n:
        return "—"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"${n:,.0f}{unit}"
        n /= 1000.0
    return f"${n:,.0f}"

def metric_card(label, value):
    st.markdown(
        f'<div class="card"><div style="color:#6b7280;font-size:.9rem;">{label}</div>'
        f'<div style="font-size:1.8rem;font-weight:700;margin-top:.2rem;">{value}</div></div>',
        unsafe_allow_html=True
    )

def normalize(close_df: pd.DataFrame) -> pd.DataFrame:
    return (close_df / close_df.iloc[0] * 100).dropna()

def compute_drawdown(close: pd.Series) -> pd.Series:
    peak = close.cummax()
    return (close / peak - 1.0) * 100.0

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_safe(ticker: str, limit: int = 12):
    """Fetch news and skip malformed items to avoid KeyErrors."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    rows = []
    for n in raw[:limit]:
        title = n.get("title")
        link = n.get("link")
        if not title or not link:
            continue
        publisher = n.get("publisher") or "Source"
        ts = n.get("providerPublishTime")
        when = ""
        if ts:
            when = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        rows.append({"title": title, "link": link, "publisher": publisher, "when": when})
    return rows

# -------------------- Sidebar --------------------
st.sidebar.subheader("Ticker")
ticker = st.sidebar.text_input("", value=DEFAULT_TICKER).upper().strip()

st.sidebar.subheader("Competitors")
peers = st.sidebar.multiselect("Competitors", DEFAULT_PEERS, default=DEFAULT_PEERS, label_visibility="collapsed")

st.sidebar.subheader("Ask a question")
question = st.sidebar.text_area("Ask a question about this stock", label_visibility="collapsed")
send = st.sidebar.button("Send", use_container_width=True)

# -------------------- Tabs --------------------
tabs = st.tabs(["Financials", "Stock", "Headlines"])

# -------------------- Financials Tab --------------------
with tabs[0]:
    info = get_info(ticker)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Market Cap", fmt_money(info["market_cap"]))
    with c2: metric_card("P/E Ratio (TTM)", f"{info['pe']:.1f}" if info["pe"] else "—")
    with c3: metric_card("EPS (TTM)", f"{info['eps']:.2f}" if info["eps"] else "—")

    # Comparison chart (Indexed to 100) for LMT + peers (3Y)
    st.markdown("#### Comparison Chart")
    compare_list = [ticker] + peers
    prices = price_history(compare_list, period="3y")
    # With multiple tickers, prices["Close"] returns a DataFrame of closes by ticker (single-level columns)
    close_df = prices["Close"] if "Close" in prices else prices
    norm = normalize(close_df)

    fig = px.line(norm.reset_index(), x="Date", y=norm.columns,
                  title="Price Performance (Indexed to 100)", template="plotly_white")
    fig.update_layout(height=420, legend_title_text="", margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_title="Indexed Price", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Stock Tab (sub-tabs) --------------------
with tabs[1]:
    st.markdown(f"### {ticker} Analytics")
    sub1, sub2, sub3, sub4, sub5 = st.tabs(
        ["Price (1Y)", "Candlestick + Volume", "Drawdown", "Rolling Metrics", "Peer Performance"]
    )

    data_3y = price_history([ticker], period="3y")  # single ticker -> single-level columns
    data_1y = price_history([ticker], period="1y")

    # ----- Price (1Y) -----
    with sub1:
        close_1y = data_1y["Close"]
        ma50  = close_1y.rolling(50).mean()
        ma200 = close_1y.rolling(200).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=close_1y.index, y=close_1y, mode="lines", name=f"{ticker} Close"))
        fig2.add_trace(go.Scatter(x=ma50.index,  y=ma50,  mode="lines", name="50-day MA"))
        fig2.add_trace(go.Scatter(x=ma200.index, y=ma200, mode="lines", name="200-day MA"))
        fig2.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=20, b=10),
                           xaxis_title=None, yaxis_title="Price")
        st.plotly_chart(fig2, use_container_width=True)

    # ----- Candlestick + Volume -----
    with sub2:
        # Robust selection regardless of yfinance quirks (already single-level, but keep guard)
        if isinstance(data_1y.columns, pd.MultiIndex):
            ohlc = data_1y.xs(["Open","High","Low","Close"], axis=1, level=0)
            ohlc = ohlc.droplevel(1, axis=1)  # drop ticker level
            ohlc = ohlc[["Open","High","Low","Close"]]
            vol  = data_1y.xs("Volume", axis=1, level=0).squeeze()
        else:
            ohlc = data_1y[["Open","High","Low","Close"]]
            vol  = data_1y["Volume"]

        fig3 = go.Figure()
        fig3.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc["Open"], high=ohlc["High"], low=ohlc["Low"], close=ohlc["Close"],
            name="OHLC"
        ))
        fig3.add_trace(go.Bar(x=vol.index, y=vol, name="Volume", yaxis="y2", opacity=0.3))
        fig3.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title="Price",
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume")
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ----- Drawdown (3Y) -----
    with sub3:
        close_3y = data_3y["Close"]
        dd = compute_drawdown(close_3y)
        fig4 = px.area(dd, x=dd.index, y=dd.values, template="plotly_white",
                       labels={"x":"Date","y":"Drawdown (%)"}, title="Peak-to-Trough Drawdown (3Y)")
        fig4.update_traces(hovertemplate="Drawdown: %{y:.2f}%<extra></extra>")
        fig4.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # ----- Rolling Metrics -----
    with sub4:
        close_3y = data_3y["Close"]
        ret = close_3y.pct_change()
        roll_vol = (ret.rolling(20).std() * np.sqrt(252)) * 100  # annualized %
        roll_ret = (close_3y / close_3y.shift(60) - 1) * 100     # ~3 months

        cA, cB = st.columns(2)
        with cA:
            fig5 = px.line(roll_vol, x=roll_vol.index, y=roll_vol.values, template="plotly_white",
                           labels={"x":"Date","y":"Volatility (%)"}, title="Rolling 20D Annualized Volatility")
            fig5.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig5, use_container_width=True)
        with cB:
            fig6 = px.line(roll_ret, x=roll_ret.index, y=roll_ret.values, template="plotly_white",
                           labels={"x":"Date","y":"Return (%)"}, title="Rolling 60D Return")
            fig6.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig6, use_container_width=True)

    # ----- Peer Performance (YTD) -----
    with sub5:
        compare = [ticker] + peers
        ytd = price_history(compare, period="ytd")
        ytd_close = ytd["Close"] if "Close" in ytd else ytd
        perf = ((ytd_close.iloc[-1] / ytd_close.iloc[0] - 1) * 100).round(2).sort_values(ascending=False)
        fig7 = px.bar(perf, x=perf.index, y=perf.values, text=perf.values,
                      template="plotly_white", labels={"x":"Ticker","y":"YTD %"},
                      title="YTD Performance vs Peers")
        fig7.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig7.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), yaxis_ticksuffix="%")
        st.plotly_chart(fig7, use_container_width=True)

# -------------------- Headlines Tab --------------------
with tabs[2]:
    st.markdown("#### Recent Headlines")
    news_items = get_news_safe(ticker, limit=12)
    if not news_items:
        st.info("No headlines available right now.")
    else:
        for n in news_items:
            st.markdown(f"**[{n['title']}]({n['link']})** — *{n['publisher']}*  {n['when']}")

# -------------------- Sidebar Q&A (placeholder) --------------------
if send and question.strip():
    st.sidebar.markdown("**Answer**")
    st.sidebar.write("Quick Q&A placeholder — we can wire this to an LLM later.")
