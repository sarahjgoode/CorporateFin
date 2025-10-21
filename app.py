 # app.py
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -------------------- Page Config --------------------
st.set_page_config(page_title="Lockheed Martin Dashboard", page_icon="📈", layout="wide")

# Minimal styling + tab label visibility helpers
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {background:#fff;border:1px solid #eee;border-radius:12px;padding:16px 20px;box-shadow:0 1px 2px rgba(0,0,0,0.05);}
.stTabs [data-baseweb="tab"] { color:#333 !important; font-weight:600 !important; font-size:1rem !important; }
.stTabs [aria-selected="true"] { color:#000 !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- Defaults --------------------
DEFAULT_TICKER = "LMT"
DEFAULT_PEERS  = ["NOC", "RTX", "GD", "BA"]

# -------------------- Helpers --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def price_history(tickers, period="3y", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)
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

# -------- Fundamentals (quarterly) helpers --------
@st.cache_data(ttl=3600, show_spinner=False)
def get_quarterly_frames(ticker: str):
    """
    Fetch quarterly financials/balance sheet/cashflow.
    Return tidy DataFrames with datetime index ascending.
    Never truth-test DataFrames; normalize None/empty to empty DFs.
    """
    t = yf.Ticker(ticker)

    fin = t.quarterly_financials
    bs  = t.quarterly_balance_sheet
    cf  = t.quarterly_cashflow

    def _norm(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        df = df.T.copy()
        # yfinance uses period labels; coerce to datetime (NaT ok)
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        df.sort_index(inplace=True)
        return df

    fin = _norm(fin)
    bs  = _norm(bs)
    cf  = _norm(cf)
    return fin, bs, cf


def series_or_none(df: pd.DataFrame, col: str):
    return df[col] if (not df.empty and col in df.columns) else None

def avg_of_series(s: pd.Series, window: int = 2):
    """Rolling average to approximate average balance sheet values between periods."""
    return s.rolling(window).mean()

def ratio_line(y, title, y_is_pct=True):
    """Plot a single-line series with sensible formatting."""
    if y is None or y.dropna().empty:
        st.info("Not enough data to display this metric.")
        return
    label_y = "Value (%)" if y_is_pct else "Value"
    fig = px.line(y, x=y.index, y=y.values, template="plotly_white",
                  labels={"x":"Date", "y":label_y}, title=title)
    if y_is_pct:
        fig.update_layout(yaxis_ticksuffix="%")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

def pick_col(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    """Return the first existing column as a Series (or None)."""
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.columns:
            s = df[name].copy()
            # numeric if possible
            with pd.option_context("future.no_silent_downcasting", True):
                s = pd.to_numeric(s, errors="coerce")
            return s
    return None

def align(*series: pd.Series) -> list[pd.Series]:
    """Align multiple series to their common index (inner join)."""
    idx = None
    for s in series:
        if s is not None and not s.dropna().empty:
            idx = s.index if idx is None else idx.intersection(s.index)
    out = []
    for s in series:
        if s is None or idx is None:
            out.append(None)
        else:
            out.append(s.reindex(idx))
    return out

def safe_ratio(numer: pd.Series | None, denom: pd.Series | None, pct: bool = True) -> pd.Series | None:
    if numer is None or denom is None:
        return None
    n, d = align(numer, denom)
    if n is None or d is None:
        return None
    d = d.replace(0, np.nan)  # avoid divide-by-zero
    r = n / d
    if pct:
        r = r * 100.0
    return r.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def compute_fundamental_series(ticker: str):
    fin, bs, cf = get_quarterly_frames(ticker)

    # Income statement
    revenue      = pick_col(fin, "Total Revenue", "Operating Revenue")
    net_income   = pick_col(fin, "Net Income")
    gross_profit = pick_col(fin, "Gross Profit")

    # Balance sheet (use multiple common name variants)
    equity = (
        pick_col(bs,
                 "Stockholders Equity",
                 "Total Stockholder Equity",
                 "Total Stockholders Equity",
                 "Total Equity Gross Minority Interest")
    )
    total_assets = pick_col(bs, "Total Assets")
    curr_assets  = pick_col(bs, "Total Current Assets")
    curr_liab    = pick_col(bs, "Total Current Liabilities")

    total_debt = pick_col(bs, "Total Debt", "Net Debt")
    if total_debt is None:
        short_debt = pick_col(bs, "Short Long Term Debt", "Short Term Debt")
        long_debt  = pick_col(bs, "Long Term Debt")
        if short_debt is not None or long_debt is not None:
            short_debt = short_debt.fillna(0) if short_debt is not None else 0
            long_debt  = long_debt.fillna(0)  if long_debt  is not None else 0
            total_debt = short_debt + long_debt

    # Cash flow
    op_cf = pick_col(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = pick_col(cf, "Capital Expenditure", "Capital Expenditures")
    fcf   = None
    if op_cf is not None and capex is not None:
        # CapEx is negative in yfinance; FCF = CFO - CapEx = CFO + (negative CapEx)
        fcf = op_cf + capex

    # Averages for ROE/ROA (2-period rolling mean)
    equity_avg = equity.rolling(2).mean() if equity is not None else None
    assets_avg = total_assets.rolling(2).mean() if total_assets is not None else None

    # Ratios
    roe          = safe_ratio(net_income, equity_avg, pct=True)     # %
    current_ratio= safe_ratio(curr_assets, curr_liab, pct=False)    # x
    net_margin   = safe_ratio(net_income, revenue, pct=True)        # %
    roa          = safe_ratio(net_income, assets_avg, pct=True)     # %
    de_ratio     = safe_ratio(total_debt, equity, pct=False)        # x
    fcf_margin   = safe_ratio(fcf, revenue, pct=True)               # %
    gross_margin = safe_ratio(gross_profit, revenue, pct=True)      # %

    def trim(s):  # last ~20 quarters
        return s.tail(20) if s is not None else None

    return {
        "roe": trim(roe),
        "current_ratio": trim(current_ratio),
        "net_margin": trim(net_margin),
        "roa": trim(roa),
        "de_ratio": trim(de_ratio),
        "fcf_margin": trim(fcf_margin),
        "gross_margin": trim(gross_margin),
    }


# -------------------- Sidebar --------------------
st.sidebar.subheader("Ticker")
ticker = st.sidebar.text_input("", value=DEFAULT_TICKER).upper().strip()

st.sidebar.subheader("Competitors")
peers = st.sidebar.multiselect("Competitors", DEFAULT_PEERS, default=DEFAULT_PEERS, label_visibility="collapsed")

st.sidebar.subheader("Ask a question")
question = st.sidebar.text_area("Ask a question about this stock", label_visibility="collapsed")
send = st.sidebar.button("Send", use_container_width=True)

# -------------------- Main Tabs --------------------
tabs = st.tabs(["💼 Financial Overview", "📊 LMT Analytics", "📑 Fundamentals"])

# -------------------- Financial Overview --------------------
with tabs[0]:
    st.markdown("# 💼 Lockheed Martin — Financial Overview")
    info = get_info(ticker)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Market Cap", fmt_money(info["market_cap"]))
    with c2: metric_card("P/E Ratio (TTM)", f"{info['pe']:.1f}" if info["pe"] else "—")
    with c3: metric_card("EPS (TTM)", f"{info['eps']:.2f}" if info["eps"] else "—")

    # Comparison chart (3Y, indexed to 100)
    st.markdown("### 📊 Comparison Chart — LMT vs Competitors")
    compare_list = [ticker] + peers
    prices = price_history(compare_list, period="3y")
    close_df = prices["Close"] if "Close" in prices else prices
    norm = normalize(close_df)

    fig = px.line(norm.reset_index(), x="Date", y=norm.columns,
                  title="Price Performance (Indexed to 100)", template="plotly_white")
    fig.update_layout(height=420, legend_title_text="", margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_title="Indexed Price", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- LMT Analytics --------------------
with tabs[1]:
    st.markdown("# 📊 LMT Analytics")
    st.info("**OHLC** = **O**pen, **H**igh, **L**ow, **C**lose — the daily trading range of the stock.")

    sub1, sub2, sub3, sub4, sub5 = st.tabs(
        ["Price (1Y)", "Candlestick + Volume", "Drawdown", "Rolling Metrics", "Peer Performance"]
    )

    data_3y = price_history([ticker], period="3y")
    data_1y = price_history([ticker], period="1y")

    # Price (1Y)
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

    # Candlestick + Volume
    with sub2:
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

    # Drawdown
    with sub3:
        close_3y = data_3y["Close"]
        dd = compute_drawdown(close_3y)
        fig4 = px.area(dd, x=dd.index, y=dd.values, template="plotly_white",
                       labels={"x":"Date","y":"Drawdown (%)"}, title="Peak-to-Trough Drawdown (3Y)")
        fig4.update_traces(hovertemplate="Drawdown: %{y:.2f}%<extra></extra>")
        fig4.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Rolling metrics
    with sub4:
        close_3y = data_3y["Close"]
        ret = close_3y.pct_change()
        roll_vol = (ret.rolling(20).std() * np.sqrt(252)) * 100
        roll_ret = (close_3y / close_3y.shift(60) - 1) * 100
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

    # Peer performance (YTD)
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

# -------------------- Fundamentals (Ratios) --------------------
with tabs[2]:
    st.markdown("# 📑 Fundamentals (Quarterly Ratios)")
    ratios = compute_fundamental_series(ticker)

    s1, s2, s3, s4, s5, s6, s7 = st.tabs(
        ["ROE", "Current Ratio", "Net Profit Margin", "ROA", "Debt-to-Equity", "FCF Margin", "Gross Profit Margin"]
    )

    with s1: ratio_line(ratios["roe"], "Return on Equity (ROE)", y_is_pct=True)
    with s2: ratio_line(ratios["current_ratio"], "Current Ratio", y_is_pct=False)
    with s3: ratio_line(ratios["net_margin"], "Net Profit Margin", y_is_pct=True)
    with s4: ratio_line(ratios["roa"], "Return on Assets (ROA)", y_is_pct=True)
    with s5: ratio_line(ratios["de_ratio"], "Debt-to-Equity (x)", y_is_pct=False)
    with s6: ratio_line(ratios["fcf_margin"], "Free Cash Flow Margin", y_is_pct=True)
    with s7: ratio_line(ratios["gross_margin"], "Gross Profit Margin", y_is_pct=True)

# -------------------- Sidebar Q&A (placeholder) --------------------
if send and question.strip():
    st.sidebar.markdown("**Answer**")
    st.sidebar.write("Quick Q&A placeholder — can be wired to an LLM later.")
