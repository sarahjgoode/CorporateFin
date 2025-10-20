import math
from datetime import datetime, timezone
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Defaults for Lockheed Martin view
DEFAULT_TICKER = "LMT"
DEFAULT_PEERS = ["NOC", "RTX", "GD", "BA"]  # Northrop, Raytheon, General Dynamics, Boeing

# -------------------- Page Config --------------------
st.set_page_config(page_title="Company Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {background:#fff;border:1px solid #eee;border-radius:12px;padding:16px 20px;box-shadow:0 1px 2px rgba(0,0,0,0.05);}
.legend-row {display:flex;gap:1.25rem;align-items:center;margin-top:.5rem;}
.legend-dot {width:10px;height:10px;border-radius:999px;display:inline-block;}
</style>
""", unsafe_allow_html=True)

# -------------------- Functions --------------------
@st.cache_data(ttl=3600)
def get_info(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        return {
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
        }
    except Exception:
        return {"market_cap": None, "pe": None, "eps": None}

@st.cache_data(ttl=3600)
def price_history(tickers, period="3y"):
    df = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers)
    return df.dropna(how="all")

def fmt_money(n):
    if not n: return "—"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"${n:,.0f}{unit}"
        n /= 1000.0
    return f"${n:,.0f}"

def metric_card(label, value):
    st.markdown(f'<div class="card"><div style="color:#6b7280;font-size:.9rem;">{label}</div>'
                f'<div style="font-size:1.8rem;font-weight:700;margin-top:.2rem;">{value}</div></div>',
                unsafe_allow_html=True)

def normalize(df):
    return (df / df.iloc[0] * 100).dropna()

# -------------------- Sidebar --------------------
# --- Sidebar ---
st.sidebar.subheader("Ticker")
ticker = st.sidebar.text_input("", value=DEFAULT_TICKER).upper().strip()

st.sidebar.subheader("Competitors")
peers = st.sidebar.multiselect(
    "Competitors",
    DEFAULT_PEERS,            # options shown
    default=DEFAULT_PEERS,    # preselected
    label_visibility="collapsed"
)

# -------------------- Tabs --------------------
tabs = st.tabs(["Financials", "Stock", "Headlines"])

# -------------------- Financials --------------------
with tabs[0]:
    info = get_info(ticker)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Market Cap", fmt_money(info["market_cap"]))
    with c2: metric_card("P/E Ratio (TTM)", f"{info['pe']:.1f}" if info["pe"] else "—")
    with c3: metric_card("EPS (TTM)", f"{info['eps']:.2f}" if info["eps"] else "—")

        # --- Comparison Chart ---
    st.markdown("#### Comparison Chart")

    # Combine current ticker with selected competitors
    compare_list = [ticker] + peers

    # Pull historical price data from yfinance
    prices = price_history(compare_list, period="3y")

    # Normalize prices (index all at 100 to compare growth)
    norm = (prices / prices.iloc[0] * 100).dropna()

    # Create interactive line chart
    fig = px.line(
        norm.reset_index(),
        x="Date",
        y=norm.columns,
        title="Price Performance (Indexed to 100)",
        template="plotly_white"
    )

    fig.update_layout(
        height=420,
        legend_title_text="",
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="Indexed Price",
        xaxis_title=None,
    )

    st.plotly_chart(fig, use_container_width=True)


# -------------------- Stock Tab --------------------
with tabs[1]:
    st.markdown(f"#### {ticker} 1-Year Performance")
    df = price_history([ticker], period="1y")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df[ticker], mode="lines", name=f"{ticker} Close"))
    fig2.update_layout(template="plotly_white", height=400, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# -------------------- Headlines Tab --------------------
with tabs[2]:
    st.markdown("#### Latest News")
    news = yf.Ticker(ticker).news
    if news:
        for n in news[:8]:
            st.markdown(f"**[{n['title']}]({n['link']})** — *{n['publisher']}*")
    else:
        st.info("No recent headlines found.")

# -------------------- Question Box --------------------
if send and question.strip():
    st.sidebar.markdown("**Answer**")
    st.sidebar.write("This section could later be connected to Gemini/OpenAI for live Q&A.")
