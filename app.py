import os
import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PROANALİZ FINAL
# BIST + KRİPTO + 100X RADAR
# ============================================================

st.set_page_config(
    page_title="ProAnaliz | BIST & Kripto Terminali",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- CSS ---------------------------
st.markdown("""
<style>
.stApp { background:#090d14; color:#e8eef7; }
.block-container { max-width:1500px; padding-top:1.2rem; }
[data-testid="stMetric"] {
    background:linear-gradient(135deg,#121a27,#182337);
    border:1px solid #26344b; border-radius:14px; padding:14px;
}
[data-testid="stMetricValue"] { font-size:1.55rem; }
.card {
    background:linear-gradient(135deg,#101722,#151f2f);
    border:1px solid #26344b; border-radius:14px;
    padding:16px; margin-bottom:10px;
}
.score {
    font-size:34px; font-weight:800; line-height:1.1;
}
.muted { color:#91a0b7; font-size:.86rem; }
.good { color:#22c55e; }
.warn { color:#eab308; }
.bad { color:#ef4444; }
.pill {
    display:inline-block; padding:4px 8px; border-radius:8px;
    background:#1b2a40; margin:2px; font-size:.78rem;
}
.small { font-size:.75rem; color:#8fa0b8; }
</style>
""", unsafe_allow_html=True)

# ----------------------- CONFIG -------------------------------
BINANCE_BASES = [
    "https://data-api.binance.vision/api/v3",
    "https://api.binance.com/api/v3",
    "https://api1.binance.com/api/v3",
]
CG_BASE = "https://api.coingecko.com/api/v3"

CRYPTO_SYMBOLS = [
    "BTC","ETH","BNB","SOL","XRP","ADA","DOGE","AVAX","DOT","LINK",
    "NEAR","APT","SUI","ARB","OP","ATOM","INJ","SEI","TIA","FIL"
]

BLOCKED_SYMBOLS = {
    "USDT","USDC","FDUSD","DAI","USDE","XAUT","PAXG","TUSD","USDD"
}

# ----------------------- HELPERS ------------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def money(x):
    x = safe_float(x)
    if not np.isfinite(x):
        return "-"
    a = abs(x)
    if a >= 1e12: return f"${x/1e12:.2f}T"
    if a >= 1e9: return f"${x/1e9:.2f}B"
    if a >= 1e6: return f"${x/1e6:.2f}M"
    if a >= 1e3: return f"${x/1e3:.2f}K"
    return f"${x:.2f}"

def fmt(x, n=2):
    x = safe_float(x)
    return "-" if not np.isfinite(x) else f"{x:,.{n}f}"

def request_json(url, params=None, headers=None, timeout=12):
    r = requests.get(
        url,
        params=params,
        headers=headers or {"User-Agent":"ProAnaliz/Final"},
        timeout=timeout
    )
    r.raise_for_status()
    return r.json()

def ema(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    s = pd.Series(series).astype(float)
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd(series):
    s = pd.Series(series).astype(float)
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist

def adx(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
    tr = pd.concat([
        h-l,
        (h-c.shift()).abs(),
        (l-c.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0,np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0,np.nan)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

def add_indicators(df):
    df = df.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = ema(df["Close"],20)
    df["EMA50"] = ema(df["Close"],50)
    df["RSI"] = rsi(df["Close"],14)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
    df["ADX"] = adx(df,14)
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    return df.dropna(subset=["Close"])

def technical_score(df):
    if len(df) < 205:
        return None

    x = df.iloc[-1]
    p = df.iloc[-2]
    score = 50
    reasons = []

    # Trend: maximum +25
    if x["Close"] > x["EMA20"]:
        score += 7; reasons.append("Fiyat EMA20 üstünde")
    else:
        score -= 6; reasons.append("Fiyat EMA20 altında")

    if x["EMA20"] > x["EMA50"]:
        score += 7; reasons.append("EMA20 > EMA50")
    else:
        score -= 5

    if x["Close"] > x["SMA200"]:
        score += 6; reasons.append("Fiyat SMA200 üstünde")
    else:
        score -= 6

    if x["SMA50"] > x["SMA200"]:
        score += 5; reasons.append("SMA50 > SMA200")
    else:
        score -= 4

    # RSI: reward healthy momentum, penalize extremes
    rv = x["RSI"]
    if 50 <= rv <= 68:
        score += 8; reasons.append(f"RSI sağlıklı ({rv:.1f})")
    elif 42 <= rv < 50:
        score += 3
    elif rv > 78:
        score -= 9; reasons.append(f"RSI aşırı yüksek ({rv:.1f})")
    elif rv < 30:
        score += 2; reasons.append(f"RSI aşırı satım ({rv:.1f})")
    else:
        score -= 2

    # MACD
    if x["MACD_Hist"] > 0:
        score += 8; reasons.append("MACD pozitif")
    else:
        score -= 6; reasons.append("MACD negatif")
    if p["MACD"] <= p["MACD_Signal"] and x["MACD"] > x["MACD_Signal"]:
        score += 7; reasons.append("MACD yukarı kesişim")
    elif p["MACD"] >= p["MACD_Signal"] and x["MACD"] < x["MACD_Signal"]:
        score -= 7; reasons.append("MACD aşağı kesişim")

    # ADX
    if x["ADX"] >= 25:
        score += 6; reasons.append(f"ADX trendi destekliyor ({x['ADX']:.1f})")

    # Volume
    if x["VolMA20"] and x["Volume"] > x["VolMA20"] * 1.5:
        score += 5; reasons.append("Hacim ortalamanın üzerinde")

    score = int(max(0, min(100, round(score))))
    return {
        "score": score,
        "rsi": float(x["RSI"]),
        "adx": float(x["ADX"]),
        "ema20": float(x["EMA20"]),
        "ema50": float(x["EMA50"]),
        "sma50": float(x["SMA50"]),
        "sma200": float(x["SMA200"]),
        "macd": float(x["MACD"]),
        "macd_signal": float(x["MACD_Signal"]),
        "macd_hist": float(x["MACD_Hist"]),
        "volume": float(x["Volume"]),
        "vol_ratio": float(x["Volume"]/x["VolMA20"]) if x["VolMA20"] else np.nan,
        "reasons": reasons,
    }

# ----------------------- BINANCE -------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def binance_klines(symbol, interval="4h", limit=250):
    last = None
    for base in BINANCE_BASES:
        try:
            data = request_json(
                f"{base}/klines",
                {"symbol":symbol.upper(), "interval":interval, "limit":limit},
                timeout=10
            )
            rows = []
            for k in data:
                rows.append([
                    pd.to_datetime(k[0], unit="ms"),
                    float(k[1]), float(k[2]), float(k[3]),
                    float(k[4]), float(k[5])
                ])
            return pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"]).set_index("Date")
        except Exception as e:
            last = e
    raise RuntimeError(f"Binance verisi alınamadı: {last}")

@st.cache_data(ttl=15, show_spinner=False)
def binance_price(symbol):
    for base in BINANCE_BASES:
        try:
            d = request_json(f"{base}/ticker/price", {"symbol":symbol.upper()}, timeout=5)
            return float(d["price"])
        except Exception:
            continue
    return None

# ----------------------- COINGECKO -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def coingecko_markets(pages=4):
    key = os.getenv("COINGECKO_API_KEY","").strip()
    headers = {"User-Agent":"ProAnaliz/Final"}
    if key:
        headers["x-cg-demo-api-key"] = key

    all_rows = []
    for page in range(1, pages+1):
        try:
            d = request_json(
                f"{CG_BASE}/coins/markets",
                {
                    "vs_currency":"usd",
                    "order":"market_cap_desc",
                    "per_page":250,
                    "page":page,
                    "sparkline":"false",
                    "price_change_percentage":"24h"
                },
                headers=headers,
                timeout=20
            )
            all_rows.extend(d)
            if len(d) < 250:
                break
        except Exception:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    needed = [
        "id","symbol","name","current_price","market_cap",
        "fully_diluted_valuation","total_volume","ath","ath_change_percentage",
        "price_change_percentage_24h"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df[needed].copy()

# ----------------------- BIST ---------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_bist_list():
    url = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/Data.aspx/HisseOzet"
    try:
        d = request_json(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        vals = d.get("value", [])
        codes = sorted({str(x.get("kod","")).upper() for x in vals if x.get("kod")})
        return codes if codes else ["THYAO","EREGL","GARAN","AKBNK","TUPRS","ASELS","KCHOL","BIMAS","SISE","FROTO"]
    except Exception:
        return ["THYAO","EREGL","GARAN","AKBNK","TUPRS","ASELS","KCHOL","BIMAS","SISE","FROTO"]

@st.cache_data(ttl=300, show_spinner=False)
def bist_history(symbol):
    df = yf.download(
        f"{symbol}.IS",
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open","High","Low","Close","Volume"]
    return df[[c for c in cols if c in df.columns]].dropna()

@st.cache_data(ttl=900, show_spinner=False)
def bist_fundamentals(symbol):
    try:
        info = yf.Ticker(f"{symbol}.IS").info
        return {
            "pe": safe_float(info.get("trailingPE")),
            "pb": safe_float(info.get("priceToBook")),
            "market_cap": safe_float(info.get("marketCap")),
            "sector": info.get("sector","-"),
            "name": info.get("longName",symbol)
        }
    except Exception:
        return {"pe":np.nan,"pb":np.nan,"market_cap":np.nan,"sector":"-","name":symbol}

def fundamental_score(f):
    score = 50
    reasons = []
    pe, pb = f["pe"], f["pb"]

    if np.isfinite(pe) and pe > 0:
        if pe <= 8: score += 20; reasons.append(f"Cazip F/K ({pe:.1f})")
        elif pe <= 15: score += 10
        elif pe <= 25: score += 2
        else: score -= 10; reasons.append(f"Yüksek F/K ({pe:.1f})")

    if np.isfinite(pb) and pb > 0:
        if pb <= 1.2: score += 20; reasons.append(f"İskontolu PD/DD ({pb:.2f})")
        elif pb <= 2.5: score += 10
        elif pb <= 5: score += 2
        else: score -= 8; reasons.append(f"Yüksek PD/DD ({pb:.2f})")

    return int(max(0,min(100,round(score)))), reasons

# ----------------------- 100X ENGINE ---------------------------
def hundred_x_score(row, tech):
    mc = safe_float(row.get("market_cap"))
    fdv = safe_float(row.get("fully_diluted_valuation"))
    vol = safe_float(row.get("total_volume"))
    change = safe_float(row.get("price_change_percentage_24h"))

    if not np.isfinite(mc) or mc <= 0:
        return None

    if not np.isfinite(fdv) or fdv <= 0:
        fdv = mc
    if not np.isfinite(vol) or vol < 0:
        vol = 0

    # Market cap: smaller = more room, but no "free 100X"
    a = 35 if mc <= 25e6 else 32 if mc <= 50e6 else 28 if mc <= 100e6 else \
        23 if mc <= 250e6 else 18 if mc <= 500e6 else 12 if mc <= 1e9 else \
        5 if mc <= 5e9 else 0

    dilution = fdv / mc
    b = 20 if dilution <= 1.10 else 17 if dilution <= 1.25 else \
        14 if dilution <= 1.50 else 9 if dilution <= 2 else \
        4 if dilution <= 3 else 0

    liq = vol / mc if mc else 0
    c = 15 if liq >= .20 else 13 if liq >= .10 else 11 if liq >= .05 else \
        8 if liq >= .02 else 5 if liq >= .01 else 0

    target = mc * 100
    d = 15 if target <= 2.5e9 else 13 if target <= 5e9 else \
        10 if target <= 10e9 else 6 if target <= 25e9 else \
        3 if target <= 50e9 else 0

    e = 10 if np.isfinite(change) and 2 <= change <= 15 else \
        6 if np.isfinite(change) and 15 < change <= 30 else \
        2 if np.isfinite(change) and change > 30 else \
        5 if np.isfinite(change) and change >= 0 else 0

    raw = a+b+c+d+e
    score = int(max(0,min(100,round(raw/95*100))))

    # Technical confirmation, not an automatic guarantee
    if tech:
        if tech["score"] >= 75: score = min(100,score+5)
        elif tech["score"] < 45: score = max(0,score-7)
        if tech["rsi"] > 80: score = max(0,score-10)
        if tech["vol_ratio"] >= 1.5: score = min(100,score+3)

    risk = "ÇOK YÜKSEK" if mc < 50e6 else "YÜKSEK" if mc < 1e9 else "ORTA"
    return {
        "score":score,
        "mc":mc,
        "fdv":fdv,
        "volume":vol,
        "liq":liq,
        "target":target,
        "risk":risk,
        "change":change,
        "dilution":dilution,
    }

def signal(score):
    if score >= 80: return "🟢 GÜÇLÜ AL","good"
    if score >= 60: return "🟩 AL","good"
    if score >= 46: return "🟡 NÖTR / TUT","warn"
    if score >= 26: return "🟧 SAT","bad"
    return "🔴 GÜÇLÜ SAT","bad"

# ----------------------- UI SIDEBAR ----------------------------
st.sidebar.title("📈 ProAnaliz")
st.sidebar.caption("BIST + Kripto + 100X Radar")

market = st.sidebar.radio("Piyasa",["₿ Kripto","🇹🇷 BIST","🔥 100X Radar"])

st.sidebar.divider()
st.sidebar.caption("Skor: 80+ Güçlü Al · 60-79 Al · 46-59 Nötr · 26-45 Sat · 0-25 Güçlü Sat")

if st.sidebar.button("🔄 Verileri Yenile", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ============================================================
# KRİPTO TEK VARLIK
# ============================================================
if market == "₿ Kripto":
    st.title("₿ Kripto Terminali")
    symbol = st.sidebar.selectbox("Kripto",CRYPTO_SYMBOLS)
    interval = st.sidebar.selectbox("Zaman Dilimi",["15m","1h","4h","1d"],index=2)

    try:
        raw = binance_klines(symbol+"USDT",interval,250)
        df = add_indicators(raw)
        tech = technical_score(df)
        price = binance_price(symbol+"USDT") or float(df.iloc[-1]["Close"])

        sig, cls = signal(tech["score"])
        st.subheader(f"{symbol} / USDT")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Fiyat",fmt(price,8))
        c2.metric("Teknik Skor",f'{tech["score"]}/100')
        c3.metric("RSI",fmt(tech["rsi"],1))
        c4.metric("ADX",fmt(tech["adx"],1))
        c5.metric("Sinyal",sig)

        st.progress(tech["score"]/100)

        left,right = st.columns(2)
        with left:
            st.markdown("### 🧠 Teknik Sinyaller")
            for r in tech["reasons"]:
                st.info("• "+r)
        with right:
            st.markdown("### 📊 İndikatörler")
            st.dataframe(pd.DataFrame({
                "Gösterge":["EMA20","EMA50","SMA50","SMA200","RSI","ADX","MACD","MACD Signal"],
                "Değer":[tech["ema20"],tech["ema50"],tech["sma50"],tech["sma200"],tech["rsi"],tech["adx"],tech["macd"],tech["macd_signal"]]
            }).style.format({"Değer":"{:.6f}"}),use_container_width=True,hide_index=True)

        fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                            row_heights=[.55,.22,.23],vertical_spacing=.04,
                            subplot_titles=("Fiyat / EMA / SMA","RSI","MACD"))
        fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Fiyat"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.EMA20,name="EMA20"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.SMA50,name="SMA50"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.SMA200,name="SMA200"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.RSI,name="RSI"),row=2,col=1)
        fig.add_hline(y=70,line_dash="dash",row=2,col=1)
        fig.add_hline(y=30,line_dash="dash",row=2,col=1)
        fig.add_trace(go.Bar(x=df.index,y=df.MACD_Hist,name="Histogram"),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.MACD,name="MACD"),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.MACD_Signal,name="Signal"),row=3,col=1)
        fig.update_layout(height=780,template="plotly_dark",paper_bgcolor="#090d14",
                          plot_bgcolor="#101722",xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)

    except Exception as e:
        st.error(f"Veri/analiz hatası: {e}")

# ============================================================
# BIST TEK VARLIK
# ============================================================
elif market == "🇹🇷 BIST":
    st.title("🇹🇷 BIST Terminali")
    bist = get_bist_list()
    symbol = st.sidebar.selectbox("Hisse",bist)

    try:
        raw = bist_history(symbol)
        if raw.empty:
            raise RuntimeError("Hisse verisi boş geldi.")
        df = add_indicators(raw)
        tech = technical_score(df)
        f = bist_fundamentals(symbol)
        fscore, freasons = fundamental_score(f)
        total = round(tech["score"]*0.55 + fscore*0.45)
        sig, cls = signal(total)

        st.subheader(f"{f['name']} · {symbol}.IS")
        a,b,c,d,e = st.columns(5)
        a.metric("Fiyat",fmt(df.iloc[-1]["Close"],2)+" ₺")
        b.metric("Genel Skor",f"{total}/100")
        c.metric("Teknik",f'{tech["score"]}/100')
        d.metric("F/K",fmt(f["pe"],2))
        e.metric("PD/DD",fmt(f["pb"],2))
        st.progress(total/100)
        st.success(sig) if cls=="good" else st.warning(sig) if cls=="warn" else st.error(sig)

        l,r=st.columns(2)
        with l:
            st.markdown("### 🧠 Teknik")
            for x in tech["reasons"]: st.info("• "+x)
        with r:
            st.markdown("### 💰 Temel")
            if freasons:
                for x in freasons:
                    st.success("• "+x) if ("Cazip" in x or "İskontolu" in x) else st.warning("• "+x)
            else:
                st.caption("F/K ve PD/DD belirgin bir avantaj/dezavantaj göstermiyor.")

        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.55,.22,.23],
                          vertical_spacing=.04,subplot_titles=("Fiyat / SMA","RSI","MACD"))
        fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Fiyat"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.SMA50,name="SMA50"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.SMA200,name="SMA200"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.RSI,name="RSI"),row=2,col=1)
        fig.add_hline(y=70,line_dash="dash",row=2,col=1); fig.add_hline(y=30,line_dash="dash",row=2,col=1)
        fig.add_trace(go.Bar(x=df.index,y=df.MACD_Hist,name="MACD Hist"),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.MACD,name="MACD"),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.MACD_Signal,name="Signal"),row=3,col=1)
        fig.update_layout(height=780,template="plotly_dark",paper_bgcolor="#090d14",plot_bgcolor="#101722",xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)

    except Exception as e:
        st.error(f"BIST verisi alınamadı: {e}")

# ============================================================
# 100X RADAR
# ============================================================
else:
    st.title("🔥 100X Radar")
    st.caption("Amaç 100X garantisi değil; düşük piyasa değeri + likidite + seyrelme + teknik yapı kombinasyonlarını sıralamaktır.")

    col1,col2,col3=st.columns(3)
    with col1:
        max_mc = st.number_input("Maksimum Market Cap ($)",value=1_000_000_000,step=50_000_000)
    with col2:
        min_volume = st.number_input("Minimum 24H Hacim ($)",value=100_000,step=50_000)
    with col3:
        max_results = st.number_input("Sonuç Sayısı",value=15,min_value=5,max_value=30,step=5)

    if st.button("🔥 100X TARAMASINI BAŞLAT",use_container_width=True,type="primary"):
        st.session_state["run_radar"]=True

    if st.session_state.get("run_radar",False):
        with st.spinner("Market verileri ve teknik göstergeler taranıyor..."):
            coins = coingecko_markets(4)

        if coins.empty:
            st.error("CoinGecko market verisi alınamadı. İstersen COINGECKO_API_KEY ortam değişkeni ekleyebilirsin.")
        else:
            coins["symbol"] = coins["symbol"].astype(str).str.upper()
            coins = coins[
                (~coins["symbol"].isin(BLOCKED_SYMBOLS)) &
                (coins["market_cap"] <= max_mc) &
                (coins["total_volume"] >= min_volume) &
                (coins["market_cap"] > 0)
            ].copy()

            # Binance ile teknik doğrulama için ilk 50 market adayı.
            coins = coins.sort_values("market_cap",ascending=True).head(60)
            rows=[]
            progress=st.progress(0)
            status=st.empty()

            for idx,(_,row) in enumerate(coins.iterrows(),1):
                sym=row["symbol"]
                status.write(f"🔎 {idx}/{len(coins)} · {row['name']} ({sym})")
                try:
                    raw=binance_klines(sym+"USDT","4h",220)
                    df=add_indicators(raw)
                    tech=technical_score(df)
                    if not tech: continue
                    hx=hundred_x_score(row,tech)
                    if not hx: continue
                    overall=round(hx["score"]*.65 + tech["score"]*.35)
                    rows.append({
                        "Coin":row["name"],
                        "Sembol":sym,
                        "Genel":overall,
                        "100X":hx["score"],
                        "Teknik":tech["score"],
                        "Market Cap":hx["mc"],
                        "FDV":hx["fdv"],
                        "Hacim":hx["volume"],
                        "Hacim/MC":hx["liq"],
                        "100X Hedef":hx["target"],
                        "Risk":hx["risk"],
                        "24H %":hx["change"],
                        "RSI":tech["rsi"],
                        "ADX":tech["adx"],
                        "MACD":"Pozitif" if tech["macd_hist"]>0 else "Negatif"
                    })
                except Exception:
                    pass
                progress.progress(idx/max(1,len(coins)))

            status.empty()
            progress.empty()

            radar_df=pd.DataFrame(rows)
            if radar_df.empty:
                st.warning("Uygun teknik Binance eşleşmesi bulunamadı.")
            else:
                radar_df=radar_df.sort_values(["Genel","100X"],ascending=False).head(int(max_results))

                st.success(f"{len(radar_df)} aday bulundu.")

                for i,r in radar_df.reset_index(drop=True).iterrows():
                    col="green" if r["Genel"]>=75 else "warn" if r["Genel"]>=55 else "bad"
                    st.markdown(f"""
                    <div class="card">
                      <div style="display:flex;justify-content:space-between;gap:15px">
                        <div>
                          <h3 style="margin:0">{i+1}. {r["Coin"]} <span class="pill">{r["Sembol"]}</span></h3>
                          <div class="muted">
                            Market Cap: {money(r["Market Cap"])} · FDV: {money(r["FDV"])} · Hacim: {money(r["Hacim"])}
                          </div>
                          <div style="margin-top:8px">
                            <span class="pill">100X Gerçekçilik: {r["100X"]}/100</span>
                            <span class="pill">Teknik: {r["Teknik"]}/100</span>
                            <span class="pill">Risk: {r["Risk"]}</span>
                          </div>
                          <div class="muted" style="margin-top:7px">
                            100X Hedef MC: {money(r["100X Hedef"])} · Hacim/MC: {r["Hacim/MC"]*100:.2f}% ·
                            24H: {r["24H %"]:.2f}% · RSI: {r["RSI"]:.1f} · ADX: {r["ADX"]:.1f} · MACD: {r["MACD"]}
                          </div>
                        </div>
                        <div style="text-align:right">
                          <div class="score {col}">{r["Genel"]}/100</div>
                          <div class="small">GENEL</div>
                        </div>
                      </div>
                    </div>
                    """,unsafe_allow_html=True)

                st.subheader("📋 Detaylı Tablo")
                display=radar_df.copy()
                for c in ["Market Cap","FDV","Hacim","100X Hedef"]:
                    display[c]=display[c].map(money)
                display["Hacim/MC"]=(display["Hacim/MC"]*100).map(lambda x:f"{x:.2f}%")
                display["24H %"]=display["24H %"].map(lambda x:f"{x:.2f}%")
                display["RSI"]=display["RSI"].map(lambda x:f"{x:.1f}")
                display["ADX"]=display["ADX"].map(lambda x:f"{x:.1f}")
                st.dataframe(display,use_container_width=True,hide_index=True)

                st.warning("⚠️ 100X skoru matematiksel bir filtreleme modelidir; 100X olacağını veya kâr sağlayacağını garanti etmez. Küçük market cap coinlerde kayıp riski çok yüksektir.")
else:
    pass
