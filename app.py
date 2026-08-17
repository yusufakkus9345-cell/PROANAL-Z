import streamlit as st

import pandas as pd

import numpy as np

import requests

import yfinance as yf

import plotly.graph_objects as go

from plotly.subplots import make_subplots

st.set_page_config(

    page_title="ProAnaliz",

    page_icon="📈",

    layout="wide"

)

# =========================

# TASARIM

# =========================

st.markdown("""

<style>

.stApp {

    background-color: #0b0f17;

    color: #e8edf5;

}

.card {

    background: linear-gradient(135deg,#131a26,#1b2433);

    border: 1px solid #2b374b;

    border-radius: 14px;

    padding: 18px;

}

</style>

""", unsafe_allow_html=True)

# =========================

# YARDIMCI FONKSİYONLAR

# =========================

def num(x, default=np.nan):

    try:

        if x is None:

            return default

        return float(x)

    except Exception:

        return default

def money(x):

    if x is None or pd.isna(x):

        return "Veri yok"

    x = float(x)

    if x >= 1_000_000_000:

        return f"${x / 1_000_000_000:.2f}B"

    if x >= 1_000_000:

        return f"${x / 1_000_000:.2f}M"

    if x >= 1_000:

        return f"${x / 1_000:.2f}K"

    return f"${x:.2f}"

def ema(series, period):

    return series.ewm(

        span=period,

        adjust=False

    ).mean()

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(

        alpha=1 / period,

        adjust=False

    ).mean()

    avg_loss = loss.ewm(

        alpha=1 / period,

        adjust=False

    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return (

        100 - (100 / (1 + rs))

    ).fillna(50)

def calculate_macd(series):

    fast = ema(series, 12)

    slow = ema(series, 26)

    macd_line = fast - slow

    signal = ema(macd_line, 9)

    return (

        macd_line,

        signal,

        macd_line - signal

    )

def calculate_adx(df, period=14):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    up = high.diff()

    down = -low.diff()

    plus_dm = up.where(

        (up > down) & (up > 0),

        0

    )

    minus_dm = down.where(

        (down > up) & (down > 0),

        0

    )

    tr1 = high - low

    tr2 = (high - close.shift()).abs()

    tr3 = (low - close.shift()).abs()

    tr = pd.concat(

        [tr1, tr2, tr3],

        axis=1

    ).max(axis=1)

    atr = tr.ewm(

        alpha=1 / period,

        adjust=False

    ).mean()

    plus_di = (

        100 *

        plus_dm.ewm(

            alpha=1 / period,

            adjust=False

        ).mean()

        / atr.replace(0, np.nan)

    )

    minus_di = (

        100 *

        minus_dm.ewm(

            alpha=1 / period,

            adjust=False

        ).mean()

        / atr.replace(0, np.nan)

    )

    dx = (

        (plus_di - minus_di).abs()

        /

        (plus_di + minus_di).replace(

            0,

            np.nan

        )

    ) * 100

    return dx.ewm(

        alpha=1 / period,

        adjust=False

    ).mean().fillna(0)

# =========================

# BINANCE

# =========================

@st.cache_data(ttl=120)

def get_binance(symbol):

    urls = [

        "https://api.binance.com/api/v3/klines",

        "https://api.binance.us/api/v3/klines"

    ]

    params = {

        "symbol": symbol.upper() + "USDT",

        "interval": "1d",

        "limit": 365

    }

    for url in urls:

        try:

            r = requests.get(

                url,

                params=params,

                timeout=10

            )

            if r.status_code != 200:

                continue

            data = r.json()

            if not isinstance(data, list):

                continue

            if len(data) < 100:

                continue

            columns = [

                "time",

                "Open",

                "High",

                "Low",

                "Close",

                "Volume",

                "close_time",

                "quote_volume",

                "trades",

                "buy_base",

                "buy_quote",

                "ignore"

            ]

            df = pd.DataFrame(

                data,

                columns=columns

            )

            for col in [

                "Open",

                "High",

                "Low",

                "Close",

                "Volume"

            ]:

                df[col] = pd.to_numeric(

                    df[col],

                    errors="coerce"

                )

            df.index = pd.to_datetime(

                df["time"],

                unit="ms"

            )

            return df[

                [

                    "Open",

                    "High",

                    "Low",

                    "Close",

                    "Volume"

                ]

            ]

        except Exception:

            pass

    return None

# =========================

# COINGECKO

# =========================

@st.cache_data(ttl=300)

def get_market_data():

    try:

        url = (

            "https://api.coingecko.com/api/v3/"

            "coins/markets"

        )

        params = {

            "vs_currency": "usd",

            "order": "market_cap_desc",

            "per_page": 250,

            "page": 1,

            "sparkline": "false"

        }

        r = requests.get(

            url,

            params=params,

            timeout=15

        )

        if r.status_code != 200:

            return {}

        data = r.json()

        result = {}

        for coin in data:

            symbol = str(

                coin.get("symbol", "")

            ).upper()

            result[symbol] = {

                "name": coin.get(

                    "name",

                    symbol

                ),

                "market_cap": num(

                    coin.get("market_cap")

                ),

                "fdv": num(

                    coin.get(

                        "fully_diluted_valuation"

                    )

                ),

                "volume": num(

                    coin.get("total_volume")

                ),

                "change": num(

                    coin.get(

                        "price_change_percentage_24h"

                    )

                ),

                "ath_change": num(

                    coin.get(

                        "ath_change_percentage"

                    )

                )

            }

        return result

    except Exception:

        return {}

# =========================

# BIST

# =========================

@st.cache_data(ttl=300)

def get_bist(symbol):

    try:

        ticker = yf.Ticker(

            symbol + ".IS"

        )

        df = ticker.history(

            period="1y",

            interval="1d",

            auto_adjust=False

        )

        if df is None or df.empty:

            return None

        required = [

            "Open",

            "High",

            "Low",

            "Close",

            "Volume"

        ]

        df = df[required].copy()

        df.dropna(

            subset=["Close"],

            inplace=True

        )

        if len(df) < 100:

            return None

        return df

    except Exception:

        return None

# =========================

# İNDİKATÖRLER

# =========================

def indicators(df):

    df = df.copy()

    df["EMA20"] = ema(

        df["Close"],

        20

    )

    df["EMA50"] = ema(

        df["Close"],

        50

    )

    df["SMA50"] = (

        df["Close"]

        .rolling(50)

        .mean()

    )

    df["SMA200"] = (

        df["Close"]

        .rolling(200)

        .mean()

    )

    df["RSI"] = calculate_rsi(

        df["Close"]

    )

    (

        df["MACD"],

        df["MACD_SIGNAL"],

        df["MACD_HIST"]

    ) = calculate_macd(

        df["Close"]

    )

    df["ADX"] = calculate_adx(df)

    df["VOL_MA20"] = (

        df["Volume"]

        .rolling(20)

        .mean()

    )

    return df

# =========================

# TEKNİK SKOR

# =========================

def technical_analysis(df):

    last = df.iloc[-1]

    prev = df.iloc[-2]

    score = 0

    reasons = []

    rsi_value = num(

        last["RSI"],

        50

    )

    if 45 <= rsi_value <= 65:

        score += 18

        reasons.append(

            "RSI sağlıklı bölgede"

        )

    elif 30 <= rsi_value < 45:

        score += 15

        reasons.append(

            "RSI düşük bölgede"

        )

    elif rsi_value < 30:

        score += 10

        reasons.append(

            "RSI aşırı satım bölgesinde"

        )

    elif rsi_value <= 70:

        score += 12

    else:

        score += 5

        reasons.append(

            "RSI yüksek / aşırı alım riski"

        )

    if last["Close"] > last["EMA20"]:

        score += 15

        reasons.append(

            "Fiyat EMA20 üzerinde"

        )

    if last["EMA20"] > last["EMA50"]:

        score += 15

        reasons.append(

            "EMA20 > EMA50"

        )

    if (

        not pd.isna(last["SMA50"])

        and last["Close"] > last["SMA50"]

    ):

        score += 10

        reasons.append(

            "Fiyat SMA50 üzerinde"

        )

    if (

        not pd.isna(last["SMA200"])

        and last["Close"] > last["SMA200"]

    ):

        score += 10

        reasons.append(

            "Fiyat SMA200 üzerinde"

        )

    if last["MACD"] > last["MACD_SIGNAL"]:

        score += 15

        reasons.append(

            "MACD pozitif"

        )

    if (

        prev["MACD"] <= prev["MACD_SIGNAL"]

        and

        last["MACD"] > last["MACD_SIGNAL"]

    ):

        score += 5

        reasons.append(

            "MACD yukarı kesişim"

        )

    adx_value = num(

        last["ADX"],

        0

    )

    if adx_value >= 25:

        score += 10

        reasons.append(

            f"ADX güçlü trend ({adx_value:.1f})"

        )

    return min(score, 100), reasons

# =========================

# 100X

# =========================

def analyze_100x(

    market_cap,

    volume,

    ath_change

):

    if (

        market_cap is None

        or pd.isna(market_cap)

        or market_cap <= 0

    ):

        return None, None, "VERİ YOK"

    score = 50

    if market_cap < 100_000_000:

        score += 20

    elif market_cap < 500_000_000:

        score += 10

    elif market_cap < 1_000_000_000:

        score += 5

    else:

        score -= 10

    if (

        volume is not None

        and not pd.isna(volume)

        and volume > 0

    ):

        ratio = volume / market_cap

        if ratio >= 0.10:

            score += 15

        elif ratio >= 0.03:

            score += 10

        elif ratio >= 0.01:

            score += 5

    if (

        ath_change is not None

        and not pd.isna(ath_change)

    ):

        if ath_change <= -80:

            score += 10

        elif ath_change <= -50:

            score += 5

    score = max(

        0,

        min(

            100,

            int(score)

        )

    )

    target = market_cap * 100

    if score >= 70:

        risk = "YÜKSEK"

    elif score >= 50:

        risk = "ÇOK YÜKSEK"

    else:

        risk = "AŞIRI YÜKSEK"

    return score, target, risk

# =========================

# LİSTELER

# =========================

CRYPTO = [

    "BTC",

    "ETH",

    "SOL",

    "XRP",

    "BNB",

    "DOGE",

    "ADA",

    "AVAX",

    "DOT",

    "LINK",

    "NEAR",

    "APT",

    "ARB",

    "OP",

    "SUI",

    "ATOM",

    "FIL",

    "INJ",

    "SEI",

    "TIA",

    "TRX",

    "LTC",

    "BCH",

    "ETC",

    "UNI",

    "AAVE",

    "MKR",

    "LDO",

    "AR",

    "CFX",

    "ZEC"

]

BIST = [

    "THYAO",

    "GARAN",

    "AKBNK",

    "ASELS",

    "TUPRS",

    "EREGL",

    "KCHOL",

    "BIMAS",

    "SISE",

    "SASA"

]

# =========================

# SIDEBAR

# =========================

st.sidebar.title(

    "📈 ProAnaliz"

)

market = st.sidebar.radio(

    "Piyasa",

    ["Kripto", "BIST"]

)

if market == "Kripto":

    symbol = st.sidebar.selectbox(

        "Coin",

        CRYPTO

    )

else:

    symbol = st.sidebar.selectbox(

        "Hisse",

        BIST

    )

if st.sidebar.button(

    "🔄 Veriyi Yenile"

):

    st.cache_data.clear()

    st.rerun()

# =========================

# VERİ

# =========================

with st.spinner(

    "Piyasa verileri alınıyor..."

):

    if market == "Kripto":

        df = get_binance(symbol)

        markets = get_market_data()

        market_info = markets.get(

            symbol,

            {}

        )

    else:

        df = get_bist(symbol)

        market_info = {}

# =========================

# VERİ KONTROL

# =========================

if df is None or df.empty:

    st.error(

        f"❌ {symbol} için veri alınamadı."

    )

    st.info(

        "Bir süre sonra 'Veriyi Yenile' "

        "butonunu deneyin."

    )

    st.stop()

# =========================

# ANALİZ

# =========================

df = indicators(df)

last = df.iloc[-1]

price = num(

    last["Close"]

)

score, reasons = technical_analysis(

    df

)

rsi_value = num(

    last["RSI"],

    50

)

adx_value = num(

    last["ADX"],

    0

)

# =========================

# SİNYAL

# =========================

if score >= 80:

    signal = "🟢 GÜÇLÜ AL"

elif score >= 65:

    signal = "🟩 AL"

elif score >= 50:

    signal = "🟡 NÖTR / TUT"

elif score >= 35:

    signal = "🟧 SAT"

else:

    signal = "🔴 GÜÇLÜ SAT"

# =========================

# BAŞLIK

# =========================

st.title(

    f"{symbol} • {market}"

)

st.caption(

    "Teknik analiz ve 100X radar terminali"

)

# =========================

# ANA METRİKLER

# =========================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(

        "Fiyat",

        f"{price:.8g}"

    )

with c2:

    st.metric(

        "Teknik Skor",

        f"{score}/100"

    )

with c3:

    st.metric(

        "RSI",

        f"{rsi_value:.1f}"

    )

with c4:

    st.metric(

        "Sinyal",

        signal

    )

# =========================

# KRİPTO VERİLERİ

# =========================

if market == "Kripto":

    market_cap = market_info.get(

        "market_cap"

    )

    fdv = market_info.get(

        "fdv"

    )

    volume = market_info.get(

        "volume"

    )

    change = market_info.get(

        "change"

    )

    ath_change = market_info.get(

        "ath_change"

    )

    x_score, x_target, x_risk = analyze_100x(

        market_cap,

        volume,

        ath_change

    )

    st.divider()

    st.subheader(

        "💰 Market & 100X"

    )

    a, b, c, d, e = st.columns(5)

    with a:

        st.metric(

            "Market Cap",

            money(market_cap)

        )

    with b:

        st.metric(

            "FDV",

            money(fdv)

        )

    with c:

        st.metric(

            "24H Hacim",

            money(volume)

        )

    with d:

        if x_score is None:

            st.metric(

                "100X Gerçekçilik",

                "Veri yok"

            )

        else:

            st.metric(

                "100X Gerçekçilik",

                f"{x_score}/100"

            )

    with e:

        if x_target is None:

            st.metric(

                "100X Hedef MC",

                "Veri yok"

            )

        else:

            st.metric(

                "100X Hedef MC",

                money(x_target)

            )

    if x_score is not None:

        st.write(

            f"**100X Risk:** {x_risk}"

        )

        st.progress(

            x_score / 100

        )

    if change is not None:

        st.write(

            f"24H Değişim: **{change:.2f}%**"

        )

# =========================

# TEKNİK GÖSTERGELER

# =========================

st.divider()

st.subheader(

    "📊 Teknik Göstergeler"

)

a, b, c, d, e = st.columns(5)

with a:

    st.metric(

        "RSI",

        f"{rsi_value:.1f}"

    )

with b:

    st.metric(

        "ADX",

        f"{adx_value:.1f}"

    )

with c:

    macd_text = (

        "POZİTİF"

        if last["MACD"] > last["MACD_SIGNAL"]

        else "NEGATİF"

    )

    st.metric(

        "MACD",

        macd_text

    )

with d:

    ema_text = (

        "ÜSTÜ"

        if price > last["EMA20"]

        else "ALTI"

    )

    st.metric(

        "EMA20",

        ema_text

    )

with e:

    if last["VOL_MA20"] > 0:

        volume_ratio = (

            last["Volume"]

            /

            last["VOL_MA20"]

        )

        st.metric(

            "Hacim / Ort.",

            f"{volume_ratio:.2f}x"

        )

    else:

        st.metric(

            "Hacim / Ort.",

            "Veri yok"

        )

# =========================

# SİNYALLER

# =========================

st.divider()

left, right = st.columns(2)

with left:

    st.subheader(

        "🧠 Teknik Sinyaller"

    )

    for reason in reasons:

        st.success(

            "✓ " + reason

        )

with right:

    st.subheader(

        "⚠️ Risk Değerlendirmesi"

    )

    if market == "Kripto":

        if x_score is None:

            st.info(

                "Market Cap verisi bulunamadığı "

                "için 100X hesabı yapılamadı."

            )

        elif x_score >= 70:

            st.warning(

                "100X potansiyeli matematiksel "

                "olarak daha ulaşılabilir görünse "

                "de risk seviyesi çok yüksektir."

            )

        elif x_score >= 50:

            st.warning(

                "100X senaryosu mümkündür ancak "

                "gerçekleşmesi zordur."

            )

        else:

            st.error(

                "100X senaryosu aşırı yüksek risklidir."

            )

    else:

        st.info(

            "BIST için teknik analiz kullanılıyor."

        )

# =========================

# GRAFİK

# =========================

st.divider()

st.subheader(

    "📈 Teknik Grafik"

)

fig = make_subplots(

    rows=3,

    cols=1,

    shared_xaxes=True,

    vertical_spacing=0.04,

    row_heights=[

        0.55,

        0.22,

        0.23

    ],

    subplot_titles=[

        "Fiyat / EMA / SMA",

        "RSI",

        "MACD"

    ]

)

# FİYAT

fig.add_trace(

    go.Candlestick(

        x=df.index,

        open=df["Open"],

        high=df["High"],

        low=df["Low"],

        close=df["Close"],

        name="Fiyat"

    ),

    row=1,

    col=1

)

# EMA20

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["EMA20"],

        name="EMA20"

    ),

    row=1,

    col=1

)

# EMA50

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["EMA50"],

        name="EMA50"

    ),

    row=1,

    col=1

)

# SMA200

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["SMA200"],

        name="SMA200"

    ),

    row=1,

    col=1

)

# RSI

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["RSI"],

        name="RSI"

    ),

    row=2,

    col=1

)

fig.add_hline(

    y=70,

    line_dash="dash",

    row=2,

    col=1

)

fig.add_hline(

    y=30,

    line_dash="dash",

    row=2,

    col=1

)

# MACD

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["MACD"],

        name="MACD"

    ),

    row=3,

    col=1

)

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["MACD_SIGNAL"],

        name="Signal"

    ),

    row=3,

    col=1

)

fig.update_layout(

    height=800,

    template="plotly_dark",

    xaxis_rangeslider_visible=False,

    margin=dict(

        l=20,

        r=20,

        t=50,

        b=20

    )

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# =========================

# ALT BİLGİ

# =========================

st.divider()

st.caption(

    "ProAnaliz otomatik hesaplanan piyasa "

    "verilerini kullanır. Skorlar yatırım "

    "tavsiyesi değildir."

)
