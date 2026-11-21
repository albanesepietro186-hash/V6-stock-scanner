
import os
import time
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# =====================================================
# V8 ELITE CRYPTO SWING SCANNER
# Established cryptocurrencies only
# BUY signals only | Coinbase USD spot markets
# =====================================================

BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

BASE = "https://api.exchange.coinbase.com"

APPROVED_COINS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "BNB-USD",
    "LINK-USD",
    "LTC-USD",
    "AVAX-USD",
    "ADA-USD",
    "BCH-USD"
]

MIN_DOLLAR_VOLUME = 5_000_000
MAX_ALERTS = 10

HEADERS = {
    "User-Agent": "V8-Crypto-Swing-Scanner/1.0",
    "Accept": "application/json"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

session = requests.Session()
session.headers.update(HEADERS)


# =====================================================
# API
# =====================================================

def api_get(path, params=None):
    for attempt in range(4):
        try:
            response = session.get(
                BASE + path,
                params=params,
                timeout=25
            )

            if response.status_code == 429:
                time.sleep(3 + attempt * 2)
                continue

            response.raise_for_status()
            return response.json()

        except Exception:
            if attempt == 3:
                raise

            time.sleep(2 + attempt)

    raise RuntimeError("Coinbase API request failed.")


def telegram(message):
    if not BOT or not CHAT:
        raise RuntimeError("Telegram secrets are missing.")

    response = requests.post(
        "https://api.telegram.org/bot" + BOT + "/sendMessage",
        data={
            "chat_id": CHAT,
            "text": message
        },
        timeout=25
    )

    response.raise_for_status()

    if not response.json().get("ok"):
        raise RuntimeError("Telegram rejected message.")


# =====================================================
# FIXED COIN WHITELIST
# =====================================================

def get_universe():
    products = api_get("/products")

    available = {
        p.get("id"): p
        for p in products
    }

    selected = []

    for pair in APPROVED_COINS:
        product = available.get(pair)

        if not product:
            logging.info("Unavailable pair skipped: %s", pair)
            continue

        if (
            product.get("status") != "online"
            or product.get("trading_disabled", False)
            or product.get("cancel_only", False)
            or product.get("limit_only", False)
        ):
            logging.info("Inactive pair skipped: %s", pair)
            continue

        try:
            stats = api_get(
                "/products/" + pair + "/stats"
            )

            price = float(stats.get("last", 0))
            volume = float(stats.get("volume", 0))

            dollar_volume = price * volume

            if (
                price <= 0
                or dollar_volume < MIN_DOLLAR_VOLUME
            ):
                logging.info("Insufficient liquidity: %s", pair)
                continue

            selected.append(pair)

            logging.info(
                "Verified market: %s | 24h dollar volume: $%.0f",
                pair,
                dollar_volume
            )

        except Exception:
            logging.exception("Market verification failed: %s", pair)

        time.sleep(0.2)

    if "BTC-USD" not in selected:
        raise RuntimeError(
            "BTC-USD unavailable or failed liquidity verification. "
            "Scanner aborted."
        )

    if len(selected) < 2:
        raise RuntimeError(
            "Too few approved markets passed verification."
        )

    return selected


# =====================================================
# HISTORICAL DAILY CANDLES
# =====================================================

def get_history(pair):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=290)

    candles = api_get(
        "/products/" + pair + "/candles",
        params={
            "granularity": 86400,
            "start": start.isoformat(),
            "end": end.isoformat()
        }
    )

    if not isinstance(candles, list) or not candles:
        raise RuntimeError("No candle data: " + pair)

    data = pd.DataFrame(
        candles,
        columns=[
            "Time",
            "Low",
            "High",
            "Open",
            "Close",
            "Volume"
        ]
    )

    data["Time"] = pd.to_datetime(
        data["Time"],
        unit="s",
        utc=True
    )

    data = data.set_index("Time").sort_index()

    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna()

    # Exclude today's incomplete candle.
    today = pd.Timestamp.now(
        tz="UTC"
    ).normalize()

    data = data[data.index < today]

    data = data[
        ~data.index.duplicated(keep="last")
    ]

    if len(data) < 220:
        raise RuntimeError(
            "Insufficient history: " + pair
        )

    if not np.isfinite(
        data[columns].to_numpy()
    ).all():
        raise RuntimeError(
            "Invalid candle values: " + pair
        )

    if (
        (data["Close"] <= 0).any()
        or (data["Volume"] < 0).any()
        or (data["High"] < data["Low"]).any()
    ):
        raise RuntimeError(
            "Invalid prices or volume: " + pair
        )

    return data


# =====================================================
# TECHNICAL INDICATORS
# =====================================================

def indicators(data):
    x = data.copy()

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    x["EMA20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    x["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    x["SMA50"] = close.rolling(50).mean()
    x["SMA200"] = close.rolling(200).mean()

    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    x["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    x["HIST"] = macd - signal

    # Average True Range
    previous = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous).abs(),
            (low - previous).abs()
        ],
        axis=1
    ).max(axis=1)

    x["ATR"] = tr.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    # ADX and directional indicators
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0
        ),
        index=x.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0
        ),
        index=x.index
    )

    smoothed_tr = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    plus_di = (
        100 * plus_dm.ewm(
            alpha=1/14,
            adjust=False
        ).mean()
        / smoothed_tr.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_dm.ewm(
            alpha=1/14,
            adjust=False
        ).mean()
        / smoothed_tr.replace(0, np.nan)
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    x["ADX"] = dx.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    x["PDI"] = plus_di
    x["MDI"] = minus_di

    # Volume and relative strength
    dollar_volume = close * volume

    x["AVG_DOLLAR_VOLUME"] = (
        dollar_volume.rolling(20).mean()
    )

    x["REL_VOLUME"] = (
        dollar_volume
        / x["AVG_DOLLAR_VOLUME"].replace(0, np.nan)
    )

    x["HIGH20"] = high.shift(1).rolling(20).max()
    x["RET20"] = close.pct_change(20)

    return x


# =====================================================
# WEEKLY TREND CONFIRMATION
# =====================================================

def weekly_confirmation(data):
    weekly = data.resample("W-SUN").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    today = pd.Timestamp.now(
        tz="UTC"
    ).date()

    # Only completed weekly candles.
    weekly = weekly[
        weekly.index.date < today
    ]

    if len(weekly) < 30:
        return False

    ema10 = weekly["Close"].ewm(
        span=10,
        adjust=False
    ).mean()

    return bool(
        weekly["Close"].iloc[-1] > ema10.iloc[-1]
        and ema10.iloc[-1] > ema10.iloc[-2]
    )


# =====================================================
# BITCOIN MARKET FILTER
# =====================================================

def btc_market_ok(btc):
    x = indicators(btc).dropna()

    if len(x) < 2:
        return False

    latest = x.iloc[-1]

    return bool(
        latest["Close"] > latest["SMA50"]
        and latest["SMA50"] > latest["SMA200"]
        and weekly_confirmation(btc)
    )


# =====================================================
# BUY SIGNAL ENGINE
# =====================================================

def analyze(pair, data, btc):
    x = indicators(data).dropna()
    benchmark = indicators(btc).dropna()

    if len(x) < 3 or len(benchmark) < 3:
        return None

    if not weekly_confirmation(data):
        return None

    latest = x.iloc[-1]
    previous = x.iloc[-2]
    market = benchmark.iloc[-1]

    # Strong daily trend
    trend = (
        latest["Close"] > latest["EMA20"]
        and latest["EMA20"] > latest["EMA50"]
        and latest["EMA50"] > latest["SMA200"]
        and latest["SMA50"] > latest["SMA200"]
    )

    # Momentum confirmation
    momentum = (
        52 <= latest["RSI"] <= 68
        and latest["HIST"] > 0
        and latest["HIST"] >= previous["HIST"]
        and latest["ADX"] >= 18
        and latest["PDI"] > latest["MDI"]
    )

    volume_ok = latest["REL_VOLUME"] >= 1.2

    # Must outperform Bitcoin over 20 days.
    relative_strength = (
        latest["RET20"] > market["RET20"]
    )

    # Breakout or pullback reclaim
    breakout = (
        latest["Close"] > latest["HIGH20"]
        and latest["Close"] > latest["Open"]
    )

    pullback = (
        latest["Low"] <= latest["EMA20"] * 1.015
        and latest["Close"] > latest["EMA20"]
        and latest["Close"] > latest["Open"]
        and latest["Close"] > previous["Close"]
    )

    if breakout:
        setup = "20-day breakout"
    elif pullback:
        setup = "EMA20 pullback reclaim"
    else:
        return None

    if not all([
        trend,
        momentum,
        volume_ok,
        relative_strength
    ]):
        return None

    price = float(latest["Close"])
    atr = float(latest["ATR"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    stop = price - 2 * atr
    risk = price - stop
    target = price + 2.5 * risk

    relative = float(
        100 * (latest["RET20"] - market["RET20"])
    )

    score = (
        (20 if latest["Close"] > latest["SMA200"] else 0)
        + (15 if latest["EMA20"] > latest["EMA50"] else 0)
        + (15 if latest["RSI"] >= 55 else 10)
        + (15 if latest["ADX"] >= 25 else 8)
        + (15 if latest["REL_VOLUME"] >= 1.5 else 8)
        + (20 if relative >= 5 else 10)
    )

    return {
        "pair": pair,
        "setup": setup,
        "price": price,
        "stop": stop,
        "target": target,
        "rsi": float(latest["RSI"]),
        "adx": float(latest["ADX"]),
        "volume": float(latest["REL_VOLUME"]),
        "relative": relative,
        "score": int(score),
        "date": x.index[-1].strftime("%Y-%m-%d")
    }


# =====================================================
# TELEGRAM ALERT
# =====================================================

def format_alert(signal):
    return (
        "🟢 V8 CRYPTO BUY WATCHLIST\n\n"
        f"Coinbase pair: {signal['pair']}\n"
        f"Setup: {signal['setup']}\n"
        f"Signal date: {signal['date']} UTC\n"
        f"Quality score: {signal['score']}/100\n\n"
        f"Reference close: ${signal['price']:,.6f}\n"
        f"ATR stop reference: ${signal['stop']:,.6f}\n"
        f"Target reference (2.5R): ${signal['target']:,.6f}\n\n"
        f"RSI: {signal['rsi']:.1f}\n"
        f"ADX: {signal['adx']:.1f}\n"
        f"Relative volume: {signal['volume']:.2f}x\n"
        f"20-day strength vs BTC: {signal['relative']:+.2f}%\n\n"
        "Daily and weekly trend confirmation.\n"
        "Spot markets only. Reference levels are not live "
        "execution prices. Check live price, spread and fees "
        "before trading. Not financial advice."
    )


# =====================================================
# MAIN SCANNER
# =====================================================

def main():
    if not BOT or not CHAT:
        raise RuntimeError("Telegram secrets are missing.")

    logging.info("V8 Crypto Scanner starting.")

    pairs = get_universe()

    data = {}
    failed = []

    for pair in pairs:
        try:
            data[pair] = get_history(pair)
            logging.info("Valid candle history: %s", pair)

        except Exception:
            failed.append(pair)
            logging.exception("History failed: %s", pair)

        time.sleep(0.2)

    if "BTC-USD" not in data:
        raise RuntimeError(
            "BTC history unavailable. Scan aborted."
        )

    if not btc_market_ok(data["BTC-USD"]):
        telegram(
            "V8 CRYPTO SCANNER\n\n"
            "No BUY alerts today.\n"
            "Bitcoin failed the bullish daily and weekly "
            "market trend filter."
        )
        return

    signals = []

    for pair in pairs:
        if pair not in data:
            continue

        try:
            result = analyze(
                pair,
                data[pair],
                data["BTC-USD"]
            )

            if result:
                signals.append(result)

        except Exception:
            logging.exception("Analysis failed: %s", pair)

    signals.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    if not signals:
        telegram(
            "V8 CRYPTO SCANNER\n\n"
            "No high-conviction BUY setups today.\n"
            "Bitcoin market filter passed, but none of the "
            "approved coins met every entry condition."
        )
        return

    for signal in signals[:MAX_ALERTS]:
        telegram(format_alert(signal))
        time.sleep(1)

    logging.info(
        "Completed. Alerts sent: %d | Data failures: %d",
        min(len(signals), MAX_ALERTS),
        len(failed)
    )


if __name__ == "__main__":
    main()
