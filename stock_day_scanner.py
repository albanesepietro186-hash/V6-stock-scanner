
import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

# ==========================================
# STOCK DAY TRADING SCANNER
# BUY ONLY | US STOCKS | 5-MINUTE SIGNALS
# ==========================================

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN",
    "META", "GOOGL", "TSLA", "AVGO", "NFLX",
    "PLTR", "COIN", "MSTR", "SMCI", "MU",
    "QCOM", "ORCL", "CRM", "UBER", "HOOD",
    "JPM", "BAC", "GS", "XOM", "LLY",
    "WMT", "COST", "SPY", "QQQ", "IWM"
]

INTERVAL = "5m"
PERIOD = "5d"
MIN_AVERAGE_VOLUME = 100_000
MIN_RELATIVE_VOLUME = 1.5
MAX_ALERTS = 10

ET = ZoneInfo("America/New_York")

session = requests.Session()


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError("Missing Telegram secrets.")

    response = session.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": True
        },
        timeout=20
    )
    response.raise_for_status()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    previous_close = df["Close"].shift(1)

    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()


def get_stock_data(symbol):
    try:
        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(column in df.columns for column in required):
            return None

        df = df[required].dropna()

        if len(df) < 60:
            return None

        return df

    except Exception as error:
        print(f"{symbol} data error: {error}")
        return None


def analyze_stock(symbol):
    df = get_stock_data(symbol)

    if df is None:
        return None

    # Exclude the currently forming candle
    df = df.iloc[:-1].copy()

    if len(df) < 50:
        return None

    df["EMA9"] = ema(df["Close"], 9)
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI"] = calculate_rsi(df["Close"])
    df["ATR"] = calculate_atr(df)

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    recent = df.iloc[-21:-1]

    price = float(latest["Close"])
    average_volume = float(recent["Volume"].mean())

    if average_volume <= 0:
        return None

    relative_volume = float(latest["Volume"]) / average_volume

    # Trend confirmation
    bullish_trend = (
        latest["EMA9"] > latest["EMA20"]
        and latest["EMA20"] > latest["EMA50"]
        and price > latest["EMA9"]
    )

    # Momentum confirmation
    momentum = (
        55 <= latest["RSI"] <= 75
        and price > previous["Close"]
    )

    # Breakout confirmation
    breakout_level = float(recent["High"].max())

    breakout = (
        price > breakout_level
        and latest["Volume"] > average_volume * MIN_RELATIVE_VOLUME
    )

    # Liquidity filter
    dollar_volume = price * float(latest["Volume"])

    if average_volume < MIN_AVERAGE_VOLUME:
        return None

    if dollar_volume < 500_000:
        return None

    if not (bullish_trend and momentum and breakout):
        return None

    volatility = float(latest["ATR"])

    if volatility <= 0:
        return None

    risk = volatility * 1.5
    stop_loss = price - risk
    target = price + (risk * 2)

    if stop_loss <= 0:
        return None

    return {
        "symbol": symbol,
        "price": price,
        "stop": stop_loss,
        "target": target,
        "rsi": float(latest["RSI"]),
        "relative_volume": relative_volume,
        "time": str(df.index[-1])
    }


def market_is_open():
    now = datetime.now(ET)

    if now.weekday() >= 5:
        return False

    current_time = now.time()

    return (
        dt_time(9, 35) <= current_time <= dt_time(15, 55)
    )


def main():
    print("Starting Stock Day-Trading Scanner...")

    if not market_is_open():
        print("US regular market is closed. No scan performed.")
        return

    signals = []

    for symbol in WATCHLIST:
        print(f"Scanning {symbol}...")

        signal = analyze_stock(symbol)

        if signal:
            signals.append(signal)
            print(f"BUY setup found: {symbol}")

        time.sleep(0.25)

    signals = signals[:MAX_ALERTS]

    if not signals:
        send_telegram(
            "STOCK DAY-TRADING SCAN COMPLETE\n"
            "No qualifying BUY setups found."
        )
        print("No signals.")
        return

    message = [
        "🚨 STOCK DAY-TRADING BUY ALERTS",
        "5-minute breakout scanner",
        ""
    ]

    for signal in signals:
        message.extend([
            f"🟢 {signal['symbol']}",
            f"Entry reference: ${signal['price']:.2f}",
            f"Stop-loss reference: ${signal['stop']:.2f}",
            f"Target reference: ${signal['target']:.2f}",
            f"RSI: {signal['rsi']:.1f}",
            f"Relative volume: {signal['relative_volume']:.2f}x",
            f"Signal candle: {signal['time']}",
            ""
        ])

    message.append(
        "Signals are not guaranteed fills or returns. "
        "Check spreads, slippage, and risk before trading."
    )

    send_telegram("\n".join(message))

    print(f"Finished. BUY signals: {len(signals)}")


if __name__ == "__main__":
    main()
