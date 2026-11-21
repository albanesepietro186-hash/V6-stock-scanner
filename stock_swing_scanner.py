
import os
import time
import requests
import pandas as pd
import yfinance as yf

# ==========================================
# STOCK SWING TRADING SCANNER
# BUY ONLY | US STOCKS | DAILY + WEEKLY
# ==========================================

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN",
    "META", "GOOGL", "TSLA", "AVGO", "NFLX",
    "PLTR", "COIN", "MSTR", "SMCI", "MU",
    "QCOM", "ORCL", "CRM", "UBER", "HOOD",
    "JPM", "BAC", "GS", "XOM", "LLY",
    "WMT", "COST", "SPY", "QQQ", "IWM"
]

PERIOD = "1y"
MAX_ALERTS = 10
MIN_AVERAGE_VOLUME = 500_000
MIN_RELATIVE_VOLUME = 1.2

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


def calculate_ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


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


def get_stock_data(symbol, interval):
    try:
        df = yf.download(
            symbol,
            period=PERIOD,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
            multi_level_index=False,
            timeout=20
        )

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = [
            "Open", "High", "Low", "Close", "Volume"
        ]

        if not all(column in df.columns for column in required):
            return None

        return df[required].dropna()

    except Exception as error:
        print(f"{symbol} data error: {error}")
        return None


def analyze_stock(symbol):
    daily = get_stock_data(symbol, "1d")
    weekly = get_stock_data(symbol, "1wk")

    if daily is None or weekly is None:
        return None

    if len(daily) < 100 or len(weekly) < 25:
        return None

    # Daily indicators
    daily["EMA20"] = calculate_ema(daily["Close"], 20)
    daily["EMA50"] = calculate_ema(daily["Close"], 50)
    daily["EMA200"] = calculate_ema(daily["Close"], 200)
    daily["RSI"] = calculate_rsi(daily["Close"])
    daily["ATR"] = calculate_atr(daily)

    # Weekly indicators
    weekly["EMA10"] = calculate_ema(weekly["Close"], 10)
    weekly["EMA20"] = calculate_ema(weekly["Close"], 20)

    latest = daily.iloc[-1]
    previous = daily.iloc[-2]
    weekly_latest = weekly.iloc[-1]

    price = float(latest["Close"])

    # Avoid incomplete current daily candle
    # when the US market is still open.
    # The scheduled workflow should run after market close.

    # Weekly trend confirmation
    weekly_bullish = (
        weekly_latest["Close"] > weekly_latest["EMA10"]
        and weekly_latest["EMA10"] > weekly_latest["EMA20"]
    )

    # Daily trend confirmation
    daily_bullish = (
        latest["Close"] > latest["EMA20"]
        and latest["EMA20"] > latest["EMA50"]
        and latest["EMA50"] > latest["EMA200"]
    )

    # Momentum confirmation
    momentum = (
        50 <= latest["RSI"] <= 70
        and latest["Close"] > previous["Close"]
    )

    recent = daily.iloc[-21:-1]

    if len(recent) < 20:
        return None

    average_volume = float(recent["Volume"].mean())

    if average_volume <= 0:
        return None

    relative_volume = (
        float(latest["Volume"]) / average_volume
    )

    if average_volume < MIN_AVERAGE_VOLUME:
        return None

    if relative_volume < MIN_RELATIVE_VOLUME:
        return None

    # Breakout or controlled pullback
    resistance = float(recent["High"].max())

    breakout = (
        price > resistance
        and relative_volume >= 1.5
    )

    pullback = (
        float(latest["Low"]) <= float(latest["EMA20"]) * 1.02
        and price > float(latest["EMA20"])
        and price > float(latest["Open"])
    )

    if not (
        weekly_bullish
        and daily_bullish
        and momentum
        and (breakout or pullback)
    ):
        return None

    volatility = float(latest["ATR"])

    if volatility <= 0:
        return None

    risk = volatility * 2

    stop = price - risk
    target = price + (risk * 2.5)

    if stop <= 0:
        return None

    return {
        "symbol": symbol,
        "price": price,
        "stop": stop,
        "target": target,
        "rsi": float(latest["RSI"]),
        "relative_volume": relative_volume,
        "setup": "BREAKOUT" if breakout else "PULLBACK"
    }


def main():
    print("Starting Stock Swing Trading Scanner...")

    signals = []
    errors = []

    for symbol in WATCHLIST:
        print(f"Scanning {symbol}...")

        try:
            signal = analyze_stock(symbol)

            if signal:
                signals.append(signal)
                print(f"BUY setup found: {symbol}")

        except Exception as error:
            errors.append(f"{symbol}: {error}")
            print(f"Error scanning {symbol}: {error}")

        time.sleep(0.3)

    signals = signals[:MAX_ALERTS]

    if signals:
        message = [
            "🚨 STOCK SWING BUY ALERTS",
            "Daily + weekly trend confirmation",
            ""
        ]

        for signal in signals:
            message.extend([
                f"🟢 {signal['symbol']}",
                f"Setup: {signal['setup']}",
                f"Entry reference: ${signal['price']:.2f}",
                f"Stop-loss reference: ${signal['stop']:.2f}",
                f"Target reference: ${signal['target']:.2f}",
                f"RSI: {signal['rsi']:.1f}",
                f"Relative volume: {signal['relative_volume']:.2f}x",
                ""
            ])

        message.append(
            "Signals are not guaranteed fills or returns. "
            "Check liquidity, fees, slippage, and risk before trading."
        )

        send_telegram("\n".join(message))

    else:
        send_telegram(
            "STOCK SWING SCAN COMPLETE\n"
            "No qualifying BUY setups found."
        )

    if errors:
        print("Warnings:")
        for error in errors:
            print(error)

    print(f"Scan complete. BUY signals: {len(signals)}")


if __name__ == "__main__":
    main()
