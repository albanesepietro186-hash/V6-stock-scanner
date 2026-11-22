
import os
import time
import requests
from datetime import datetime, timezone

# ==========================================
# V10 CRYPTO DAY TRADING SCANNER
# BUY + SELL SIGNALS
# ==========================================

API = "https://api.exchange.coinbase.com"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

COINS = [
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

HEADERS = {
    "User-Agent": "V10-Crypto-Scanner"
}

# ==========================================
# DATA
# ==========================================

def get_candles(symbol, granularity, limit=100):
    try:
        response = requests.get(
            f"{API}/products/{symbol}/candles",
            params={
                "granularity": granularity,
                "limit": limit
            },
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()
        candles = response.json()

        candles.sort(key=lambda x: x[0])

        # Exclude the currently forming candle
        candles = candles[:-1]

        return [
            {
                "time": c[0],
                "low": float(c[1]),
                "high": float(c[2]),
                "open": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5])
            }
            for c in candles
        ]

    except Exception as e:
        print(f"Data error {symbol}: {e}")
        return []


# ==========================================
# INDICATORS
# ==========================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    result = sum(values[:period]) / period

    for value in values[period:]:
        result = (value * multiplier) + (
            result * (1 - multiplier)
        )

    return result


def rsi(values, period=14):
    if len(values) <= period:
        return None

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            avg_gain * (period - 1) + gains[i]
        ) / period

        avg_loss = (
            avg_loss * (period - 1) + losses[i]
        ) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def atr(candles, period=14):
    if len(candles) <= period:
        return None

    ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        true_range = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        )

        ranges.append(true_range)

    return sum(ranges[-period:]) / period


def relative_volume(candles, period=20):
    if len(candles) < period + 1:
        return None

    previous_volumes = [
        c["volume"] for c in candles[-period-1:-1]
    ]

    average = sum(previous_volumes) / period

    if average <= 0:
        return None

    return candles[-1]["volume"] / average


# ==========================================
# TELEGRAM
# ==========================================

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets missing.")
        return

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=15
        )

        response.raise_for_status()

        print("Telegram notification sent.")

    except Exception as e:
        print(f"Telegram error: {e}")


# ==========================================
# BTC MARKET FILTER
# ==========================================

def btc_trend():
    candles = get_candles("BTC-USD", 900)

    if len(candles) < 30:
        return None

    closes = [c["close"] for c in candles]

    fast = ema(closes, 9)
    slow = ema(closes, 20)

    if fast is None or slow is None:
        return None

    if fast > slow:
        return "BULLISH"

    if fast < slow:
        return "BEARISH"

    return "NEUTRAL"


# ==========================================
# SCANNER
# ==========================================

def scan_coin(symbol, market_trend):

    candles_1m = get_candles(symbol, 60)
    candles_5m = get_candles(symbol, 300)
    candles_15m = get_candles(symbol, 900)

    if (
        len(candles_1m) < 30
        or len(candles_5m) < 30
        or len(candles_15m) < 30
    ):
        return None

    closes_1m = [c["close"] for c in candles_1m]
    closes_5m = [c["close"] for c in candles_5m]

    fast_5m = ema(closes_5m, 9)
    slow_5m = ema(closes_5m, 20)

    rsi_5m = rsi(closes_5m)

    volume = relative_volume(candles_5m)

    volatility = atr(candles_5m)

    if None in (fast_5m, slow_5m, rsi_5m, volume, volatility):
        return None

    current = candles_1m[-1]
    previous = candles_1m[-2]

    entry = current["close"]

    if entry <= 0 or volatility <= 0:
        return None

    # ======================================
    # BUY CONDITIONS
    # ======================================

    bullish_15m = (
        ema([c["close"] for c in candles_15m], 9)
        >
        ema([c["close"] for c in candles_15m], 20)
    )

    bullish_5m = (
        fast_5m > slow_5m
        and entry > fast_5m
        and 52 <= rsi_5m <= 75
        and volume >= 1.1
    )

    bullish_breakout = (
        current["close"] > previous["high"]
        and current["close"] > ema(closes_1m, 9)
    )

    buy_signal = (
        market_trend == "BULLISH"
        and bullish_15m
        and bullish_5m
        and bullish_breakout
    )

    # ======================================
    # SELL CONDITIONS
    # ======================================

    bearish_15m = (
        ema([c["close"] for c in candles_15m], 9)
        <
        ema([c["close"] for c in candles_15m], 20)
    )

    bearish_5m = (
        fast_5m < slow_5m
        and entry < fast_5m
        and 25 <= rsi_5m <= 48
        and volume >= 1.1
    )

    bearish_breakdown = (
        current["close"] < previous["low"]
        and current["close"] < ema(closes_1m, 9)
    )

    sell_signal = (
        market_trend == "BEARISH"
        and bearish_15m
        and bearish_5m
        and bearish_breakdown
    )

    # ======================================
    # SIGNAL OUTPUT
    # ======================================

    if buy_signal:

        stop = entry - max(
            volatility * 1.5,
            entry * 0.002
        )

        risk = entry - stop
        target = entry + risk * 1.5

        return {
            "type": "BUY",
            "symbol": symbol,
            "entry": entry,
            "stop": stop,
            "target": target,
            "rsi": rsi_5m,
            "volume": volume
        }

    if sell_signal:

        stop = entry + max(
            volatility * 1.5,
            entry * 0.002
        )

        risk = stop - entry
        target = entry - risk * 1.5

        return {
            "type": "SELL",
            "symbol": symbol,
            "entry": entry,
            "stop": stop,
            "target": target,
            "rsi": rsi_5m,
            "volume": volume
        }

    return None


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 45)
    print("V10 CRYPTO DAY TRADING SCANNER")
    print("BUY + SELL ENABLED")
    print("=" * 45)

    market_trend = btc_trend()

    if market_trend is None:
        print("Could not establish BTC market trend.")
        send_telegram(
            "⚠️ CRYPTO SCANNER\n"
            "Unable to establish BTC market trend.\n"
            "No signals generated."
        )
        return

    print(f"BTC MARKET TREND: {market_trend}")

    signals = []

    for symbol in COINS:

        print(f"Scanning {symbol}...")

        try:
            signal = scan_coin(symbol, market_trend)

            if signal:
                signals.append(signal)

                print(
                    f"{signal['type']} SIGNAL: {symbol}"
                )

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

        time.sleep(0.25)

    # ======================================
    # TELEGRAM REPORT
    # ======================================

    if not signals:

        message = (
            "🔍 V10 CRYPTO DAY-TRADING SCANNER\n\n"
            "We couldn't find any qualified "
            "BUY or SELL entries right now.\n\n"
            f"BTC Market Trend: {market_trend}\n"
            "All approved cryptocurrencies scanned.\n\n"
            "Next scan in approximately 5 minutes."
        )

        send_telegram(message)

        print("No qualified entries found.")

        return

    for signal in signals:

        if signal["type"] == "BUY":

            header = "🟢 CRYPTO DAY-TRADING BUY ALERT"

        else:

            header = "🔴 CRYPTO DAY-TRADING SELL ALERT"

        message = (
            f"{header}\n\n"
            f"Coin: {signal['symbol']}\n"
            f"Entry Price: {signal['entry']:.8f}\n"
            f"Stop Loss: {signal['stop']:.8f}\n"
            f"Take Profit: {signal['target']:.8f}\n"
            f"Risk/Reward: 1:1.5\n"
            f"RSI: {signal['rsi']:.2f}\n"
            f"Relative Volume: {signal['volume']:.2f}x\n"
            f"BTC Trend: {market_trend}\n\n"
            f"Signal: {signal['type']}\n"
            "Signal only — no automatic orders."
        )

        send_telegram(message)

    print(f"Completed. Signals found: {len(signals)}")


if __name__ == "__main__":
    main()
