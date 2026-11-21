
import os
import time
import requests
from datetime import datetime, timezone

# ==========================================
# V9 CRYPTO DAY-TRADING SCANNER
# Coinbase USD markets | BUY signals only
# ==========================================

BASE_URL = "https://api.exchange.coinbase.com"

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
    "BCH-USD",
]

MIN_DAILY_DOLLAR_VOLUME = 1_000_000
MIN_CANDLES = {
    60: 30,
    300: 30,
    900: 30,
}

session = requests.Session()
session.headers.update({
    "User-Agent": "V9-Crypto-Day-Scanner/1.0",
    "Accept": "application/json",
})


def get_json(url, params=None):
    response = session.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def get_products():
    products = get_json(f"{BASE_URL}/products")
    return {
        p["id"]
        for p in products
        if p.get("quote_currency") == "USD"
        and p.get("status") == "online"
        and p.get("trading_disabled") is not True
    }


def get_stats(pair):
    data = get_json(f"{BASE_URL}/products/{pair}/stats")
    last_price = float(data["last"])
    base_volume = float(data["volume"])
    return last_price, last_price * base_volume


def get_candles(pair, granularity, count):
    now = int(time.time())
    start = now - granularity * count

    params = {
        "start": datetime.fromtimestamp(
            start, timezone.utc
        ).isoformat(),
        "end": datetime.fromtimestamp(
            now, timezone.utc
        ).isoformat(),
        "granularity": granularity,
    }

    raw = get_json(
        f"{BASE_URL}/products/{pair}/candles",
        params=params
    )

    # Coinbase format: time, low, high, open, close, volume
    candles = sorted(raw, key=lambda x: x[0])

    # Remove the currently forming candle
    last_closed_start = (now // granularity) * granularity - granularity
    candles = [c for c in candles if c[0] <= last_closed_start]

    return [
        {
            "time": int(c[0]),
            "low": float(c[1]),
            "high": float(c[2]),
            "open": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        }
        for c in candles
    ]


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
    if len(values) < period + 1:
        return None

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    gains = [max(change, 0) for change in changes]
    losses = [abs(min(change, 0)) for change in changes]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    relative_strength = avg_gain / avg_loss
    return 100 - (100 / (1 + relative_strength))


def atr(candles, period=14):
    if len(candles) < period + 1:
        return None

    true_ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous_close = candles[i - 1]["close"]

        true_range = max(
            current["high"] - current["low"],
            abs(current["high"] - previous_close),
            abs(current["low"] - previous_close),
        )
        true_ranges.append(true_range)

    return sum(true_ranges[-period:]) / period


def bullish_trend(candles):
    closes = [c["close"] for c in candles]

    fast = ema(closes, 9)
    slow = ema(closes, 20)

    return (
        fast is not None
        and slow is not None
        and fast > slow
        and closes[-1] > fast
    )


def analyze_coin(pair, approved_products, btc_bullish):
    if pair not in approved_products:
        return None, f"{pair} has no verified online USD market"

    try:
        price, dollar_volume = get_stats(pair)

        if dollar_volume < MIN_DAILY_DOLLAR_VOLUME:
            return None, f"{pair} skipped: insufficient 24-hour dollar volume"

        one_minute = get_candles(pair, 60, 100)
        five_minute = get_candles(pair, 300, 100)
        fifteen_minute = get_candles(pair, 900, 100)

        if (
            len(one_minute) < MIN_CANDLES[60]
            or len(five_minute) < MIN_CANDLES[300]
            or len(fifteen_minute) < MIN_CANDLES[900]
        ):
            return None, f"{pair} skipped: insufficient candle data"

        now = int(time.time())

        # Reject stale market data
        if now - one_minute[-1]["time"] > 180:
            return None, f"{pair} skipped: stale 1-minute data"

        if now - five_minute[-1]["time"] > 600:
            return None, f"{pair} skipped: stale 5-minute data"

        if now - fifteen_minute[-1]["time"] > 1800:
            return None, f"{pair} skipped: stale 15-minute data"

        one_closes = [c["close"] for c in one_minute]
        five_closes = [c["close"] for c in five_minute]
        fifteen_closes = [c["close"] for c in fifteen_minute]

        # 15-minute trend confirmation
        if not bullish_trend(fifteen_minute):
            return None, None

        # 5-minute momentum confirmation
        ema9 = ema(five_closes, 9)
        ema20 = ema(five_closes, 20)
        momentum_rsi = rsi(five_closes)

        if ema9 is None or ema20 is None or momentum_rsi is None:
            return None, None

        latest_5m = five_minute[-1]
        previous_5m = five_minute[-21:-1]

        if len(previous_5m) < 20:
            return None, None

        average_volume = sum(
            c["volume"] for c in previous_5m
        ) / len(previous_5m)

        if average_volume <= 0:
            return None, None

        relative_volume = latest_5m["volume"] / average_volume

        if not (
            ema9 > ema20
            and latest_5m["close"] > ema9
            and 52 <= momentum_rsi <= 75
            and relative_volume >= 1.1
        ):
            return None, None

        # BTC market direction filter
        if not btc_bullish:
            return None, None

        # 1-minute entry confirmation:
        # latest closed candle must break the previous candle high
        latest_1m = one_minute[-1]
        previous_1m = one_minute[-2]

        one_minute_ema = ema(one_closes, 9)

        if one_minute_ema is None:
            return None, None

        if not (
            latest_1m["close"] > previous_1m["high"]
            and latest_1m["close"] > one_minute_ema
        ):
            return None, None

        entry = latest_1m["close"]
        volatility = atr(one_minute)

        if volatility is None or volatility <= 0:
            return None, None

        risk = max(volatility * 1.5, entry * 0.002)
        stop = entry - risk
        target = entry + (risk * 1.5)

        if stop <= 0:
            return None, None

        return {
            "pair": pair,
            "entry": entry,
            "stop": stop,
            "target": target,
            "rsi": momentum_rsi,
            "relative_volume": relative_volume,
            "dollar_volume": dollar_volume,
            "candle_time": latest_1m["time"],
        }, None

    except Exception as error:
        return None, f"{pair}: {str(error)[:180]}"


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError("Telegram secrets are missing.")

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )
    response.raise_for_status()


def main():
    print("Starting V9 Crypto Day-Trading Scanner...")

    approved_products = get_products()

    if "BTC-USD" not in approved_products:
        raise RuntimeError("BTC-USD market is unavailable.")

    # BTC must be bullish on the 15-minute timeframe
    btc_candles = get_candles("BTC-USD", 900, 100)
    btc_bullish = (
        len(btc_candles) >= MIN_CANDLES[900]
        and bullish_trend(btc_candles)
    )

    signals = []
    errors = []
    checked = 0

    for pair in APPROVED_COINS:
        signal, error = analyze_coin(
            pair,
            approved_products,
            btc_bullish
        )

        if signal:
            signals.append(signal)
            print(f"BUY setup found: {pair}")
        elif error:
            errors.append(error)
            print(error)
        else:
            print(f"No qualifying setup: {pair}")

        if pair in approved_products:
            checked += 1

        time.sleep(0.15)

    if signals:
        lines = [
            "🚨 V9 CRYPTO DAY-TRADING ALERT",
            "BUY setups only — review before trading",
            "",
        ]

        for signal in signals:
            lines.extend([
                f"🟢 {signal['pair']}",
                f"Entry reference: ${signal['entry']:.8f}",
                f"Stop-loss reference: ${signal['stop']:.8f}",
                f"Target reference: ${signal['target']:.8f}",
                f"5m RSI: {signal['rsi']:.1f}",
                f"5m relative volume: {signal['relative_volume']:.2f}x",
                f"24h dollar volume: ${signal['dollar_volume']:,.0f}",
                "",
            ])

        lines.append(
            "These are algorithmic signals, not guaranteed fills or returns. "
            "Check spread, fees, slippage, and risk before trading."
        )

        send_telegram("\n".join(lines))

    elif checked == 0:
        send_telegram(
            "⚠️ V9 Crypto Scanner could not verify any approved USD markets. "
            "Check the GitHub Actions logs."
        )

    else:
        send_telegram(
            "V9 Crypto Day-Trading Scan Complete.\n"
            "No qualifying BUY setups right now."
        )

    if errors:
        print("Warnings:")
        for error in errors:
            print(error)

    print(f"Scan complete. Markets checked: {checked}. BUY setups: {len(signals)}.")


if __name__ == "__main__":
    main()
