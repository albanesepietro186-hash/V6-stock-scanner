
import os
import time
import requests
from datetime import datetime, timezone

# ==========================================
# V9 CRYPTO DAY-TRADING SCANNER
# BUY ONLY | 5-MINUTE SCANS | COINBASE USD
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
MAX_ALERTS = 10

session = requests.Session()
session.headers.update({
    "User-Agent": "V9-Crypto-Day-Scanner/1.0",
    "Accept": "application/json",
})


def get_json(url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=20
    )
    response.raise_for_status()
    return response.json()


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
    data = get_json(
        f"{BASE_URL}/products/{pair}/stats"
    )

    price = float(data["last"])
    volume = float(data["volume"])

    return price, price * volume


def get_candles(pair, granularity, count=100):
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

    # Coinbase format:
    # time, low, high, open, close, volume

    raw = sorted(raw, key=lambda candle: candle[0])

    # Exclude the currently forming candle.
    last_closed_start = (
        (now // granularity) * granularity
        - granularity
    )

    candles = []

    for c in raw:
        if int(c[0]) <= last_closed_start:
            candles.append({
                "time": int(c[0]),
                "low": float(c[1]),
                "high": float(c[2]),
                "open": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })

    return candles


def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = (
            value * multiplier
            + result * (1 - multiplier)
        )

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    gains = [max(x, 0) for x in changes]
    losses = [abs(min(x, 0)) for x in changes]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def atr(candles, period=14):
    if len(candles) < period + 1:
        return None

    ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous_close = candles[i - 1]["close"]

        true_range = max(
            current["high"] - current["low"],
            abs(current["high"] - previous_close),
            abs(current["low"] - previous_close),
        )

        ranges.append(true_range)

    return sum(ranges[-period:]) / period


def bullish_trend(candles):
    closes = [c["close"] for c in candles]

    fast = ema(closes, 9)
    slow = ema(closes, 20)

    if fast is None or slow is None:
        return False

    return (
        fast > slow
        and closes[-1] > fast
    )


def analyze_coin(pair, available_products, btc_bullish):
    if pair not in available_products:
        return None, f"{pair}: USD market unavailable; skipped"

    try:
        price, dollar_volume = get_stats(pair)

        if dollar_volume < MIN_DAILY_DOLLAR_VOLUME:
            return None, None

        one_minute = get_candles(pair, 60, 100)
        five_minute = get_candles(pair, 300, 100)
        fifteen_minute = get_candles(pair, 900, 100)

        if (
            len(one_minute) < 30
            or len(five_minute) < 30
            or len(fifteen_minute) < 30
        ):
            return None, f"{pair}: insufficient candle data"

        now = int(time.time())

        # Reject stale market data.
        if now - one_minute[-1]["time"] > 180:
            return None, f"{pair}: stale 1-minute data"

        if now - five_minute[-1]["time"] > 600:
            return None, f"{pair}: stale 5-minute data"

        if now - fifteen_minute[-1]["time"] > 1800:
            return None, f"{pair}: stale 15-minute data"

        if not btc_bullish:
            return None, None

        # 15-minute trend confirmation.
        if not bullish_trend(fifteen_minute):
            return None, None

        five_closes = [
            c["close"] for c in five_minute
        ]

        one_closes = [
            c["close"] for c in one_minute
        ]

        ema9 = ema(five_closes, 9)
        ema20 = ema(five_closes, 20)
        momentum_rsi = rsi(five_closes)

        if (
            ema9 is None
            or ema20 is None
            or momentum_rsi is None
        ):
            return None, None

        latest_5m = five_minute[-1]
        previous_20 = five_minute[-21:-1]

        if len(previous_20) < 20:
            return None, None

        average_volume = sum(
            c["volume"] for c in previous_20
        ) / 20

        if average_volume <= 0:
            return None, None

        relative_volume = (
            latest_5m["volume"] / average_volume
        )

        # 5-minute momentum and volume.
        if not (
            ema9 > ema20
            and latest_5m["close"] > ema9
            and 52 <= momentum_rsi <= 75
            and relative_volume >= 1.1
        ):
            return None, None

        # 1-minute entry confirmation.
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

        risk = max(
            volatility * 1.5,
            entry * 0.002
        )

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


def main():
    print("Starting V9 Crypto Day-Trading Scanner...")

    available_products = get_products()

    if "BTC-USD" not in available_products:
        raise RuntimeError("BTC-USD market unavailable.")

    # Bitcoin is the overall market trend filter.
    btc_candles = get_candles(
        "BTC-USD", 900, 100
    )

    if len(btc_candles) < 30:
        raise RuntimeError("Insufficient Bitcoin trend data.")

    btc_bullish = bullish_trend(btc_candles)

    signals = []
    errors = []
    checked = 0

    for pair in APPROVED_COINS:
        print(f"Scanning {pair}...")

        if pair in available_products:
            checked += 1

        signal, error = analyze_coin(
            pair,
            available_products,
            btc_bullish
        )

        if signal:
            signals.append(signal)
            print(f"BUY setup found: {pair}")

        elif error:
            errors.append(error)
            print(error)

        time.sleep(0.2)

    signals = signals[:MAX_ALERTS]

    if signals:
        message = [
            "🚨 V9 CRYPTO DAY-TRADING BUY ALERT",
            "1m entry | 5m setup | 15m trend",
            "",
        ]

        for signal in signals:
            message.extend([
                f"🟢 {signal['pair']}",
                f"Entry reference: ${signal['entry']:.8f}",
                f"Stop-loss reference: ${signal['stop']:.8f}",
                f"Target reference: ${signal['target']:.8f}",
                f"5m RSI: {signal['rsi']:.1f}",
                f"Relative volume: {signal['relative_volume']:.2f}x",
                f"24h dollar volume: ${signal['dollar_volume']:,.0f}",
                "",
            ])

        message.append(
            "Signals are not guaranteed fills or returns. "
            "Check fees, spread, slippage, and risk before trading."
        )

        send_telegram("\n".join(message))

    elif checked == 0:
        send_telegram(
            "V9 WARNING: No approved USD markets could be verified. "
            "Check GitHub Actions logs."
        )

    else:
        send_telegram(
            "V9 CRYPTO DAY-TRADING SCAN COMPLETE\n"
            "No qualifying BUY setups found."
        )

    if errors:
        print("Warnings:")
        for error in errors:
            print(error)

    print(
        f"Scan complete. Markets checked: {checked}. "
        f"BUY setups: {len(signals)}."
    )


if __name__ == "__main__":
    main()
