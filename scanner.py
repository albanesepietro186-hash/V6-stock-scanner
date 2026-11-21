
import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# V6 US STOCK DAY TRADING SCANNER
# BUY ONLY | TELEGRAM | GITHUB ACTIONS
# ============================================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

NY = ZoneInfo("America/New_York")
STATE_FILE = "v6_state.json"

# Scanner settings
MIN_PRICE = 5.00
MIN_DAILY_VOLUME = 100000
MAX_CANDIDATES = 100

# Risk settings
RISK_PER_TRADE = 0.005
MAX_DAILY_LOSS = 0.015
MAX_TRADES_PER_DAY = 4
MAX_CONSECUTIVE_LOSSES = 2

# Entry settings
RSI_MIN = 52
RSI_MAX = 68
ADX_MIN = 20
VOLUME_MULTIPLIER = 1.5
ATR_STOP_MULTIPLIER = 1.5

# Scan settings
SCAN_START = (9, 45)
SCAN_END = (15, 30)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/stocks/screener"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# TIME AND STATE
# ============================================================

def now_et():
    return datetime.now(NY)


def today_string():
    return now_et().strftime("%Y-%m-%d")


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        if state.get("date") != today_string():
            return {
                "date": today_string(),
                "signals": [],
                "trades": 0,
                "daily_loss": 0.0,
                "consecutive_losses": 0
            }

        return state

    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "date": today_string(),
            "signals": [],
            "trades": 0,
            "daily_loss": 0.0,
            "consecutive_losses": 0
        }


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        logging.error("Telegram credentials missing.")
        return False

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=20
        )

        response.raise_for_status()
        return response.json().get("ok", False)

    except requests.RequestException as error:
        logging.error("Telegram error: %s", error)
        return False


# ============================================================
# STOCK UNIVERSE
# ============================================================

def to_number(value):
    try:
        return float(
            str(value)
            .replace("$", "")
            .replace(",", "")
            .replace("%", "")
            .strip()
        )
    except (ValueError, TypeError):
        return 0.0


def get_universe():
    """
    Discover stocks listed on Nasdaq, NYSE and NYSE American.
    This is a public screener source, not a guaranteed
    complete or real-time exchange master list.
    """

    stocks = {}

    for exchange in ["nasdaq", "nyse", "amex"]:

        url = (
            "https://api.nasdaq.com/api/screener/stocks"
            f"?tableonly=true&limit=5000&offset=0"
            f"&exchange={exchange}"
        )

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=30
            )

            response.raise_for_status()

            rows = (
                response.json()
                .get("data", {})
                .get("rows", [])
            )

            logging.info(
                "%s returned %d listings",
                exchange,
                len(rows)
            )

            for row in rows:
                symbol = str(
                    row.get("symbol", "")
                ).strip().upper()

                if not re.fullmatch(
                    r"[A-Z][A-Z0-9.-]{0,9}",
                    symbol
                ):
                    continue

                price = to_number(row.get("lastsale"))
                volume = to_number(row.get("volume"))
                change = to_number(row.get("pctchange"))

                if price < MIN_PRICE:
                    continue

                if volume < MIN_DAILY_VOLUME:
                    continue

                stocks[symbol] = {
                    "price": price,
                    "volume": volume,
                    "change": change
                }

        except Exception as error:
            logging.error(
                "Universe error for %s: %s",
                exchange,
                error
            )

        time.sleep(1)

    if not stocks:
        raise RuntimeError(
            "Could not retrieve a usable US stock universe."
        )

    return stocks


def choose_candidates(universe):
    """
    Rank the discovered universe by volume and percentage
    movement. Analyze the highest-priority candidates.
    """

    volume_ranked = sorted(
        universe.items(),
        key=lambda item: item[1]["volume"],
        reverse=True
    )

    movement_ranked = sorted(
        universe.items(),
        key=lambda item: item[1]["change"],
        reverse=True
    )

    selected = {}

    for symbol, info in volume_ranked[:MAX_CANDIDATES]:
        selected[symbol] = info

    for symbol, info in movement_ranked[:MAX_CANDIDATES]:
        selected[symbol] = info

    return list(selected.keys())[:MAX_CANDIDATES]


# ============================================================
# MARKET DATA
# ============================================================

def download_history(symbols, interval, period):
    results = {}

    for start in range(0, len(symbols), 20):
        batch = symbols[start:start + 20]

        try:
            data = yf.download(
                batch,
                period=period,
                interval=interval,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=False,
                timeout=30
            )

            for symbol in batch:
                try:
                    if len(batch) == 1:
                        frame = data.copy()
                    else:
                        frame = data[symbol].copy()

                    if isinstance(frame.columns, pd.MultiIndex):
                        frame.columns = (
                            frame.columns.get_level_values(0)
                        )

                    if not frame.empty:
                        frame = frame.dropna(
                            subset=[
                                "Open",
                                "High",
                                "Low",
                                "Close",
                                "Volume"
                            ]
                        )

                        results[symbol] = frame

                except Exception:
                    logging.warning(
                        "No usable %s data",
                        symbol
                    )

        except Exception as error:
            logging.error(
                "Download error: %s",
                error
            )

        time.sleep(1)

    return results


def to_new_york_index(frame):
    frame = frame.copy()

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")

    frame.index = frame.index.tz_convert(NY)

    return frame


def regular_session(frame):
    frame = to_new_york_index(frame)

    frame = frame[
        (frame.index.time >= datetime.strptime(
            "09:30", "%H:%M"
        ).time())
        &
        (frame.index.time < datetime.strptime(
            "16:00", "%H:%M"
        ).time())
    ]

    return frame


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(frame):
    df = frame.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["EMA9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["EMA21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
        span=200, adjust=False
    ).mean()

    # RSI
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    # ATR
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    # ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        ),
        index=df.index
    )

    smoothed_atr = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        / smoothed_atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        / smoothed_atr.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["ADX"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    # Volume
    df["AVG_VOLUME"] = volume.rolling(20).mean()

    # Session VWAP
    typical_price = (
        high + low + close
    ) / 3

    session_dates = pd.Series(
        df.index.date,
        index=df.index
    )

    cumulative_pv = (
        typical_price * volume
    ).groupby(session_dates).cumsum()

    cumulative_volume = volume.groupby(
        session_dates
    ).cumsum()

    df["VWAP"] = (
        cumulative_pv
        / cumulative_volume.replace(0, np.nan)
    )

    return df


# ============================================================
# MARKET DIRECTION
# ============================================================

def market_is_bullish(market_data):
    for symbol in ["SPY", "QQQ"]:

        frame = market_data.get(symbol)

        if frame is None or len(frame) < 30:
            return False

        df = calculate_indicators(
            regular_session(frame)
        ).dropna()

        if len(df) < 2:
            return False

        last = df.iloc[-1]

        if not (
            last["Close"] > last["VWAP"]
            and last["EMA9"] > last["EMA21"]
        ):
            return False

    return True


# ============================================================
# BUY SIGNAL ENGINE
# ============================================================

def check_buy(symbol, five_data, fifteen_data, spy_data):

    if symbol not in five_data:
        return None

    if symbol not in fifteen_data:
        return None

    try:
        five = regular_session(five_data[symbol])
        fifteen = regular_session(fifteen_data[symbol])

        now = now_et()

        completed_before = now - timedelta(minutes=5)

        five = five[
            five.index + timedelta(minutes=5)
            <= completed_before
        ]

        five = five[
            five.index.date == now.date()
        ]

        if len(five) < 25:
            return None

        five = calculate_indicators(five).dropna()
        fifteen = calculate_indicators(fifteen).dropna()

        if len(five) < 3 or len(fifteen) < 200:
            return None

        last = five.iloc[-1]
        previous = five.iloc[-2]
        trend = fifteen.iloc[-1]

        trend_ok = (
            trend["Close"] > trend["EMA200"]
            and trend["EMA9"] > trend["EMA21"]
        )

        intraday_trend = (
            last["Close"] > last["VWAP"]
            and last["EMA9"] > last["EMA21"]
            and last["EMA21"] > previous["EMA21"]
        )

        recent = five.iloc[-7:-1]

        pullback = (
            (recent["Low"] <= recent["EMA9"])
            | (recent["Low"] <= recent["EMA21"])
        ).any()

        breakout = (
            last["Close"] > last["Open"]
            and last["Close"] > previous["High"]
        )

        rsi_ok = (
            RSI_MIN <= last["RSI"] <= RSI_MAX
        )

        macd_ok = (
            last["MACD_HIST"] > 0
            and last["MACD_HIST"] > previous["MACD_HIST"]
        )

        adx_ok = (
            last["ADX"] >= ADX_MIN
            and last["PLUS_DI"] > last["MINUS_DI"]
        )

        volume_ok = (
            last["AVG_VOLUME"] > 0
            and last["Volume"]
            >= VOLUME_MULTIPLIER * last["AVG_VOLUME"]
        )

        spy = regular_session(spy_data)
        spy = spy[spy.index.date == now.date()]

        relative_strength = False

        if len(spy) >= 2:
            stock_open = float(five_data[symbol]["Open"].iloc[0])
            stock_change = (
                float(last["Close"]) / stock_open
            ) - 1

            spy_open = float(spy["Open"].iloc[0])
            spy_change = (
                float(spy["Close"].iloc[-1]) / spy_open
            ) - 1

            relative_strength = stock_change > spy_change

        conditions = [
            trend_ok,
            intraday_trend,
            pullback,
            breakout,
            rsi_ok,
            macd_ok,
            adx_ok,
            volume_ok,
            relative_strength
        ]

        if not all(conditions):
            return None

        price = float(last["Close"])
        atr = float(last["ATR"])

        if not np.isfinite(atr) or atr <= 0:
            return None

        stop = price - (
            ATR_STOP_MULTIPLIER * atr
        )

        risk = price - stop

        return {
            "symbol": symbol,
            "price": price,
            "stop": stop,
            "target1": price + 2 * risk,
            "target2": price + 3 * risk,
            "rsi": float(last["RSI"]),
            "adx": float(last["ADX"]),
            "volume_ratio": float(
                last["Volume"] / last["AVG_VOLUME"]
            ),
            "candle": last.name.isoformat(),
            "time": now.strftime("%I:%M %p ET")
        }

    except Exception:
        logging.exception(
            "Signal error for %s",
            symbol
        )
        return None


# ============================================================
# ALERT FORMAT
# ============================================================

def format_alert(signal):
    return (
        "🚨 V6 DAY-TRADING BUY 🚨\n\n"
        f"STOCK: {signal['symbol']}\n"
        f"TIME: {signal['time']}\n\n"
        f"ENTRY REFERENCE: ${signal['price']:.2f}\n"
        f"STOP REFERENCE: ${signal['stop']:.2f}\n"
        f"TARGET 1 (2R): ${signal['target1']:.2f}\n"
        f"TARGET 2 (3R): ${signal['target2']:.2f}\n\n"
        "INDICATORS\n"
        f"RSI: {signal['rsi']:.1f}\n"
        f"ADX: {signal['adx']:.1f}\n"
        f"RELATIVE VOLUME: {signal['volume_ratio']:.2f}x\n\n"
        "BUY ONLY | 5-MINUTE ENTRY\n"
        "15-MINUTE TREND CONFIRMATION\n\n"
        "Signal only. Verify live prices before trading."
    )


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    now = now_et()

    if now.weekday() >= 5:
        logging.info("Market closed for weekend.")
        return

    current_time = (now.hour, now.minute)

    if not (
        SCAN_START <= current_time <= SCAN_END
    ):
        logging.info("Outside trading window.")
        return

    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID."
        )

    state = load_state()

    if state["trades"] >= MAX_TRADES_PER_DAY:
        logging.info("Daily trade limit reached.")
        return

    if state["daily_loss"] >= MAX_DAILY_LOSS:
        logging.info("Daily loss limit reached.")
        return

    if (
        state["consecutive_losses"]
        >= MAX_CONSECUTIVE_LOSSES
    ):
        logging.info("Consecutive loss limit reached.")
        return

    logging.info("Discovering US stock universe...")

    universe = get_universe()

    logging.info(
        "Discovered %d eligible stocks.",
        len(universe)
    )

    candidates = choose_candidates(universe)

    logging.info(
        "Analyzing %d priority candidates.",
        len(candidates)
    )

    symbols = list(
        set(candidates + ["SPY", "QQQ"])
    )

    five = download_history(
        symbols,
        interval="5m",
        period="5d"
    )

    fifteen = download_history(
        symbols,
        interval="15m",
        period="60d"
    )

    if "SPY" not in five or "QQQ" not in five:
        logging.error("Market data unavailable.")
        return

    if not market_is_bullish(five):
        logging.info("Market trend filter failed.")
        return

    alerts_sent = 0

    for symbol in candidates:

        if state["trades"] >= MAX_TRADES_PER_DAY:
            break

        if symbol not in five or symbol not in fifteen:
            continue

        signal = check_buy(
            symbol,
            five,
            fifteen,
            five["SPY"]
        )

        if not signal:
            continue

        signal_id = (
            symbol + "|" + signal["candle"]
        )

        if signal_id in state["signals"]:
            continue

        message = format_alert(signal)

        if send_telegram(message):

            state["signals"].append(signal_id)

            state["trades"] += 1

            save_state(state)

            alerts_sent += 1

            logging.info(
                "BUY alert sent for %s",
                symbol
            )

        time.sleep(1)

    save_state(state)

    logging.info(
        "Scan finished. Alerts sent: %d",
        alerts_sent
    )


if __name__ == "__main__":
    main()
