
import os
import re
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# V7 ELITE SWING SCANNER
# BUY ONLY | DAILY + WEEKLY | TELEGRAM

BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
NY = ZoneInfo("America/New_York")

MIN_PRICE = 5.0
MIN_SHARES = 300000
MIN_DOLLARS = 10000000
MAX_CANDIDATES = 150
MAX_ALERTS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/stocks/screener"
}


def telegram(msg):
    if not BOT or not CHAT:
        raise RuntimeError("Missing Telegram secrets.")

    r = requests.post(
        "https://api.telegram.org/bot" + BOT + "/sendMessage",
        data={"chat_id": CHAT, "text": msg},
        timeout=25
    )

    r.raise_for_status()

    if not r.json().get("ok"):
        raise RuntimeError("Telegram rejected message.")


def num(x):
    try:
        return float(
            str(x).replace("$", "").replace(",", "").replace("%", "").strip()
        )
    except (ValueError, TypeError):
        return 0.0


def universe():
    out = {}

    for ex in ("nasdaq", "nyse", "amex"):
        try:
            url = (
                "https://api.nasdaq.com/api/screener/stocks"
                "?tableonly=true&limit=5000&offset=0&exchange=" + ex
            )

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=30
            )

            response.raise_for_status()

            rows = response.json().get(
                "data", {}
            ).get("rows", [])

            for row in rows:
                s = str(row.get("symbol", "")).strip().upper()

                if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", s):
                    continue

                p = num(row.get("lastsale"))
                v = num(row.get("volume"))

                if (
                    p >= MIN_PRICE
                    and v >= MIN_SHARES
                    and p * v >= MIN_DOLLARS
                ):
                    out[s] = {
                        "dollar": p * v,
                        "change": num(row.get("pctchange"))
                    }

            logging.info("%s listings: %d", ex, len(rows))

        except Exception:
            logging.exception("Screener error: %s", ex)

        time.sleep(1)

    if not out:
        raise RuntimeError("No eligible US stocks found.")

    liquid = sorted(
        out,
        key=lambda s: out[s]["dollar"],
        reverse=True
    )[:MAX_CANDIDATES]

    movers = sorted(
        out,
        key=lambda s: out[s]["change"],
        reverse=True
    )[:MAX_CANDIDATES]

    return list(dict.fromkeys(liquid + movers))[:MAX_CANDIDATES]


def history(symbols):
    out = {}

    for i in range(0, len(symbols), 25):
        batch = symbols[i:i + 25]

        try:
            raw = yf.download(
                batch,
                period="18mo",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=False,
                timeout=35
            )

            for s in batch:
                try:
                    d = raw.copy() if len(batch) == 1 else raw[s].copy()

                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)

                    cols = [
                        "Open", "High", "Low",
                        "Close", "Volume"
                    ]

                    if all(c in d for c in cols):
                        d = d[cols].dropna()

                        if len(d) >= 210:
                            out[s] = d

                except Exception:
                    pass

        except Exception:
            logging.exception("Yahoo batch error")

        time.sleep(1)

    return out


def calc(d):
    x = d.copy()

    c = x.Close
    h = x.High
    l = x.Low
    v = x.Volume

    x["EMA20"] = c.ewm(
        span=20, adjust=False
    ).mean()

    x["EMA50"] = c.ewm(
        span=50, adjust=False
    ).mean()

    x["SMA50"] = c.rolling(50).mean()
    x["SMA200"] = c.rolling(200).mean()

    delta = c.diff()

    ag = delta.clip(lower=0).ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    al = (-delta.clip(upper=0)).ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    x["RSI"] = 100 - 100 / (
        1 + ag / al.replace(0, np.nan)
    )

    m = (
        c.ewm(span=12, adjust=False).mean()
        - c.ewm(span=26, adjust=False).mean()
    )

    x["HIST"] = m - m.ewm(
        span=9, adjust=False
    ).mean()

    pc = c.shift(1)

    tr = pd.concat(
        [
            h-l,
            (h-pc).abs(),
            (l-pc).abs()
        ],
        axis=1
    ).max(axis=1)

    x["ATR"] = tr.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    up = h.diff()
    down = -l.diff()

    pdm = pd.Series(
        np.where((up > down) & (up > 0), up, 0),
        index=x.index
    )

    mdm = pd.Series(
        np.where((down > up) & (down > 0), down, 0),
        index=x.index
    )

    sm = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    pdi = (
        100 * pdm.ewm(alpha=1/14, adjust=False).mean()
        / sm.replace(0, np.nan)
    )

    mdi = (
        100 * mdm.ewm(alpha=1/14, adjust=False).mean()
        / sm.replace(0, np.nan)
    )

    dx = (
        100 * (pdi-mdi).abs()
        / (pdi+mdi).replace(0, np.nan)
    )

    x["ADX"] = dx.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    x["PDI"] = pdi
    x["MDI"] = mdi

    x["AVGV"] = v.rolling(20).mean()
    x["HIGH20"] = h.shift(1).rolling(20).max()
    x["RET20"] = c.pct_change(20)

    return x


def weekly_ok(d):
    w = d.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    if len(w) < 30:
        return False

    e = w.Close.ewm(
        span=10,
        adjust=False
    ).mean()

    return bool(
        w.Close.iloc[-1] > e.iloc[-1]
        and e.iloc[-1] > e.iloc[-2]
    )


def market_ok(spy):
    x = calc(spy).dropna()

    if len(x) < 2:
        return False

    z = x.iloc[-1]

    return bool(
        z.Close > z.SMA50
        and z.SMA50 > z.SMA200
    )


def analyze(s, d, spy):
    x = calc(d).dropna()
    q = calc(spy).dropna()

    if len(x) < 3 or len(q) < 25:
        return None

    if not weekly_ok(d):
        return None

    a = x.iloc[-1]
    b = x.iloc[-2]

    trend = (
        a.Close > a.EMA20 > a.EMA50 > a.SMA200
        and a.SMA50 > a.SMA200
    )

    momentum = (
        52 <= a.RSI <= 68
        and a.HIST > 0
        and a.HIST >= b.HIST
        and a.ADX >= 18
        and a.PDI > a.MDI
    )

    vr = float(a.Volume / a.AVGV) if a.AVGV else 0

    relative = a.RET20 > q.RET20.iloc[-1]

    breakout = (
        a.Close > a.HIGH20
        and a.Close > a.Open
    )

    pullback = (
        a.Low <= a.EMA20 * 1.015
        and a.Close > a.EMA20
        and a.Close > a.Open
        and a.Close > b.Close
    )

    setup = (
        "20-day breakout" if breakout
        else "EMA20 pullback reclaim" if pullback
        else None
    )

    if not all([trend, momentum, vr >= 1.2, relative]) or not setup:
        return None

    price = float(a.Close)
    atr = float(a.ATR)

    if not np.isfinite(atr) or atr <= 0:
        return None

    stop = price - 2 * atr
    risk = price - stop

    score = (
        (25 if a.Close > a.SMA200 else 0)
        + (15 if a.EMA20 > a.EMA50 else 0)
        + (15 if a.RSI >= 55 else 10)
        + (15 if a.ADX >= 25 else 8)
        + (15 if vr >= 1.5 else 8)
        + (15 if (a.RET20-q.RET20.iloc[-1]) >= .05 else 8)
    )

    return {
        "symbol": s,
        "setup": setup,
        "price": price,
        "stop": stop,
        "target": price + 2.5 * risk,
        "rsi": a.RSI,
        "adx": a.ADX,
        "vr": vr,
        "rs": 100 * (a.RET20-q.RET20.iloc[-1]),
        "score": score,
        "date": x.index[-1].strftime("%Y-%m-%d")
    }


def alert(s):
    return (
        "📈 V7 ELITE SWING BUY WATCHLIST\n\n"
        f"Ticker: {s['symbol']}\n"
        f"Setup: {s['setup']}\n"
        f"Signal date: {s['date']}\n"
        f"Quality score: {s['score']}/100\n\n"
        f"Reference close: ${s['price']:.2f}\n"
        f"ATR stop reference: ${s['stop']:.2f}\n"
        f"Target reference (2.5R): ${s['target']:.2f}\n\n"
        f"RSI: {s['rsi']:.1f} | ADX: {s['adx']:.1f}\n"
        f"Relative volume: {s['vr']:.2f}x\n"
        f"20-day relative strength vs SPY: {s['rs']:+.2f}%\n\n"
        "Daily setup + weekly trend confirmation.\n"
        "Reference levels only. Verify live price, liquidity "
        "and earnings before entry. Not a guarantee."
    )


def main():
    if not BOT or not CHAT:
        raise RuntimeError("Telegram secrets are not configured.")

    now = datetime.now(NY)

    if now.weekday() >= 5:
        logging.info("Weekend; no scan.")
        return

    logging.info("V7 swing scan starting: %s ET", now.isoformat())

    names = universe()

    data = history(
        list(dict.fromkeys(names + ["SPY"]))
    )

    if "SPY" not in data:
        raise RuntimeError("SPY data unavailable; aborting.")

    if not market_ok(data["SPY"]):
        telegram(
            "V7 Swing Scanner: no BUY alerts. "
            "SPY is below the required 50/200-day bullish trend."
        )
        return

    signals = []

    for s in names:
        if s not in data:
            continue

        try:
            z = analyze(s, data[s], data["SPY"])

            if z:
                signals.append(z)

        except Exception:
            logging.exception("Analysis failed: %s", s)

    signals.sort(
        key=lambda z: z["score"],
        reverse=True
    )

    if not signals:
        telegram(
            "V7 Swing Scanner: no high-conviction BUY setups today. "
            "Market filter passed, but no stocks met every entry requirement."
        )
        return

    for z in signals[:MAX_ALERTS]:
        telegram(alert(z))
        time.sleep(1)

    logging.info(
        "Sent %d alerts.",
        min(len(signals), MAX_ALERTS)
    )


if __name__ == "__main__":
    main()
