import ccxt
import pandas as pd
import ta
import time
import logging
import requests
import os
import threading
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from alpaca_trade_api.rest import REST

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ══════════════════════════════════════════════════════════════
#  SHARED
# ══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logging.error(f"Telegram error: {e}")


# ══════════════════════════════════════════════════════════════
#  HEALTH SERVER
# ══════════════════════════════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bots running")

    def log_message(self, format, *args):
        pass



def start_health_server():
    port = int(os.getenv("PORT", 8080))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


# ══════════════════════════════════════════════════════════════
#  CRYPTO SETTINGS
# ══════════════════════════════════════════════════════════════

CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CRYPTO_TF = "30m"
CRYPTO_HTF = "1h"
FAST_EMA = 7
SLOW_EMA = 18
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
CRYPTO_BAL = 10000.0
RISK = 0.02
CHECK_INTERVAL = 60 * 3

# crypto strategy / risk params
CRYPTO_STOP_ATR = 1.2
CRYPTO_TP_ATR = 2.5
BREAKOUT_LOOKBACK = 10
BREAKOUT_VOL_MULT = 1.5
STRONG_BODY_MULT = 1.2
REGIME_MIN_ATR_PCT = 0.002
PULLBACK_BUFFER = 0.003
COOLDOWN_MINUTES = 30
DAILY_LOSS_LIMIT = 0.03

# Trading fees (Coinbase Advanced tier 1: maker 0.40%, taker 0.60%)
# Using taker fee since bot uses market orders
# Change these when you move to a different exchange or tier
TRADING_FEE_PCT = 0.006   # 0.60% taker fee per trade (Coinbase Advanced default)
# Other exchanges for reference:
#   Binance US:  0.001 (0.10% maker/taker)
#   Coinbase T1: 0.006 (0.60% taker) / 0.004 (0.40% maker)
#   Coinbase T2: 0.004 (0.40% taker) — $10k+ monthly volume
#   Kraken:      0.0026 (0.26% taker)
#   Bybit:       0.001 (0.10% taker)

exchange = ccxt.binanceus()
last_candle_ts = {s: None for s in CRYPTO_SYMBOLS}
last_seen_prices = {s: None for s in CRYPTO_SYMBOLS}
prev_regime = {s: {"up": False, "down": False} for s in CRYPTO_SYMBOLS}

crypto_paper = {
    s: {
        "balance": CRYPTO_BAL / len(CRYPTO_SYMBOLS),
        "coin_held": 0.0,
        "in_trade": False,
        "trade_direction": "",
        "entry_price": 0.0,
        "entry_atr": 0.0,
        "stop_price": 0.0,
        "tp_price": 0.0,
        "entry_strategy": "",
        "highest_price": 0.0,
        "lowest_price": 999999.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "daily_start_bal": CRYPTO_BAL / len(CRYPTO_SYMBOLS),
        "last_loss_time": None,
        "total_pnl": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
    }
    for s in CRYPTO_SYMBOLS
}

perf = {
    "date": datetime.now().date(),
    "start_bal": CRYPTO_BAL,
    "trades": 0,
    "wins": 0,
    "losses": 0,
    "pnl": 0.0,
    "paused": False,
    "pause_reason": "",
}


# ══════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
#  RISK MANAGEMENT CHECKS
# ══════════════════════════════════════════════════════════════

def get_crypto_symbol_equity(symbol, price=None):
    p = crypto_paper[symbol]
    equity = p["balance"]
    if p["in_trade"] and p["coin_held"] > 0:
        mark_price = price
        if mark_price is None:
            mark_price = last_seen_prices.get(symbol)
        if mark_price is None or mark_price <= 0:
            mark_price = p["entry_price"]
        equity += p["coin_held"] * mark_price
    return equity


def get_total_crypto_equity(price_overrides=None):
    price_overrides = price_overrides or {}
    return sum(get_crypto_symbol_equity(symbol, price_overrides.get(symbol)) for symbol in CRYPTO_SYMBOLS)


def reset_daily_stats():
    today = datetime.now().date()
    if perf["date"] != today:
        perf["date"] = today
        perf["trades"] = 0
        perf["wins"] = 0
        perf["losses"] = 0
        perf["pnl"] = 0.0
        perf["paused"] = False
        perf["pause_reason"] = ""
        total_equity = get_total_crypto_equity()
        perf["start_bal"] = total_equity
        for symbol in CRYPTO_SYMBOLS:
            crypto_paper[symbol]["daily_start_bal"] = get_crypto_symbol_equity(symbol)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] New day — stats reset. Equity: ${total_equity:,.2f}")



def is_trading_allowed(symbol):
    reset_daily_stats()
    total_equity = get_total_crypto_equity()
    daily_loss = (perf["start_bal"] - total_equity) / perf["start_bal"] if perf["start_bal"] > 0 else 0
    if daily_loss >= DAILY_LOSS_LIMIT:
        if not perf["paused"]:
            perf["paused"] = True
            perf["pause_reason"] = f"Daily loss limit hit ({daily_loss*100:.1f}%)"
            msg = (
                f"⛔ <b>Bot paused</b>\n"
                f"Daily loss limit reached: {daily_loss*100:.1f}%\n"
                f"Will resume tomorrow."
            )
            print(f"  {perf['pause_reason']}")
            send_telegram(msg)
        return False

    p = crypto_paper[symbol]
    if p["last_loss_time"]:
        elapsed = (datetime.now() - p["last_loss_time"]).total_seconds() / 60
        if elapsed < COOLDOWN_MINUTES:
            remaining = int(COOLDOWN_MINUTES - elapsed)
            print(f"  {symbol.split('/')[0]} cooldown — {remaining} min remaining")
            return False
    return True


# ══════════════════════════════════════════════════════════════
#  DATA & INDICATORS
# ══════════════════════════════════════════════════════════════

def fetch_crypto(symbol, tf, limit=250):
    bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df



def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    macd = ta.trend.MACD(
        df["close"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=ATR_PERIOD)
    df["atr_pct"] = df["atr"] / df["close"]

    df["vol_avg"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_avg"] * BREAKOUT_VOL_MULT

    df["body"] = (df["close"] - df["open"]).abs()
    df["body_avg"] = df["body"].rolling(20).mean()
    df["bullish_candle"] = df["close"] > df["open"]
    df["bearish_candle"] = df["close"] < df["open"]
    df["strong_bullish"] = df["bullish_candle"] & (df["body"] > df["body_avg"] * STRONG_BODY_MULT)
    df["strong_bearish"] = df["bearish_candle"] & (df["body"] > df["body_avg"] * STRONG_BODY_MULT)

    df["recent_high"] = df["high"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["recent_low"] = df["low"].shift(1).rolling(BREAKOUT_LOOKBACK).min()
    return df



def get_crypto_regime(symbol):
    try:
        df = add_indicators(fetch_crypto(symbol, CRYPTO_HTF, 120))
        last = df.iloc[-1]
        prev = df.iloc[-2]
        fast_above = safe_float(last["ema_fast"]) > safe_float(last["ema_slow"])
        fast_below = safe_float(last["ema_fast"]) < safe_float(last["ema_slow"])
        slow_rising = safe_float(last["ema_slow"]) > safe_float(prev["ema_slow"])
        slow_falling = safe_float(last["ema_slow"]) < safe_float(prev["ema_slow"])
        atr_pct = safe_float(last["atr_pct"])

        regime_up = fast_above and slow_rising and atr_pct >= REGIME_MIN_ATR_PCT
        regime_down = fast_below and slow_falling and atr_pct >= REGIME_MIN_ATR_PCT
        regime_not_bearish = fast_above or slow_rising
        regime_not_bullish = fast_below or slow_falling

        if fast_above:
            trend_label = "UP"
        elif fast_below and slow_falling:
            trend_label = "DOWN"
        elif not fast_above and slow_rising:
            trend_label = "RECOVERING"
        else:
            trend_label = "NEUTRAL"

        return {
            "label": trend_label, "up": regime_up, "down": regime_down,
            "not_bearish": regime_not_bearish, "not_bullish": regime_not_bullish,
            "slow_rising": slow_rising, "slow_falling": slow_falling, "atr_pct": atr_pct,
        }
    except Exception:
        return {"label": "NEUTRAL", "up": False, "down": False, "not_bearish": False,
                "not_bullish": False, "slow_rising": False, "slow_falling": False, "atr_pct": 0.0}



def get_crypto_signal(df, regime, prev_regime_state):
    if len(df) < max(MACD_SLOW + 10, BREAKOUT_LOOKBACK + 5):
        return {"signal": "HOLD", "strategy": "none", "direction": "", "price": safe_float(df["close"].iloc[-1]), "atr": 0.0, "rsi": 50.0}

    prev = df.iloc[-2]
    last = df.iloc[-1]
    price = safe_float(last["close"])
    atr = safe_float(last["atr"])
    rsi = safe_float(last["rsi"], 50.0)

    ema_cross_up = safe_float(prev["ema_fast"]) < safe_float(prev["ema_slow"]) and safe_float(last["ema_fast"]) > safe_float(last["ema_slow"])
    ema_cross_down = safe_float(prev["ema_fast"]) > safe_float(prev["ema_slow"]) and safe_float(last["ema_fast"]) < safe_float(last["ema_slow"])
    ema_already_above = safe_float(last["ema_fast"]) > safe_float(last["ema_slow"])
    ema_already_below = safe_float(last["ema_fast"]) < safe_float(last["ema_slow"])

    macd_bullish = safe_float(last["macd"]) > safe_float(last["macd_signal"]) and safe_float(last["macd_hist"]) > 0
    macd_bearish = safe_float(last["macd"]) < safe_float(last["macd_signal"]) and safe_float(last["macd_hist"]) < 0
    macd_improving = safe_float(last["macd_hist"]) > safe_float(prev["macd_hist"])
    macd_worsening = safe_float(last["macd_hist"]) < safe_float(prev["macd_hist"])

    regime_just_flipped_up = regime["up"] and not prev_regime_state.get("up", False)
    regime_just_flipped_down = regime["down"] and not prev_regime_state.get("down", False)

    # ══ BULLISH (LONG) ════════════════════════════════
    trend_buy = ema_cross_up and macd_bullish and regime["up"]
    breakout_buy = (price > safe_float(last["recent_high"]) and bool(last["strong_bullish"])
                    and bool(last["vol_spike"]) and regime["not_bearish"])
    prev_near_ema = safe_float(prev["close"]) <= safe_float(prev["ema_fast"]) * (1 + PULLBACK_BUFFER)
    rebound_candle = bool(last["bullish_candle"]) and price > safe_float(last["ema_fast"]) and price > safe_float(prev["high"])
    pullback_buy = regime["up"] and prev_near_ema and rebound_candle and macd_improving
    regimeflip_buy = regime_just_flipped_up and ema_already_above and macd_bullish

    # ══ BEARISH (SHORT) ═══════════════════════════════
    short_trend = ema_cross_down and macd_bearish and regime["down"]
    short_breakout = (price < safe_float(last["recent_low"]) and bool(last["strong_bearish"])
                      and bool(last["vol_spike"]) and regime["not_bullish"])
    prev_near_ema_above = safe_float(prev["close"]) >= safe_float(prev["ema_fast"]) * (1 - PULLBACK_BUFFER)
    rejection_candle = bool(last["bearish_candle"]) and price < safe_float(last["ema_fast"]) and price < safe_float(prev["low"])
    short_pullback = regime["down"] and prev_near_ema_above and rejection_candle and macd_worsening
    short_regimeflip = regime_just_flipped_down and ema_already_below and macd_bearish

    sell_signal = ema_cross_down and macd_bearish
    cover_signal = ema_cross_up and macd_bullish

    if trend_buy:        return {"signal": "BUY",   "strategy": "Trend",           "direction": "long",  "price": price, "atr": atr, "rsi": rsi}
    if breakout_buy:     return {"signal": "BUY",   "strategy": "Breakout",        "direction": "long",  "price": price, "atr": atr, "rsi": rsi}
    if pullback_buy:     return {"signal": "BUY",   "strategy": "Pullback",        "direction": "long",  "price": price, "atr": atr, "rsi": rsi}
    if regimeflip_buy:   return {"signal": "BUY",   "strategy": "RegimeFlip",      "direction": "long",  "price": price, "atr": atr, "rsi": rsi}
    if short_trend:      return {"signal": "SHORT", "strategy": "ShortTrend",      "direction": "short", "price": price, "atr": atr, "rsi": rsi}
    if short_breakout:   return {"signal": "SHORT", "strategy": "ShortBreakout",   "direction": "short", "price": price, "atr": atr, "rsi": rsi}
    if short_pullback:   return {"signal": "SHORT", "strategy": "ShortPullback",   "direction": "short", "price": price, "atr": atr, "rsi": rsi}
    if short_regimeflip: return {"signal": "SHORT", "strategy": "ShortRegimeFlip", "direction": "short", "price": price, "atr": atr, "rsi": rsi}
    if sell_signal:      return {"signal": "SELL",  "strategy": "Signal",          "direction": "",      "price": price, "atr": atr, "rsi": rsi}
    if cover_signal:     return {"signal": "COVER", "strategy": "Signal",          "direction": "",      "price": price, "atr": atr, "rsi": rsi}
    return {"signal": "HOLD", "strategy": "none", "direction": "", "price": price, "atr": atr, "rsi": rsi}


# ══════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════

def calc_position_size(balance, price, atr, stop_atr_mult):
    risk_amount = balance * RISK
    if atr > 0:
        stop_distance = atr * stop_atr_mult
        qty = risk_amount / stop_distance if stop_distance > 0 else 0
        spend = min(qty * price, balance * 0.25)
        spend = max(spend, 1.0)
    else:
        spend = max(risk_amount, 1.0)
    return round(spend, 2)


# ══════════════════════════════════════════════════════════════
#  PAPER TRADING - CRYPTO
# ══════════════════════════════════════════════════════════════

def crypto_buy(symbol, price, rsi, atr, strategy):
    p = crypto_paper[symbol]
    if p["in_trade"]:
        return
    spend = calc_position_size(p["balance"], price, atr, CRYPTO_STOP_ATR)
    if spend < 1.0 or spend > p["balance"]:
        return

    # Deduct entry fee
    entry_fee = spend * TRADING_FEE_PCT
    coin_qty = (spend - entry_fee) / price  # fee reduces how much coin you get
    stop_price = price - (atr * CRYPTO_STOP_ATR) if atr > 0 else price * 0.99
    tp_price = price + (atr * CRYPTO_TP_ATR) if atr > 0 else price * 1.02

    p["balance"] -= spend
    p["coin_held"] = coin_qty
    p["in_trade"] = True
    p["trade_direction"] = "long"
    p["entry_price"] = price
    p["entry_atr"] = atr
    p["stop_price"] = stop_price
    p["tp_price"] = tp_price
    p["entry_strategy"] = strategy
    p["highest_price"] = price

    coin = symbol.split("/")[0]
    msg = (
        f"🟢 <b>LONG {coin}</b>\n"
        f"Strategy: {strategy}\n"
        f"Price:   ${price:,.2f}\n"
        f"Spent:   ${spend:.2f} (fee: ${entry_fee:.2f})\n"
        f"Amount:  {coin_qty:.6f} {coin}\n"
        f"RSI:     {rsi:.1f} | ATR: ${atr:.2f}\n"
        f"Balance: ${p['balance']:,.2f}\n"
        f"Stop: ${stop_price:,.2f} | TP: ${tp_price:,.2f}"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>', '').replace('</b>', '')}")
    send_telegram(msg)



def crypto_short(symbol, price, rsi, atr, strategy):
    """Open a short position — bet on price going DOWN."""
    p = crypto_paper[symbol]
    if p["in_trade"]:
        return
    spend = calc_position_size(p["balance"], price, atr, CRYPTO_STOP_ATR)
    if spend < 1.0 or spend > p["balance"]:
        return

    entry_fee = spend * TRADING_FEE_PCT
    coin_qty = (spend - entry_fee) / price
    stop_price = price + (atr * CRYPTO_STOP_ATR) if atr > 0 else price * 1.01
    tp_price = price - (atr * CRYPTO_TP_ATR) if atr > 0 else price * 0.98

    p["balance"] -= spend
    p["coin_held"] = coin_qty
    p["in_trade"] = True
    p["trade_direction"] = "short"
    p["entry_price"] = price
    p["entry_atr"] = atr
    p["stop_price"] = stop_price
    p["tp_price"] = tp_price
    p["entry_strategy"] = strategy
    p["lowest_price"] = price

    coin = symbol.split("/")[0]
    msg = (
        f"🔴 <b>SHORT {coin}</b>\n"
        f"Strategy: {strategy}\n"
        f"Price:   ${price:,.2f}\n"
        f"Spent:   ${spend:.2f} (fee: ${entry_fee:.2f})\n"
        f"Amount:  {coin_qty:.6f} {coin}\n"
        f"RSI:     {rsi:.1f} | ATR: ${atr:.2f}\n"
        f"Balance: ${p['balance']:,.2f}\n"
        f"Stop: ${stop_price:,.2f} | TP: ${tp_price:,.2f}"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>', '').replace('</b>', '')}")
    send_telegram(msg)


def crypto_close(symbol, price, reason="Signal"):
    """Close any open position — works for both long and short. Deducts exit fee."""
    p = crypto_paper[symbol]
    if not p["in_trade"]:
        return

    direction = p["trade_direction"]

    if direction == "long":
        proceeds = p["coin_held"] * price
        exit_fee = proceeds * TRADING_FEE_PCT
        proceeds_after_fee = proceeds - exit_fee
        pnl = proceeds_after_fee - (p["coin_held"] * p["entry_price"])
        p["balance"] += proceeds_after_fee
    elif direction == "short":
        raw_pnl = p["coin_held"] * (p["entry_price"] - price)
        exit_value = p["coin_held"] * price
        exit_fee = exit_value * TRADING_FEE_PCT
        pnl = raw_pnl - exit_fee
        p["balance"] += p["coin_held"] * p["entry_price"] + pnl
    else:
        proceeds = p["coin_held"] * price
        exit_fee = proceeds * TRADING_FEE_PCT
        pnl = (proceeds - exit_fee) - (p["coin_held"] * p["entry_price"])
        p["balance"] += proceeds - exit_fee

    p["total_trades"] += 1
    p["total_pnl"] += pnl
    p["coin_held"] = 0.0
    p["in_trade"] = False
    p["trade_direction"] = ""
    p["highest_price"] = 0.0
    p["lowest_price"] = 999999.0
    p["entry_price"] = 0.0
    p["entry_atr"] = 0.0
    p["stop_price"] = 0.0
    p["tp_price"] = 0.0
    p["entry_strategy"] = ""

    if pnl >= 0:
        p["wins"] += 1
        p["best_trade"] = max(p["best_trade"], pnl)
        emoji = "✅"
        res = f"WIN  +${pnl:.2f}"
    else:
        p["losses"] += 1
        p["worst_trade"] = min(p["worst_trade"], pnl)
        p["last_loss_time"] = datetime.now()
        emoji = "❌"
        res = f"LOSS -${abs(pnl):.2f}"

    perf["trades"] += 1
    perf["pnl"] += pnl
    if pnl >= 0:
        perf["wins"] += 1
    else:
        perf["losses"] += 1

    wr = (p["wins"] / p["total_trades"] * 100) if p["total_trades"] > 0 else 0
    coin = symbol.split("/")[0]
    dir_label = "CLOSE LONG" if direction == "long" else "CLOSE SHORT" if direction == "short" else "CLOSE"
    msg = (
        f"{emoji} <b>{dir_label} {coin}</b> ({reason})\n"
        f"Price:    ${price:,.2f}\n"
        f"Fee:      ${exit_fee:.2f}\n"
        f"Result:   {res}\n"
        f"Balance:  ${p['balance']:,.2f}\n"
        f"Win rate: {wr:.1f}% ({p['wins']}W/{p['losses']}L)\n"
        f"Total P&L: ${p['total_pnl']:+.2f}"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>', '').replace('</b>', '')}")
    send_telegram(msg)



def check_crypto_exits(symbol, price):
    p = crypto_paper[symbol]
    if not p["in_trade"]:
        return

    if p["trade_direction"] == "long":
        if price > p["highest_price"]:
            p["highest_price"] = price
        if price <= p["stop_price"]:
            crypto_close(symbol, price, reason="ATR stop")
        elif price >= p["tp_price"]:
            crypto_close(symbol, price, reason="ATR take profit")

    elif p["trade_direction"] == "short":
        if price < p["lowest_price"]:
            p["lowest_price"] = price
        if price >= p["stop_price"]:
            crypto_close(symbol, price, reason="ATR stop")
        elif price <= p["tp_price"]:
            crypto_close(symbol, price, reason="ATR take profit")


# ══════════════════════════════════════════════════════════════
#  DAILY REPORT
# ══════════════════════════════════════════════════════════════

def send_daily_report():
    total_equity = get_total_crypto_equity()
    total_pnl = sum(p["total_pnl"] for p in crypto_paper.values())
    total_w = sum(p["wins"] for p in crypto_paper.values())
    total_l = sum(p["losses"] for p in crypto_paper.values())
    total_t = total_w + total_l
    wr = (total_w / total_t * 100) if total_t > 0 else 0
    best = max(p["best_trade"] for p in crypto_paper.values())
    worst = min(p["worst_trade"] for p in crypto_paper.values())

    msg = (
        f"📊 <b>Daily Report — {datetime.now().strftime('%b %d %Y')}</b>\n\n"
        f"Equity:     ${total_equity:,.2f}\n"
        f"Today P&L:  ${perf['pnl']:+.2f}\n"
        f"Total P&L:  ${total_pnl:+.2f}\n\n"
        f"Trades:     {total_t}\n"
        f"Win rate:   {wr:.1f}% ({total_w}W / {total_l}L)\n"
        f"Best trade: +${best:.2f}\n"
        f"Worst trade: -${abs(worst):.2f}\n\n"
        f"{'⚠️ Daily loss limit was hit today' if perf['paused'] else '✅ No limits hit today'}"
    )
    send_telegram(msg)
    print(f"\n{'='*55}\nDAILY REPORT SENT\n{'='*55}")



def schedule_daily_report():
    while True:
        now = datetime.now()
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        sleep_secs = (next_midnight - now).total_seconds()
        time.sleep(sleep_secs)
        send_daily_report()


# ══════════════════════════════════════════════════════════════
#  CRYPTO MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run_crypto():
    coins = ", ".join(s.split("/")[0] for s in CRYPTO_SYMBOLS)
    msg = (
        f"🤖 <b>Crypto bot started</b>\n"
        f"Coins:     {coins}\n"
        f"Timeframe: {CRYPTO_TF}  |  HTF: {CRYPTO_HTF}\n"
        f"Long:      Trend + Breakout + Pullback + RegimeFlip\n"
        f"Short:     ShortTrend + ShortBreakout + ShortPullback + ShortRegimeFlip\n"
        f"Risk:      Stop {CRYPTO_STOP_ATR} ATR | TP {CRYPTO_TP_ATR} ATR\n"
        f"Fee:       {TRADING_FEE_PCT*100:.1f}% per trade (Coinbase taker)\n"
        f"Daily limit: {DAILY_LOSS_LIMIT*100:.0f}%  |  Cooldown: {COOLDOWN_MINUTES}min\n"
        f"Balance:   ${CRYPTO_BAL:,.2f}  (paper)"
    )
    print("=" * 58)
    print(msg.replace("<b>", "").replace("</b>", ""))
    print("=" * 58)
    send_telegram(msg)

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking crypto...")
            if perf["paused"]:
                reset_daily_stats()
                if perf["paused"]:
                    time.sleep(CHECK_INTERVAL)
                    continue
                print("New day — resuming trading")

            for symbol in CRYPTO_SYMBOLS:
                coin = symbol.split("/")[0]
                try:
                    df = add_indicators(fetch_crypto(symbol, CRYPTO_TF, 250))
                    last = df.iloc[-1]
                    price = safe_float(last["close"])
                    last_seen_prices[symbol] = price
                    check_crypto_exits(symbol, price)

                    candle_ts = df.index[-1]
                    if candle_ts == last_candle_ts[symbol]:
                        continue
                    last_candle_ts[symbol] = candle_ts

                    regime = get_crypto_regime(symbol)
                    sig = get_crypto_signal(df, regime, prev_regime[symbol])
                    prev_regime[symbol] = {"up": regime["up"], "down": regime["down"]}
                    p = crypto_paper[symbol]
                    macd_dir = "↑" if safe_float(last["macd_hist"]) > 0 else "↓"

                    if p["in_trade"]:
                        dir_label = "LONG" if p["trade_direction"] == "long" else "SHORT"
                        st = f"IN {dir_label}"
                    else:
                        st = "watching"

                    print(
                        f"  {coin:<4} | {sig['signal']:<5} | {sig['strategy']:<14} | "
                        f"1h:{regime['label']:<10} | MACD:{macd_dir} | "
                        f"ATR%:{regime['atr_pct']*100:>4.2f} | ${sig['price']:>10,.2f} | {st}"
                    )

                    if not p["in_trade"] and is_trading_allowed(symbol):
                        if sig["signal"] == "BUY":
                            if sig["strategy"] == "Breakout" and not regime["not_bearish"]:
                                print("        BUY blocked — regime bearish")
                            else:
                                crypto_buy(symbol, sig["price"], sig["rsi"], sig["atr"], sig["strategy"])
                        elif sig["signal"] == "SHORT":
                            if sig["strategy"] == "ShortBreakout" and not regime["not_bullish"]:
                                print("        SHORT blocked — regime bullish")
                            else:
                                crypto_short(symbol, sig["price"], sig["rsi"], sig["atr"], sig["strategy"])

                    elif p["in_trade"]:
                        if sig["signal"] == "SELL" and p["trade_direction"] == "long":
                            crypto_close(symbol, sig["price"], reason="Signal")
                        elif sig["signal"] == "COVER" and p["trade_direction"] == "short":
                            crypto_close(symbol, sig["price"], reason="Cover signal")

                except Exception as e:
                    print(f"  {coin} error: {e}")

        except Exception as e:
            print(f"Crypto loop error: {e}")

        time.sleep(CHECK_INTERVAL)


# ══════════════════════════════════════════════════════════════
#  STOCKS SETTINGS
# ══════════════════════════════════════════════════════════════

STOCK_SYMBOLS = ["SPY", "QQQ", "TSLA", "NVDA", "AMD"]
S_FAST_EMA = 10
S_SLOW_EMA = 50
STOCK_BAL = 10000.0
STOCK_SLEEP = 60 * 5
MARKET_OPEN_AVOID_MINS = 0
STOCK_STOP_ATR = 1.2
STOCK_TP_ATR = 2.5
STOCK_PULLBACK_BUFFER = 0.003

ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_URL = "https://paper-api.alpaca.markets"

stock_paper = {
    s: {
        "balance": STOCK_BAL / len(STOCK_SYMBOLS),
        "shares_held": 0.0,
        "in_trade": False,
        "entry_price": 0.0,
        "entry_atr": 0.0,
        "stop_price": 0.0,
        "tp_price": 0.0,
        "entry_strategy": "",
        "highest_price": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "last_loss_time": None,
        "total_pnl": 0.0,
    }
    for s in STOCK_SYMBOLS
}


# ══════════════════════════════════════════════════════════════
#  STOCKS FUNCTIONS
# ══════════════════════════════════════════════════════════════

def is_safe_trading_time(alpaca):
    try:
        clock = alpaca.get_clock()
        if not clock.is_open:
            return False
        now = pd.Timestamp(clock.timestamp).tz_convert("America/New_York")
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        avoid_open = market_open + timedelta(minutes=MARKET_OPEN_AVOID_MINS)
        avoid_close = market_close - timedelta(minutes=MARKET_OPEN_AVOID_MINS)
        if now < avoid_open:
            print(f"  Time filter: too close to open ({MARKET_OPEN_AVOID_MINS}min buffer)")
            return False
        if now > avoid_close:
            print(f"  Time filter: too close to close ({MARKET_OPEN_AVOID_MINS}min buffer)")
            return False
        return True
    except Exception:
        return True



def add_stock_indicators(bars):
    bars = bars.copy()
    bars["ema_fast"] = ta.trend.ema_indicator(bars["close"], window=S_FAST_EMA)
    bars["ema_slow"] = ta.trend.ema_indicator(bars["close"], window=S_SLOW_EMA)
    bars["ema_200"] = ta.trend.ema_indicator(bars["close"], window=200)
    bars["atr"] = ta.volatility.average_true_range(bars["high"], bars["low"], bars["close"], window=ATR_PERIOD)
    bars["body"] = (bars["close"] - bars["open"]).abs()
    bars["bullish_candle"] = bars["close"] > bars["open"]
    bars["body_avg"] = bars["body"].rolling(20).mean()

    macd = ta.trend.MACD(bars["close"], window_fast=12, window_slow=26, window_sign=9)
    bars["macd"] = macd.macd()
    bars["macd_signal"] = macd.macd_signal()
    bars["macd_hist"] = macd.macd_diff()
    return bars



def stock_regime(last, prev):
    above_200 = safe_float(last["close"]) > safe_float(last["ema_200"])
    ema200_rising = safe_float(last["ema_200"]) > safe_float(prev["ema_200"])
    return {"up": above_200 and ema200_rising, "label": "UP" if above_200 else "DOWN", "ema200_rising": ema200_rising}



def get_stock_signal(bars):
    prev = bars.iloc[-2]
    last = bars.iloc[-1]
    regime = stock_regime(last, prev)

    ema_cross_up = safe_float(prev["ema_fast"]) < safe_float(prev["ema_slow"]) and safe_float(last["ema_fast"]) > safe_float(last["ema_slow"])
    ema_cross_down = safe_float(prev["ema_fast"]) > safe_float(prev["ema_slow"]) and safe_float(last["ema_fast"]) < safe_float(last["ema_slow"])
    macd_bullish = safe_float(last["macd_hist"]) > 0
    macd_bearish = safe_float(last["macd_hist"]) < 0
    macd_improving = safe_float(last["macd_hist"]) > safe_float(prev["macd_hist"])

    trend_buy = ema_cross_up and macd_bullish and regime["up"]

    prev_near_ema = safe_float(prev["close"]) <= safe_float(prev["ema_fast"]) * (1 + STOCK_PULLBACK_BUFFER)
    rebound_candle = bool(last["bullish_candle"]) and safe_float(last["close"]) > safe_float(last["ema_fast"]) and safe_float(last["close"]) > safe_float(prev["high"])
    pullback_buy = regime["up"] and prev_near_ema and rebound_candle and macd_improving

    sell_signal = ema_cross_down and macd_bearish

    if trend_buy:
        return "BUY", "Trend", regime
    if pullback_buy:
        return "BUY", "Pullback", regime
    if sell_signal:
        return "SELL", "Signal", regime
    return "HOLD", "none", regime



def s_buy(symbol, price, atr, strategy):
    p = stock_paper[symbol]
    if p["in_trade"]:
        return
    spend = calc_position_size(p["balance"], price, atr, STOCK_STOP_ATR)
    if spend < 1.0 or spend > p["balance"]:
        return
    shares = spend / price
    stop_price = price - (atr * STOCK_STOP_ATR) if atr > 0 else price * 0.99
    tp_price = price + (atr * STOCK_TP_ATR) if atr > 0 else price * 1.02

    p["balance"] -= spend
    p["shares_held"] = shares
    p["in_trade"] = True
    p["entry_price"] = price
    p["entry_atr"] = atr
    p["stop_price"] = stop_price
    p["tp_price"] = tp_price
    p["entry_strategy"] = strategy
    p["highest_price"] = price

    msg = (
        f"🟢 <b>BUY {symbol}</b>\n"
        f"Strategy: {strategy}\n"
        f"Price:   ${price:,.2f}\n"
        f"Spent:   ${spend:.2f}\n"
        f"Shares:  {shares:.4f}\n"
        f"Balance: ${p['balance']:,.2f}\n"
        f"Stop: ${stop_price:,.2f} | TP: ${tp_price:,.2f}"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>', '').replace('</b>', '')}")
    send_telegram(msg)



def s_sell(symbol, price, reason="Signal"):
    p = stock_paper[symbol]
    if not p["in_trade"]:
        return
    proceeds = p["shares_held"] * price
    pnl = proceeds - (p["shares_held"] * p["entry_price"])
    p["balance"] += proceeds
    p["total_trades"] += 1
    p["total_pnl"] += pnl
    p["shares_held"] = 0.0
    p["in_trade"] = False
    p["highest_price"] = 0.0
    p["entry_price"] = 0.0
    p["entry_atr"] = 0.0
    p["stop_price"] = 0.0
    p["tp_price"] = 0.0
    p["entry_strategy"] = ""

    if pnl >= 0:
        p["wins"] += 1
        emoji = "✅"
        res = f"WIN  +${pnl:.2f}"
    else:
        p["losses"] += 1
        p["last_loss_time"] = datetime.now()
        emoji = "❌"
        res = f"LOSS -${abs(pnl):.2f}"

    wr = (p["wins"] / p["total_trades"] * 100) if p["total_trades"] > 0 else 0
    msg = (
        f"{emoji} <b>SELL {symbol}</b> ({reason})\n"
        f"Price:    ${price:,.2f}\n"
        f"Result:   {res}\n"
        f"Balance:  ${p['balance']:,.2f}\n"
        f"Win rate: {wr:.1f}% ({p['wins']}W/{p['losses']}L)\n"
        f"Total P&L: ${p['total_pnl']:+.2f}"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>', '').replace('</b>', '')}")
    send_telegram(msg)



def check_stock_exits(symbol, price):
    p = stock_paper[symbol]
    if not p["in_trade"]:
        return
    if price > p["highest_price"]:
        p["highest_price"] = price
    if price <= p["stop_price"]:
        s_sell(symbol, price, "ATR stop")
    elif price >= p["tp_price"]:
        s_sell(symbol, price, "ATR take profit")



def run_stocks():
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("Stocks bot: No Alpaca keys — skipping")
        return
    try:
        alpaca = REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)
        alpaca.get_account()
        print("Stocks bot: Alpaca connected!")
    except Exception as e:
        print(f"Stocks bot: connection failed — {e}")
        return

    msg = (
        f"📈 <b>Stocks bot started</b>\n"
        f"Stocks:   {', '.join(STOCK_SYMBOLS)}\n"
        f"Modes:    Trend + Pullback\n"
        f"Risk:     Stop {STOCK_STOP_ATR} ATR | TP {STOCK_TP_ATR} ATR\n"
        f"Time filter: avoid first/last {MARKET_OPEN_AVOID_MINS}min\n"
        f"Cooldown: {COOLDOWN_MINUTES}min after loss\n"
        f"Balance:  ${STOCK_BAL:,.2f}  (paper)"
    )
    print("=" * 58)
    print(msg.replace("<b>", "").replace("</b>", ""))
    print("=" * 58)
    send_telegram(msg)

    while True:
        try:
            clock = alpaca.get_clock()
            if not clock.is_open:
                now = pd.Timestamp(clock.timestamp)
                nxt = pd.Timestamp(clock.next_open)
                mins = int((nxt - now).total_seconds() / 60)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed — {mins} min until open")
                time.sleep(60 * 15)
                continue

            if not is_safe_trading_time(alpaca):
                time.sleep(60)
                continue

            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Market open — checking stocks...")

            import yfinance as yf

            for symbol in STOCK_SYMBOLS:
                try:
                    bars = yf.Ticker(symbol).history(period="2y", interval="1d")
                    bars = bars.reset_index(drop=True)
                    bars.columns = [c.lower() for c in bars.columns]
                    if len(bars) < 220:
                        print(f"  {symbol:<4} | not enough bars — skipping")
                        continue

                    bars = add_stock_indicators(bars)
                    prev = bars.iloc[-2]
                    last = bars.iloc[-1]
                    price = safe_float(last["close"])
                    atr = safe_float(last["atr"])
                    signal, strategy, regime = get_stock_signal(bars)
                    p = stock_paper[symbol]
                    status = "IN TRADE" if p["in_trade"] else "watching"
                    macd_dir = "↑" if safe_float(last["macd_hist"]) > 0 else "↓"

                    check_stock_exits(symbol, price)

                    can_trade = True
                    if p["last_loss_time"]:
                        elapsed = (datetime.now() - p["last_loss_time"]).total_seconds() / 60
                        if elapsed < COOLDOWN_MINUTES:
                            can_trade = False

                    print(
                        f"  {symbol:<4} | {signal:<4} | {strategy:<8} | "
                        f"200:{regime['label']:<4} | MACD:{macd_dir} | ${price:>8,.2f} | {status}"
                    )

                    if signal == "BUY" and not p["in_trade"] and can_trade:
                        s_buy(symbol, price, atr, strategy)
                    elif signal == "SELL" and p["in_trade"]:
                        s_sell(symbol, price)

                    time.sleep(1)
                except Exception as e:
                    print(f"  {symbol} error: {e}")
        except Exception as e:
            print(f"Stocks loop error: {e}")
        time.sleep(STOCK_SLEEP)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Thread(target=start_health_server, daemon=True).start()
    print(f"Health server running on port {os.getenv('PORT', 8080)}")

    time.sleep(2)
    threading.Thread(target=run_crypto, daemon=True).start()
    time.sleep(3)
    threading.Thread(target=run_stocks, daemon=True).start()
    time.sleep(1)
    threading.Thread(target=schedule_daily_report, daemon=True).start()

    print("All bots running.")

    while True:
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("Shutting down...")
            break
        except Exception as e:
            print(f"Main thread error (continuing): {e}")
            time.sleep(30)
