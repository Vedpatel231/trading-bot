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
from alpaca_trade_api.rest import REST, TimeFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ══════════════════════════════════════════════════════════════
#  SHARED
# ══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
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

CRYPTO_SYMBOLS   = ["ETH/USDT", "SOL/USDT"]
CRYPTO_TF        = "1h"
CRYPTO_HTF       = "2h"
FAST_EMA         = 7
SLOW_EMA         = 18
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
RSI_OB           = 65
RSI_OS           = 35
SL_PCT           = 0.003
TP_PCT           = 0.015
CRYPTO_BAL       = 10000.0
RISK             = 0.02
CHECK_INTERVAL   = 60 * 5     # check every 5 min, act on 1h candles only

# ── Risk management settings ───────────────────────────────
DAILY_LOSS_LIMIT  = 0.03      # stop trading if down 3% today
COOLDOWN_MINUTES  = 30        # wait 30 min after a loss
AVOID_OPEN_MINS   = 0         # crypto trades 24/7, no open/close
MIN_VOL_MULT      = 1.2

exchange = ccxt.binanceus()
last_candle_ts = {s: None for s in CRYPTO_SYMBOLS}

crypto_paper = {
    s: {
        "balance":        CRYPTO_BAL / len(CRYPTO_SYMBOLS),
        "coin_held":      0.0,
        "in_trade":       False,
        "entry_price":    0.0,
        "highest_price":  0.0,
        "total_trades":   0,
        "wins":           0,
        "losses":         0,
        "daily_start_bal": CRYPTO_BAL / len(CRYPTO_SYMBOLS),
        "last_loss_time": None,    # cooldown tracking
        "total_pnl":      0.0,
        "best_trade":     0.0,
        "worst_trade":    0.0,
    } for s in CRYPTO_SYMBOLS
}

# Daily performance tracking
perf = {
    "date":       datetime.now().date(),
    "start_bal":  CRYPTO_BAL,
    "trades":     0,
    "wins":       0,
    "losses":     0,
    "pnl":        0.0,
    "paused":     False,
    "pause_reason": ""
}

# ══════════════════════════════════════════════════════════════
#  RISK MANAGEMENT CHECKS
# ══════════════════════════════════════════════════════════════

def reset_daily_stats():
    """Reset daily stats at midnight."""
    today = datetime.now().date()
    if perf["date"] != today:
        perf["date"]      = today
        perf["trades"]    = 0
        perf["wins"]      = 0
        perf["losses"]    = 0
        perf["pnl"]       = 0.0
        perf["paused"]    = False
        perf["pause_reason"] = ""
        total = sum(p["balance"] for p in crypto_paper.values())
        perf["start_bal"] = total
        for p in crypto_paper.values():
            p["daily_start_bal"] = p["balance"]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] New day — stats reset. Balance: ${total:,.2f}")

def is_trading_allowed(symbol):
    """Check all risk rules before allowing a trade."""
    reset_daily_stats()

    # 1. Daily loss limit
    total_bal   = sum(p["balance"] for p in crypto_paper.values())
    daily_loss  = (perf["start_bal"] - total_bal) / perf["start_bal"]
    if daily_loss >= DAILY_LOSS_LIMIT:
        if not perf["paused"]:
            perf["paused"]      = True
            perf["pause_reason"] = f"Daily loss limit hit ({daily_loss*100:.1f}%)"
            msg = (f"⛔ <b>Bot paused</b>\n"
                   f"Daily loss limit reached: {daily_loss*100:.1f}%\n"
                   f"Will resume tomorrow.")
            print(f"  {perf['pause_reason']}")
            send_telegram(msg)
        return False

    # 2. Cooldown after loss
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

def fetch_crypto(symbol, tf, limit=200):
    bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def add_indicators(df):
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()

    # MACD
    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST,
                          window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # ATR — for dynamic position sizing
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14)

    # VWAP
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # Volume
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_avg"] * MIN_VOL_MULT

    return df

def htf_trend(symbol):
    try:
        df = add_indicators(fetch_crypto(symbol, CRYPTO_HTF, 100))
        l  = df.iloc[-1]
        if l["ema_fast"] > l["ema_slow"]: return "UP"
        if l["ema_fast"] < l["ema_slow"]: return "DOWN"
        return "NEUTRAL"
    except:
        return "NEUTRAL"

def get_signal(df):
    """
    Multi-indicator signal with MACD confirmation added.
    BUY:  EMA cross UP (2 candle) + RSI zone + MACD bullish
          + above VWAP + volume spike
    SELL: EMA cross DOWN (2 candle) + RSI zone + MACD bearish
          + volume spike
    """
    if len(df) < max(BB_PERIOD, MACD_SLOW) + 5:
        return "HOLD", df["close"].iloc[-1], 50.0, 0.0

    prev2 = df.iloc[-3]
    prev  = df.iloc[-2]
    last  = df.iloc[-1]
    price = last["close"]
    rsi   = last["rsi"]
    atr   = last["atr"] if not pd.isna(last["atr"]) else 0

    if pd.isna(rsi) or pd.isna(last["bb_lower"]) or pd.isna(last["macd"]):
        return "HOLD", price, 50.0, atr

    ema_cross_up   = (prev2["ema_fast"] < prev2["ema_slow"] and
                      prev["ema_fast"]  > prev["ema_slow"]  and
                      last["ema_fast"]  > last["ema_slow"])

    ema_cross_down = (prev2["ema_fast"] > prev2["ema_slow"] and
                      prev["ema_fast"]  < prev["ema_slow"]  and
                      last["ema_fast"]  < last["ema_slow"])

    # MACD confirmation
    macd_bullish = (last["macd"] > last["macd_signal"] and
                    last["macd_hist"] > 0)
    macd_bearish = (last["macd"] < last["macd_signal"] and
                    last["macd_hist"] < 0)

    above_vwap = price > last["vwap"]
    vol_spike  = bool(last["vol_spike"])

    buy = (ema_cross_up   and
           RSI_OS < rsi < RSI_OB and
           macd_bullish   and
           above_vwap     and
           vol_spike)

    sell = (ema_cross_down and
            rsi > RSI_OS   and
            macd_bearish   and
            vol_spike)

    if buy:  return "BUY",  price, rsi, atr
    if sell: return "SELL", price, rsi, atr
    return "HOLD", price, rsi, atr

# ══════════════════════════════════════════════════════════════
#  ATR-BASED POSITION SIZING
# ══════════════════════════════════════════════════════════════

def calc_position_size(balance, price, atr):
    """
    Risk a fixed % of balance, but size position based on ATR.
    Smaller position when ATR is high (volatile market).
    Larger position when ATR is low (calm market).
    """
    risk_amount = balance * RISK
    if atr > 0:
        # Risk amount / ATR = number of units where 1 ATR move = risk amount
        atr_units = risk_amount / atr
        spend = min(atr_units * price, balance * 0.1)  # cap at 10% of balance
        spend = max(spend, 1.0)
    else:
        spend = risk_amount
    return round(spend, 2)

# ══════════════════════════════════════════════════════════════
#  PAPER TRADING
# ══════════════════════════════════════════════════════════════

def crypto_buy(symbol, price, rsi, atr):
    p = crypto_paper[symbol]
    if p["in_trade"]: return
    spend    = calc_position_size(p["balance"], price, atr)
    coin_qty = spend / price
    if spend < 1.0: return
    p["balance"]       -= spend
    p["coin_held"]      = coin_qty
    p["in_trade"]       = True
    p["entry_price"]    = price
    p["highest_price"]  = price
    coin = symbol.split("/")[0]
    msg = (f"🟢 <b>BUY {coin}</b>\n"
           f"Price:   ${price:,.2f}\n"
           f"Spent:   ${spend:.2f}\n"
           f"Amount:  {coin_qty:.6f} {coin}\n"
           f"RSI:     {rsi:.1f} | ATR: ${atr:.2f}\n"
           f"Balance: ${p['balance']:,.2f}\n"
           f"SL: ${price*(1-SL_PCT):,.2f}  |  TP: ${price*(1+TP_PCT):,.2f}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)

def crypto_sell(symbol, price, reason="Signal"):
    p = crypto_paper[symbol]
    if not p["in_trade"]: return
    proceeds = p["coin_held"] * price
    pnl      = proceeds - (p["coin_held"] * p["entry_price"])
    p["balance"]      += proceeds
    p["total_trades"] += 1
    p["total_pnl"]    += pnl
    p["coin_held"]     = 0.0
    p["in_trade"]      = False
    p["highest_price"] = 0.0

    if pnl >= 0:
        p["wins"]       += 1
        p["best_trade"]  = max(p["best_trade"], pnl)
        emoji = "✅"; res = f"WIN  +${pnl:.2f}"
    else:
        p["losses"]      += 1
        p["worst_trade"]  = min(p["worst_trade"], pnl)
        p["last_loss_time"] = datetime.now()   # start cooldown
        emoji = "❌"; res = f"LOSS -${abs(pnl):.2f}"

    # Update daily perf
    perf["trades"] += 1
    perf["pnl"]    += pnl
    if pnl >= 0: perf["wins"]   += 1
    else:        perf["losses"] += 1

    wr   = (p["wins"]/p["total_trades"]*100) if p["total_trades"] > 0 else 0
    coin = symbol.split("/")[0]
    msg = (f"{emoji} <b>SELL {coin}</b> ({reason})\n"
           f"Price:    ${price:,.2f}\n"
           f"Result:   {res}\n"
           f"Balance:  ${p['balance']:,.2f}\n"
           f"Win rate: {wr:.1f}% ({p['wins']}W/{p['losses']}L)\n"
           f"Total P&L: ${p['total_pnl']:+.2f}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)

def check_exits(symbol, price):
    p = crypto_paper[symbol]
    if not p["in_trade"]: return
    if price > p["highest_price"]: p["highest_price"] = price
    trail_sl    = p["highest_price"] * (1 - SL_PCT)
    take_profit = p["entry_price"]   * (1 + TP_PCT)
    if price <= trail_sl:
        crypto_sell(symbol, price, reason=f"Trail SL")
    elif price >= take_profit:
        crypto_sell(symbol, price, reason="Take profit")

# ══════════════════════════════════════════════════════════════
#  DAILY PERFORMANCE REPORT
# ══════════════════════════════════════════════════════════════

def send_daily_report():
    """Send a daily performance summary to Telegram at end of day."""
    total_bal = sum(p["balance"] for p in crypto_paper.values())
    total_pnl = sum(p["total_pnl"] for p in crypto_paper.values())
    total_w   = sum(p["wins"]   for p in crypto_paper.values())
    total_l   = sum(p["losses"] for p in crypto_paper.values())
    total_t   = total_w + total_l
    wr        = (total_w / total_t * 100) if total_t > 0 else 0
    best      = max(p["best_trade"]  for p in crypto_paper.values())
    worst     = min(p["worst_trade"] for p in crypto_paper.values())

    msg = (f"📊 <b>Daily Report — {datetime.now().strftime('%b %d %Y')}</b>\n\n"
           f"Balance:    ${total_bal:,.2f}\n"
           f"Today P&L:  ${perf['pnl']:+.2f}\n"
           f"Total P&L:  ${total_pnl:+.2f}\n\n"
           f"Trades:     {total_t}\n"
           f"Win rate:   {wr:.1f}% ({total_w}W / {total_l}L)\n"
           f"Best trade: +${best:.2f}\n"
           f"Worst trade: -${abs(worst):.2f}\n\n"
           f"{'⚠️ Daily loss limit was hit today' if perf['paused'] else '✅ No limits hit today'}")
    send_telegram(msg)
    print(f"\n{'='*55}\nDAILY REPORT SENT\n{'='*55}")

def schedule_daily_report():
    """Send daily report at midnight every day."""
    while True:
        now  = datetime.now()
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        sleep_secs = (next_midnight - now).total_seconds()
        time.sleep(sleep_secs)
        send_daily_report()

# ══════════════════════════════════════════════════════════════
#  CRYPTO MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run_crypto():
    coins = ", ".join(s.split("/")[0] for s in CRYPTO_SYMBOLS)
    msg = (f"🤖 <b>Crypto bot started</b>\n"
           f"Coins:     {coins}\n"
           f"Timeframe: {CRYPTO_TF}  |  HTF: {CRYPTO_HTF}\n"
           f"Strategy:  EMA {FAST_EMA}/{SLOW_EMA} + RSI + MACD + BB + VWAP + ATR\n"
           f"SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%\n"
           f"Daily limit: {DAILY_LOSS_LIMIT*100:.0f}%  |  Cooldown: {COOLDOWN_MINUTES}min\n"
           f"Balance:   ${CRYPTO_BAL:,.2f}  (paper)")
    print("="*58); print(msg.replace("<b>","").replace("</b>","")); print("="*58)
    send_telegram(msg)

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking crypto...")

            if perf["paused"]:
                reset_daily_stats()
                if not perf["paused"]:
                    print("New day — resuming trading")
                else:
                    time.sleep(CHECK_INTERVAL)
                    continue

            for symbol in CRYPTO_SYMBOLS:
                coin = symbol.split("/")[0]
                try:
                    df    = add_indicators(fetch_crypto(symbol, CRYPTO_TF, 200))
                    last  = df.iloc[-1]
                    price = float(last["close"])

                    # Always check exits fast
                    check_exits(symbol, price)

                    # Only signal on new candles
                    candle_ts = df.index[-1]
                    if candle_ts == last_candle_ts[symbol]:
                        continue
                    last_candle_ts[symbol] = candle_ts

                    trend           = htf_trend(symbol)
                    sig, price, rsi, atr = get_signal(df)
                    p               = crypto_paper[symbol]
                    st              = "IN TRADE" if p["in_trade"] else "watching"
                    macd_dir        = "↑" if not pd.isna(last["macd_hist"]) and last["macd_hist"] > 0 else "↓"
                    bb_pct          = ((price - float(last["bb_lower"])) / price * 100) if not pd.isna(last["bb_lower"]) else 0

                    print(f"  {coin:<4} | {sig:<4} | 2h:{trend:<7} | "
                          f"RSI:{rsi:>5.1f} | MACD:{macd_dir} | "
                          f"BB%:{bb_pct:>4.1f} | ${price:>10,.2f} | {st}")

                    if sig == "BUY" and not p["in_trade"]:
                        if trend == "UP" and is_trading_allowed(symbol):
                            crypto_buy(symbol, price, rsi, atr)
                        elif trend != "UP":
                            print(f"        BUY blocked — 2h: {trend}")

                    elif sig == "SELL" and p["in_trade"]:
                        crypto_sell(symbol, price, reason="Signal")

                except Exception as e:
                    print(f"  {coin} error: {e}")

        except Exception as e:
            print(f"Crypto loop error: {e}")

        time.sleep(CHECK_INTERVAL)

# ══════════════════════════════════════════════════════════════
#  STOCKS SETTINGS
# ══════════════════════════════════════════════════════════════

STOCK_SYMBOLS  = ["SPY", "QQQ"]
S_FAST_EMA     = 10
S_SLOW_EMA     = 50
STOCK_BAL      = 10000.0
STOCK_SLEEP    = 60 * 5      # check every 5 min
MARKET_OPEN_AVOID_MINS = 0    # trade full market hours 9:30am-4pm

ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_URL    = "https://paper-api.alpaca.markets"

stock_paper = {
    s: {
        "balance":       STOCK_BAL / len(STOCK_SYMBOLS),
        "shares_held":   0.0,
        "in_trade":      False,
        "entry_price":   0.0,
        "highest_price": 0.0,
        "total_trades":  0,
        "wins":          0,
        "losses":        0,
        "last_loss_time": None,
        "total_pnl":     0.0,
    } for s in STOCK_SYMBOLS
}

# ══════════════════════════════════════════════════════════════
#  STOCKS FUNCTIONS
# ══════════════════════════════════════════════════════════════

def is_safe_trading_time(alpaca):
    """Avoid first and last 30 min of market session."""
    try:
        clock = alpaca.get_clock()
        if not clock.is_open:
            return False
        now        = pd.Timestamp(clock.timestamp).tz_convert("America/New_York")
        market_open  = now.replace(hour=9,  minute=30, second=0)
        market_close = now.replace(hour=16, minute=0,  second=0)
        avoid_open   = market_open  + timedelta(minutes=MARKET_OPEN_AVOID_MINS)
        avoid_close  = market_close - timedelta(minutes=MARKET_OPEN_AVOID_MINS)
        if now < avoid_open:
            print(f"  Time filter: too close to open ({MARKET_OPEN_AVOID_MINS}min buffer)")
            return False
        if now > avoid_close:
            print(f"  Time filter: too close to close ({MARKET_OPEN_AVOID_MINS}min buffer)")
            return False
        return True
    except:
        return True

def s_buy(symbol, price):
    p = stock_paper[symbol]
    if p["in_trade"]: return
    spend  = p["balance"] * RISK
    shares = spend / price
    if spend < 1.0: return
    p["balance"]      -= spend
    p["shares_held"]   = shares
    p["in_trade"]      = True
    p["entry_price"]   = price
    p["highest_price"] = price
    msg = (f"🟢 <b>BUY {symbol}</b>\n"
           f"Price:   ${price:,.2f}\n"
           f"Spent:   ${spend:.2f}\n"
           f"Shares:  {shares:.4f}\n"
           f"Balance: ${p['balance']:,.2f}\n"
           f"SL: ${price*(1-SL_PCT):,.2f}  |  TP: ${price*(1+TP_PCT):,.2f}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)

def s_sell(symbol, price, reason="Signal"):
    p = stock_paper[symbol]
    if not p["in_trade"]: return
    proceeds = p["shares_held"] * price
    pnl      = proceeds - (p["shares_held"] * p["entry_price"])
    p["balance"]      += proceeds
    p["total_trades"] += 1
    p["total_pnl"]    += pnl
    p["shares_held"]   = 0.0
    p["in_trade"]      = False
    p["highest_price"] = 0.0
    if pnl >= 0: p["wins"] += 1;  emoji = "✅"; res = f"WIN  +${pnl:.2f}"
    else:
        p["losses"] += 1; emoji = "❌"; res = f"LOSS -${abs(pnl):.2f}"
        p["last_loss_time"] = datetime.now()
    wr = (p["wins"]/p["total_trades"]*100) if p["total_trades"] > 0 else 0
    msg = (f"{emoji} <b>SELL {symbol}</b> ({reason})\n"
           f"Price:    ${price:,.2f}\n"
           f"Result:   {res}\n"
           f"Balance:  ${p['balance']:,.2f}\n"
           f"Win rate: {wr:.1f}% ({p['wins']}W/{p['losses']}L)\n"
           f"Total P&L: ${p['total_pnl']:+.2f}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)

def run_stocks():
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("Stocks bot: No Alpaca keys — skipping"); return
    try:
        alpaca = REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)
        alpaca.get_account()
        print("Stocks bot: Alpaca connected!")
    except Exception as e:
        print(f"Stocks bot: connection failed — {e}"); return

    msg = (f"📈 <b>Stocks bot started</b>\n"
           f"Stocks:   {', '.join(STOCK_SYMBOLS)}\n"
           f"Strategy: EMA {S_FAST_EMA}/{S_SLOW_EMA} + RSI + BB + 200EMA + MACD + Vol\n"
           f"Time filter: avoid first/last {MARKET_OPEN_AVOID_MINS}min\n"
           f"Cooldown: {COOLDOWN_MINUTES}min after loss\n"
           f"Balance:  ${STOCK_BAL:,.2f}  (paper)")
    print("="*58); print(msg.replace("<b>","").replace("</b>","")); print("="*58)
    send_telegram(msg)

    while True:
        try:
            clock = alpaca.get_clock()
            if not clock.is_open:
                now  = pd.Timestamp(clock.timestamp)
                nxt  = pd.Timestamp(clock.next_open)
                mins = int((nxt - now).total_seconds() / 60)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed — {mins} min until open")
                time.sleep(60 * 15)
                continue

            if not is_safe_trading_time(alpaca):
                time.sleep(60)
                continue

            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Market open — checking stocks...")

            for symbol in STOCK_SYMBOLS:
                try:
                    import yfinance as yf
                    bars = yf.Ticker(symbol).history(period="2y", interval="1d")
                    bars = bars.reset_index(drop=True)
                    bars.columns = [c.lower() for c in bars.columns]
                    if len(bars) < 200:
                        print(f"  {symbol:<4} | not enough bars — skipping"); continue

                    bars["ema_fast"] = ta.trend.ema_indicator(bars["close"], window=S_FAST_EMA)
                    bars["ema_slow"] = ta.trend.ema_indicator(bars["close"], window=S_SLOW_EMA)
                    bars["ema_200"]  = ta.trend.ema_indicator(bars["close"], window=200)
                    bars["rsi"]      = ta.momentum.rsi(bars["close"], window=RSI_PERIOD)
                    bars["vol_avg"]  = bars["volume"].rolling(20).mean()
                    bars["vol_spike"]= bars["volume"] > bars["vol_avg"] * 1.2

                    macd_s = ta.trend.MACD(bars["close"], window_fast=12, window_slow=26, window_sign=9)
                    bars["macd"]        = macd_s.macd()
                    bars["macd_signal"] = macd_s.macd_signal()
                    bars["macd_hist"]   = macd_s.macd_diff()

                    prev  = bars.iloc[-2]
                    last  = bars.iloc[-1]
                    price = float(last["close"])
                    rsi   = float(last["rsi"])   if not pd.isna(last["rsi"])   else None
                    e200  = float(last["ema_200"]) if not pd.isna(last["ema_200"]) else None
                    if rsi is None or e200 is None: continue

                    above_200    = price > e200
                    vol_spike    = bool(last["vol_spike"])
                    macd_bullish = (not pd.isna(last["macd_hist"]) and last["macd_hist"] > 0)
                    macd_bearish = (not pd.isna(last["macd_hist"]) and last["macd_hist"] < 0)

                    ema_cross_up   = (prev["ema_fast"] < prev["ema_slow"] and
                                      last["ema_fast"] > last["ema_slow"])
                    ema_cross_down = (prev["ema_fast"] > prev["ema_slow"] and
                                      last["ema_fast"] < last["ema_slow"])

                    buy  = (ema_cross_up   and above_200 and
                            35 < rsi < 65  and macd_bullish and vol_spike)
                    sell = (ema_cross_down and rsi > 35 and
                            macd_bearish   and vol_spike)

                    sig    = "BUY" if buy else "SELL" if sell else "HOLD"
                    trend  = "UP" if last["ema_fast"] > last["ema_slow"] else "DOWN"
                    inst   = "↑" if above_200 else "↓"
                    status = "IN TRADE" if stock_paper[symbol]["in_trade"] else "watching"
                    p      = stock_paper[symbol]

                    if p["in_trade"]:
                        if price > p["highest_price"]: p["highest_price"] = price
                        if price <= p["highest_price"] * (1 - SL_PCT):
                            s_sell(symbol, price, "Trailing stop")
                        elif price >= p["entry_price"] * (1 + TP_PCT):
                            s_sell(symbol, price, "Take profit")

                    # Cooldown check
                    can_trade = True
                    if p["last_loss_time"]:
                        elapsed = (datetime.now() - p["last_loss_time"]).total_seconds() / 60
                        if elapsed < COOLDOWN_MINUTES:
                            can_trade = False

                    print(f"  {symbol:<4} | {sig:<4} | EMA:{trend:<5} | "
                          f"200:{inst} | MACD:{'↑' if macd_bullish else '↓'} | "
                          f"RSI:{rsi:>5.1f} | ${price:>8,.2f} | {status}")

                    if sig == "BUY" and not p["in_trade"] and can_trade:
                        s_buy(symbol, price)
                    elif sig == "SELL" and p["in_trade"]:
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
    # Health server FIRST — Railway needs this responding immediately
    threading.Thread(target=start_health_server, daemon=True).start()
    print(f"Health server running on port {os.getenv('PORT', 8080)}")

    # Small delay then start bots
    time.sleep(2)
    threading.Thread(target=run_crypto, daemon=True).start()
    time.sleep(3)
    threading.Thread(target=run_stocks, daemon=True).start()
    time.sleep(1)
    threading.Thread(target=schedule_daily_report, daemon=True).start()

    print("All bots running.")

    # Keep main thread alive forever — never let this crash
    while True:
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("Shutting down...")
            break
        except Exception as e:
            print(f"Main thread error (continuing): {e}")
            time.sleep(30)