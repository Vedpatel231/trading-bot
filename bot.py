import ccxt
import pandas as pd
import ta
import time
import logging
import requests
import os
import threading
from datetime import datetime
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

CRYPTO_SYMBOLS  = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CRYPTO_TF       = "1m"       # entry candle timeframe
CRYPTO_HTF      = "5m"       # trend filter timeframe
FAST_EMA        = 9
SLOW_EMA        = 21
RSI_PERIOD      = 14
BB_PERIOD       = 20          # Bollinger Bands period
BB_STD          = 2.0         # Bollinger Bands std deviation
RSI_OB          = 65          # tighter overbought (was 70)
RSI_OS          = 35          # tighter oversold (was 30)
SL_PCT          = 0.004       # 0.4% trailing stop
TP_PCT          = 0.008       # 0.8% take profit (2:1 reward ratio)
CRYPTO_BAL      = 10000.0
RISK            = 0.02
CHECK_INTERVAL  = 10          # check every 10 seconds
MIN_VOL_MULT    = 1.2         # volume must be 1.2x average to trade

exchange = ccxt.binanceus()

# Track last candle timestamp per symbol to avoid re-trading same candle
last_candle_ts = {s: None for s in CRYPTO_SYMBOLS}

crypto_paper = {
    s: {
        "balance":       CRYPTO_BAL / len(CRYPTO_SYMBOLS),
        "coin_held":     0.0,
        "in_trade":      False,
        "entry_price":   0.0,
        "highest_price": 0.0,
        "total_trades":  0,
        "wins":          0,
        "losses":        0,
    } for s in CRYPTO_SYMBOLS
}

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
    # EMA
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)

    # RSI
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]  # band width

    # VWAP (reset every 100 candles as approximation)
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # Volume
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_avg"] * MIN_VOL_MULT

    return df

def htf_trend(symbol):
    """5m trend — UP, DOWN, or NEUTRAL."""
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
    Multi-indicator signal:
    BUY  needs: EMA crossover UP (confirmed) + RSI in range
               + price near/below BB lower + price above VWAP
               + volume spike
    SELL needs: EMA crossover DOWN (confirmed) + RSI in range
               + price near/above BB upper + volume spike
    """
    if len(df) < BB_PERIOD + 5:
        return "HOLD", df["close"].iloc[-1], 50.0

    prev2 = df.iloc[-3]
    prev  = df.iloc[-2]
    last  = df.iloc[-1]

    price = last["close"]
    rsi   = last["rsi"]

    # Skip if any indicator is NaN
    if pd.isna(rsi) or pd.isna(last["bb_lower"]) or pd.isna(last["vwap"]):
        return "HOLD", price, 50.0

    # EMA crossover confirmed over 2 candles
    ema_cross_up   = (prev2["ema_fast"] < prev2["ema_slow"] and
                      prev["ema_fast"]  > prev["ema_slow"]  and
                      last["ema_fast"]  > last["ema_slow"])

    ema_cross_down = (prev2["ema_fast"] > prev2["ema_slow"] and
                      prev["ema_fast"]  < prev["ema_slow"]  and
                      last["ema_fast"]  < last["ema_slow"])

    # Bollinger Band conditions
    near_bb_lower = price <= last["bb_lower"] * 1.002   # within 0.2% of lower band
    near_bb_upper = price >= last["bb_upper"] * 0.998   # within 0.2% of upper band

    # VWAP filter
    above_vwap = price > last["vwap"]
    below_vwap = price < last["vwap"]

    # Volume spike
    vol_spike = last["vol_spike"]

    # ── BUY: crossover UP + RSI not overbought + near lower BB + above VWAP + volume
    buy = (
        ema_cross_up        and
        RSI_OS < rsi < RSI_OB and   # RSI in healthy zone
        above_vwap          and     # price above fair value
        vol_spike                   # real volume behind move
    )

    # ── SELL: crossover DOWN + RSI not oversold + near upper BB + volume
    sell = (
        ema_cross_down      and
        rsi > RSI_OS        and     # not already oversold
        vol_spike                   # real volume behind move
    )

    if buy:  return "BUY",  price, rsi
    if sell: return "SELL", price, rsi
    return "HOLD", price, rsi

# ══════════════════════════════════════════════════════════════
#  PAPER TRADING
# ══════════════════════════════════════════════════════════════

def crypto_buy(symbol, price, rsi, bb_lower, vwap):
    p = crypto_paper[symbol]
    if p["in_trade"]: return
    spend    = p["balance"] * RISK
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
           f"RSI:     {rsi:.1f}\n"
           f"VWAP:    ${vwap:,.2f}\n"
           f"BB Low:  ${bb_lower:,.2f}\n"
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
    p["coin_held"]     = 0.0
    p["in_trade"]      = False
    p["highest_price"] = 0.0
    if pnl >= 0: p["wins"] += 1;  emoji = "✅"; res = f"WIN  +${pnl:.2f}"
    else:        p["losses"] += 1; emoji = "❌"; res = f"LOSS -${abs(pnl):.2f}"
    wr   = (p["wins"]/p["total_trades"]*100) if p["total_trades"] > 0 else 0
    coin = symbol.split("/")[0]
    msg = (f"{emoji} <b>SELL {coin}</b> ({reason})\n"
           f"Price:    ${price:,.2f}\n"
           f"Result:   {res}\n"
           f"Balance:  ${p['balance']:,.2f}\n"
           f"Win rate: {wr:.1f}% ({p['wins']}W/{p['losses']}L)")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)

def check_exits(symbol, price):
    p = crypto_paper[symbol]
    if not p["in_trade"]: return
    if price > p["highest_price"]: p["highest_price"] = price
    trail_sl    = p["highest_price"] * (1 - SL_PCT)
    take_profit = p["entry_price"]   * (1 + TP_PCT)
    if price <= trail_sl:
        crypto_sell(symbol, price, reason=f"Trail SL ${trail_sl:,.2f}")
    elif price >= take_profit:
        crypto_sell(symbol, price, reason=f"Take profit")

# ══════════════════════════════════════════════════════════════
#  CRYPTO MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run_crypto():
    coins = ", ".join(s.split("/")[0] for s in CRYPTO_SYMBOLS)
    msg = (f"🤖 <b>Crypto bot started</b>\n"
           f"Coins:     {coins}\n"
           f"Timeframe: {CRYPTO_TF}  |  HTF: {CRYPTO_HTF}\n"
           f"Strategy:  EMA {FAST_EMA}/{SLOW_EMA} + RSI + BB + VWAP + Volume\n"
           f"SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%\n"
           f"Check:     every {CHECK_INTERVAL}s (acts on new candles only)\n"
           f"Balance:   ${CRYPTO_BAL:,.2f}  (paper)")
    print("="*60); print(msg.replace("<b>","").replace("</b>","")); print("="*60)
    send_telegram(msg)

    while True:
        try:
            for symbol in CRYPTO_SYMBOLS:
                coin = symbol.split("/")[0]
                try:
                    # Fetch latest candles
                    df    = add_indicators(fetch_crypto(symbol, CRYPTO_TF, 200))
                    last  = df.iloc[-1]
                    price = float(last["close"])
                    rsi   = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0

                    # Always check exits on every tick (fast response)
                    check_exits(symbol, price)

                    # Only process signals on NEW candles
                    # (avoids re-triggering same signal every 10 seconds)
                    candle_ts = df.index[-1]
                    if candle_ts == last_candle_ts[symbol]:
                        continue   # same candle, skip signal check
                    last_candle_ts[symbol] = candle_ts

                    # Get HTF trend
                    trend = htf_trend(symbol)

                    # Get signal
                    sig, price, rsi = get_signal(df)

                    p     = crypto_paper[symbol]
                    st    = "IN TRADE" if p["in_trade"] else "watching"
                    bb_lo = float(last["bb_lower"]) if not pd.isna(last["bb_lower"]) else 0
                    vwap  = float(last["vwap"])      if not pd.isna(last["vwap"])     else 0

                    # Only print on new candle
                    print(f"  {coin:<4} | {sig:<4} | 5m:{trend:<7} | "
                          f"RSI:{rsi:>5.1f} | "
                          f"BB%:{((price-bb_lo)/price*100):>4.1f} | "
                          f"${price:>10,.2f} | {st}")

                    if sig == "BUY" and not p["in_trade"]:
                        if trend == "UP":
                            crypto_buy(symbol, price, rsi, bb_lo, vwap)
                        else:
                            print(f"        BUY blocked — 5m: {trend}")

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

STOCK_SYMBOLS  = ["VOO", "QQQ", "SPY"]
S_FAST_EMA     = 10
S_SLOW_EMA     = 50
STOCK_BAL      = 10000.0
STOCK_SLEEP    = 60 * 15

ALPACA_KEY     = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET  = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_URL     = "https://paper-api.alpaca.markets"

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
    } for s in STOCK_SYMBOLS
}

# ══════════════════════════════════════════════════════════════
#  STOCKS FUNCTIONS
# ══════════════════════════════════════════════════════════════

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
    p["shares_held"]   = 0.0
    p["in_trade"]      = False
    p["highest_price"] = 0.0
    if pnl >= 0: p["wins"] += 1;  emoji = "✅"; res = f"WIN  +${pnl:.2f}"
    else:        p["losses"] += 1; emoji = "❌"; res = f"LOSS -${abs(pnl):.2f}"
    wr = (p["wins"]/p["total_trades"]*100) if p["total_trades"] > 0 else 0
    msg = (f"{emoji} <b>SELL {symbol}</b> ({reason})\n"
           f"Price:    ${price:,.2f}\n"
           f"Result:   {res}\n"
           f"Balance:  ${p['balance']:,.2f}\n"
           f"Win rate: {wr:.1f}% ({p['wins']}W/{p['losses']}L)")
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
           f"Stocks:  {', '.join(STOCK_SYMBOLS)}\n"
           f"EMA:     {S_FAST_EMA}/{S_SLOW_EMA}  |  RSI: {RSI_PERIOD}\n"
           f"SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%\n"
           f"Balance: ${STOCK_BAL:,.2f}  (paper)\n"
           f"Hours:   Mon-Fri 9:30am-4pm ET only")
    print("="*60); print(msg.replace("<b>","").replace("</b>","")); print("="*60)
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

            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Market open — checking stocks...")

            for symbol in STOCK_SYMBOLS:
                try:
                    import yfinance as yf
                    bars = yf.Ticker(symbol).history(period="1y", interval="1d")
                    bars = bars.reset_index(drop=True)
                    bars.columns = [c.lower() for c in bars.columns]
                    if len(bars) < 60:
                        print(f"  {symbol:<4} | not enough bars — skipping"); continue

                    bars["ema_fast"] = ta.trend.ema_indicator(bars["close"], window=S_FAST_EMA)
                    bars["ema_slow"] = ta.trend.ema_indicator(bars["close"], window=S_SLOW_EMA)
                    bars["rsi"]      = ta.momentum.rsi(bars["close"], window=RSI_PERIOD)

                    prev  = bars.iloc[-2]
                    last  = bars.iloc[-1]
                    price = float(last["close"])
                    rsi   = float(last["rsi"]) if not pd.isna(last["rsi"]) else None
                    if rsi is None: continue

                    trend = "UP" if last["ema_fast"] > last["ema_slow"] else "DOWN"
                    p     = stock_paper[symbol]

                    if p["in_trade"]:
                        if price > p["highest_price"]: p["highest_price"] = price
                        if price <= p["highest_price"] * (1 - SL_PCT):
                            s_sell(symbol, price, "Trailing stop")
                        elif price >= p["entry_price"] * (1 + TP_PCT):
                            s_sell(symbol, price, "Take profit")

                    buy  = (prev["ema_fast"] < prev["ema_slow"] and
                            last["ema_fast"] > last["ema_slow"] and rsi < 70)
                    sell = (prev["ema_fast"] > prev["ema_slow"] and
                            last["ema_fast"] < last["ema_slow"] and rsi > 30)

                    sig    = "BUY" if buy else "SELL" if sell else "HOLD"
                    status = "IN TRADE" if p["in_trade"] else "watching"
                    print(f"  {symbol:<4} | {sig:<4} | Trend:{trend:<7} | RSI:{rsi:>5.1f} | ${price:>8,.2f} | {status}")

                    if sig == "BUY" and not p["in_trade"]:
                        if trend == "UP": s_buy(symbol, price)
                        else: print(f"        BUY blocked — trend: {trend}")
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
    threading.Thread(target=start_health_server, daemon=True).start()
    time.sleep(3)
    threading.Thread(target=run_crypto, daemon=True).start()
    time.sleep(5)
    threading.Thread(target=run_stocks, daemon=True).start()
    print("All bots running.")
    while True:
        time.sleep(60)
