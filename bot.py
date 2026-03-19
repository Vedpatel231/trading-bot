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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# ══════════════════════════════════════════════════════════════
#  SHARED SETTINGS
# ══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       msg,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# ══════════════════════════════════════════════════════════════
#  HEALTH CHECK SERVER — keeps Railway alive
# ══════════════════════════════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bots are running")
    def log_message(self, format, *args):
        pass

def start_health_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"Health server running on port {port}")
    server.serve_forever()

# ══════════════════════════════════════════════════════════════
#  CRYPTO BOT SETTINGS
# ══════════════════════════════════════════════════════════════

CRYPTO_SYMBOLS      = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CRYPTO_TIMEFRAME    = "5m"       # entry timeframe — changed to 5m
CRYPTO_HTF          = "1h"       # higher timeframe — changed to 1h
FAST_EMA            = 5
SLOW_EMA            = 50
RSI_PERIOD          = 14
RSI_OVERBOUGHT      = 70
RSI_OVERSOLD        = 30
STOP_LOSS_PCT       = 0.02
TAKE_PROFIT_PCT     = 0.04
CRYPTO_BALANCE      = 10000.0
RISK_PER_TRADE      = 0.02
CRYPTO_INTERVAL     = 60 * 5    # check every 5 minutes

exchange = ccxt.binanceus()

crypto_paper = {
    symbol: {
        "balance":       CRYPTO_BALANCE / len(CRYPTO_SYMBOLS),
        "coin_held":     0.0,
        "in_trade":      False,
        "entry_price":   0.0,
        "highest_price": 0.0,
        "total_trades":  0,
        "wins":          0,
        "losses":        0,
    }
    for symbol in CRYPTO_SYMBOLS
}

# ══════════════════════════════════════════════════════════════
#  CRYPTO DATA & INDICATORS
# ══════════════════════════════════════════════════════════════

def fetch_crypto_candles(symbol, timeframe, limit=200):
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def add_indicators(df):
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["vol_avg"]  = df["volume"].rolling(window=20).mean()
    return df

def get_htf_trend_crypto(symbol):
    try:
        df   = fetch_crypto_candles(symbol, CRYPTO_HTF, limit=100)
        df   = add_indicators(df)
        last = df.iloc[-1]
        if last["ema_fast"] > last["ema_slow"]:   return "UP"
        elif last["ema_fast"] < last["ema_slow"]: return "DOWN"
        return "NEUTRAL"
    except Exception as e:
        logging.error(f"Crypto HTF error ({symbol}): {e}")
        return "NEUTRAL"

def get_signal(df):
    prev = df.iloc[-2]
    last = df.iloc[-1]
    buy = (
        prev["ema_fast"] < prev["ema_slow"] and
        last["ema_fast"] > last["ema_slow"] and
        last["rsi"]      < RSI_OVERBOUGHT   and
        last["volume"]   > last["vol_avg"] * 0.8
    )
    sell = (
        prev["ema_fast"] > prev["ema_slow"] and
        last["ema_fast"] < last["ema_slow"] and
        last["rsi"]      > RSI_OVERSOLD
    )
    if buy:  return "BUY",  last["close"], last["rsi"]
    if sell: return "SELL", last["close"], last["rsi"]
    return "HOLD", last["close"], last["rsi"]

# ══════════════════════════════════════════════════════════════
#  CRYPTO PAPER TRADING
# ══════════════════════════════════════════════════════════════

def crypto_buy(symbol, price):
    p = crypto_paper[symbol]
    if p["in_trade"]: return
    spend    = p["balance"] * RISK_PER_TRADE
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
           f"Balance: ${p['balance']:,.2f}\n"
           f"SL: ${price*(1-STOP_LOSS_PCT):,.2f}  |  TP: ${price*(1+TAKE_PROFIT_PCT):,.2f}")
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
    if pnl >= 0:
        p["wins"] += 1; emoji = "✅"; result = f"WIN  +${pnl:.2f}"
    else:
        p["losses"] += 1; emoji = "❌"; result = f"LOSS -${abs(pnl):.2f}"
    win_rate = (p["wins"] / p["total_trades"] * 100) if p["total_trades"] > 0 else 0
    coin = symbol.split("/")[0]
    msg = (f"{emoji} <b>SELL {coin}</b>  ({reason})\n"
           f"Price:    ${price:,.2f}\n"
           f"Result:   {result}\n"
           f"Balance:  ${p['balance']:,.2f}\n"
           f"Win rate: {win_rate:.1f}% ({p['wins']}W/{p['losses']}L)")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)

def check_crypto_exits(symbol, price):
    p = crypto_paper[symbol]
    if not p["in_trade"]: return
    if price > p["highest_price"]: p["highest_price"] = price
    if price <= p["highest_price"] * (1 - STOP_LOSS_PCT):
        crypto_sell(symbol, price, reason="Trailing stop")
    elif price >= p["entry_price"] * (1 + TAKE_PROFIT_PCT):
        crypto_sell(symbol, price, reason="Take profit")

def run_crypto():
    coin_names = ", ".join(s.split("/")[0] for s in CRYPTO_SYMBOLS)
    msg = (f"🤖 <b>Crypto bot started</b>\n"
           f"Coins:     {coin_names}\n"
           f"Timeframe: {CRYPTO_TIMEFRAME}  |  HTF: {CRYPTO_HTF}\n"
           f"EMA:       {FAST_EMA}/{SLOW_EMA}  |  RSI: {RSI_PERIOD}\n"
           f"SL: {STOP_LOSS_PCT*100:.0f}%  |  TP: {TAKE_PROFIT_PCT*100:.0f}%\n"
           f"Balance:   ${CRYPTO_BALANCE:,.2f}  (paper)")
    print("=" * 58)
    print(msg.replace("<b>","").replace("</b>",""))
    print("=" * 58)
    send_telegram(msg)

    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] Checking crypto...")
            for symbol in CRYPTO_SYMBOLS:
                coin = symbol.split("/")[0]
                try:
                    df_now = fetch_crypto_candles(symbol, CRYPTO_TIMEFRAME, limit=5)
                    check_crypto_exits(symbol, df_now["close"].iloc[-1])
                    htf    = get_htf_trend_crypto(symbol)
                    df     = fetch_crypto_candles(symbol, CRYPTO_TIMEFRAME, limit=200)
                    df     = add_indicators(df)
                    signal, price, rsi = get_signal(df)
                    status = "IN TRADE" if crypto_paper[symbol]["in_trade"] else "watching"
                    print(f"  {coin:<4} | {signal:<4} | 1h: {htf:<7} | RSI: {rsi:>5.1f} | ${price:>10,.2f} | {status}")
                    if signal == "BUY":
                        if htf == "UP": crypto_buy(symbol, price)
                        else: print(f"        BUY blocked — 1h: {htf}")
                    elif signal == "SELL":
                        crypto_sell(symbol, price)
                    time.sleep(2)
                except Exception as e:
                    print(f"  {coin} error: {e}")
        except Exception as e:
            print(f"Crypto loop error: {e}")
        time.sleep(CRYPTO_INTERVAL)

# ══════════════════════════════════════════════════════════════
#  STOCKS BOT SETTINGS
# ══════════════════════════════════════════════════════════════

STOCK_SYMBOLS    = ["VOO", "QQQ", "SPY"]
STOCK_FAST_EMA   = 10
STOCK_SLOW_EMA   = 50
STOCK_BALANCE    = 10000.0
STOCK_INTERVAL   = 60 * 15

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"

stock_paper = {
    symbol: {
        "balance":       STOCK_BALANCE / len(STOCK_SYMBOLS),
        "shares_held":   0.0,
        "in_trade":      False,
        "entry_price":   0.0,
        "highest_price": 0.0,
        "total_trades":  0,
        "wins":          0,
        "losses":        0,
    }
    for symbol in STOCK_SYMBOLS
}

# ══════════════════════════════════════════════════════════════
#  STOCKS PAPER TRADING
# ══════════════════════════════════════════════════════════════

def stock_buy(symbol, price, alpaca):
    p = stock_paper[symbol]
    if p["in_trade"]: return
    spend  = p["balance"] * RISK_PER_TRADE
    shares = spend / price
    if spend < 1.0: return
    p["balance"]       -= spend
    p["shares_held"]    = shares
    p["in_trade"]       = True
    p["entry_price"]    = price
    p["highest_price"]  = price
    msg = (f"🟢 <b>BUY {symbol}</b>\n"
           f"Price:   ${price:,.2f}\n"
           f"Spent:   ${spend:.2f}\n"
           f"Shares:  {shares:.4f}\n"
           f"Balance: ${p['balance']:,.2f}\n"
           f"SL: ${price*(1-STOP_LOSS_PCT):,.2f}  |  TP: ${price*(1+TAKE_PROFIT_PCT):,.2f}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)

def stock_sell(symbol, price, alpaca, reason="Signal"):
    p = stock_paper[symbol]
    if not p["in_trade"]: return
    proceeds = p["shares_held"] * price
    pnl      = proceeds - (p["shares_held"] * p["entry_price"])
    p["balance"]      += proceeds
    p["total_trades"] += 1
    p["shares_held"]   = 0.0
    p["in_trade"]      = False
    p["highest_price"] = 0.0
    if pnl >= 0:
        p["wins"] += 1; emoji = "✅"; result = f"WIN  +${pnl:.2f}"
    else:
        p["losses"] += 1; emoji = "❌"; result = f"LOSS -${abs(pnl):.2f}"
    win_rate = (p["wins"] / p["total_trades"] * 100) if p["total_trades"] > 0 else 0
    msg = (f"{emoji} <b>SELL {symbol}</b>  ({reason})\n"
           f"Price:    ${price:,.2f}\n"
           f"Result:   {result}\n"
           f"Balance:  ${p['balance']:,.2f}\n"
           f"Win rate: {win_rate:.1f}% ({p['wins']}W/{p['losses']}L)")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)

def run_stocks():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Stocks bot: No Alpaca keys — skipping")
        return
    try:
        alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    except Exception as e:
        print(f"Stocks bot: Alpaca connection failed — {e}")
        return

    msg = (f"📈 <b>Stocks bot started</b>\n"
           f"Stocks:  {', '.join(STOCK_SYMBOLS)}\n"
           f"EMA:     {STOCK_FAST_EMA}/{STOCK_SLOW_EMA}  |  RSI: {RSI_PERIOD}\n"
           f"SL: {STOP_LOSS_PCT*100:.0f}%  |  TP: {TAKE_PROFIT_PCT*100:.0f}%\n"
           f"Balance: ${STOCK_BALANCE:,.2f}  (paper)\n"
           f"Hours:   Mon-Fri 9:30am-4pm ET only")
    print("=" * 58)
    print(msg.replace("<b>","").replace("</b>",""))
    print("=" * 58)
    send_telegram(msg)

    while True:
        try:
            clock = alpaca.get_clock()
            if not clock.is_open:
                now  = pd.Timestamp(clock.timestamp)
                nxt  = pd.Timestamp(clock.next_open)
                mins = int((nxt - now).total_seconds() / 60)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Market closed — opens in ~{mins} min")
                time.sleep(min(60 * 15, mins * 60))
                continue

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] Market open — checking stocks...")

            for symbol in STOCK_SYMBOLS:
                try:
                    # Current price
                    bars_now      = alpaca.get_bars(symbol, TimeFrame.Minute, limit=5).df
                    current_price = bars_now["close"].iloc[-1]

                    # Check exits
                    p = stock_paper[symbol]
                    if p["in_trade"]:
                        if current_price > p["highest_price"]:
                            p["highest_price"] = current_price
                        if current_price <= p["highest_price"] * (1 - STOP_LOSS_PCT):
                            stock_sell(symbol, current_price, alpaca, reason="Trailing stop")
                        elif current_price >= p["entry_price"] * (1 + TAKE_PROFIT_PCT):
                            stock_sell(symbol, current_price, alpaca, reason="Take profit")

                    # Daily trend
                    bars_day = alpaca.get_bars(symbol, TimeFrame.Day, limit=200).df
                    bars_day["ema_fast"] = ta.trend.ema_indicator(bars_day["close"], window=STOCK_FAST_EMA)
                    bars_day["ema_slow"] = ta.trend.ema_indicator(bars_day["close"], window=STOCK_SLOW_EMA)
                    last_day = bars_day.iloc[-1]
                    htf = "UP" if last_day["ema_fast"] > last_day["ema_slow"] else "DOWN"

                    # Hourly signal
                    bars_hr = alpaca.get_bars(symbol, TimeFrame.Hour, limit=500).df
                    bars_hr["ema_fast"] = ta.trend.ema_indicator(bars_hr["close"], window=STOCK_FAST_EMA)
                    bars_hr["ema_slow"] = ta.trend.ema_indicator(bars_hr["close"], window=STOCK_SLOW_EMA)
                    bars_hr["rsi"]      = ta.momentum.rsi(bars_hr["close"], window=RSI_PERIOD)
                    bars_hr["vol_avg"]  = bars_hr["volume"].rolling(window=20).mean()

                    prev = bars_hr.iloc[-2]
                    last = bars_hr.iloc[-1]

                    buy  = (prev["ema_fast"] < prev["ema_slow"] and
                            last["ema_fast"] > last["ema_slow"] and
                            last["rsi"] < RSI_OVERBOUGHT)
                    sell = (prev["ema_fast"] > prev["ema_slow"] and
                            last["ema_fast"] < last["ema_slow"] and
                            last["rsi"] > RSI_OVERSOLD)

                    # Skip if RSI is nan (not enough data yet)
                    if pd.isna(last["rsi"]):
                        print(f"  {symbol:<4} | waiting for enough data...")
                        time.sleep(2)
                        continue

                    signal = "BUY" if buy else "SELL" if sell else "HOLD"
                    price  = last["close"]
                    rsi    = last["rsi"]
                    status = "IN TRADE" if p["in_trade"] else "watching"

                    print(f"  {symbol:<4} | {signal:<4} | "
                          f"Daily: {htf:<7} | "
                          f"RSI: {rsi:>5.1f} | "
                          f"${price:>8,.2f} | {status}")

                    if signal == "BUY":
                        if htf == "UP": stock_buy(symbol, price, alpaca)
                        else: print(f"        BUY blocked — daily: {htf}")
                    elif signal == "SELL":
                        stock_sell(symbol, price, alpaca)

                    time.sleep(2)

                except Exception as e:
                    print(f"  {symbol} error: {e}")

        except Exception as e:
            print(f"Stocks loop error: {e}")

        time.sleep(STOCK_INTERVAL)

# ══════════════════════════════════════════════════════════════
#  MAIN — everything runs in threads
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Start health server first — Railway needs this to stay alive
    threading.Thread(target=start_health_server, daemon=True).start()
    time.sleep(3)

    # 2. Start crypto bot
    threading.Thread(target=run_crypto, daemon=True).start()

    # 3. Start stocks bot 5 seconds later
    time.sleep(5)
    threading.Thread(target=run_stocks, daemon=True).start()

    # 4. Keep main thread alive forever
    print("All bots running. Press Ctrl+C to stop.")
    while True:
        time.sleep(60)
