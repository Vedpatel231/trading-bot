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
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"Health server running on port {port}")
    server.serve_forever()
 
# ══════════════════════════════════════════════════════════════
#  CRYPTO SETTINGS
# ══════════════════════════════════════════════════════════════
 
CRYPTO_SYMBOLS   = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CRYPTO_TF        = "1m"
CRYPTO_HTF       = "5m"
FAST_EMA         = 5
SLOW_EMA         = 50
RSI_PERIOD       = 14
RSI_OB           = 70
RSI_OS           = 30
SL_PCT           = 0.02
TP_PCT           = 0.04
CRYPTO_BAL       = 10000.0
RISK             = 0.02
CRYPTO_SLEEP     = 60
 
exchange = ccxt.binanceus()
 
crypto_paper = {
    s: {"balance": CRYPTO_BAL/len(CRYPTO_SYMBOLS), "coin_held": 0.0,
        "in_trade": False, "entry_price": 0.0, "highest_price": 0.0,
        "total_trades": 0, "wins": 0, "losses": 0}
    for s in CRYPTO_SYMBOLS
}
 
# ══════════════════════════════════════════════════════════════
#  CRYPTO FUNCTIONS
# ══════════════════════════════════════════════════════════════
 
def fetch_crypto(symbol, tf, limit=200):
    bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df
 
def add_ind(df):
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["vol_avg"]  = df["volume"].rolling(20).mean()
    return df
 
def htf_trend(symbol):
    try:
        df = add_ind(fetch_crypto(symbol, CRYPTO_HTF, 100))
        l  = df.iloc[-1]
        if l["ema_fast"] > l["ema_slow"]: return "UP"
        if l["ema_fast"] < l["ema_slow"]: return "DOWN"
        return "NEUTRAL"
    except:
        return "NEUTRAL"
 
def get_signal(df):
    p, l = df.iloc[-2], df.iloc[-1]
    buy  = (p["ema_fast"] < p["ema_slow"] and l["ema_fast"] > l["ema_slow"]
            and l["rsi"] < RSI_OB and l["volume"] > l["vol_avg"] * 0.8)
    sell = (p["ema_fast"] > p["ema_slow"] and l["ema_fast"] < l["ema_slow"]
            and l["rsi"] > RSI_OS)
    if buy:  return "BUY",  l["close"], l["rsi"]
    if sell: return "SELL", l["close"], l["rsi"]
    return "HOLD", l["close"], l["rsi"]
 
def c_buy(symbol, price):
    p = crypto_paper[symbol]
    if p["in_trade"]: return
    spend = p["balance"] * RISK
    if spend < 1: return
    qty = spend / price
    p["balance"] -= spend; p["coin_held"] = qty; p["in_trade"] = True
    p["entry_price"] = price; p["highest_price"] = price
    coin = symbol.split("/")[0]
    msg = (f"🟢 <b>BUY {coin}</b>\nPrice: ${price:,.2f}\nSpent: ${spend:.2f}\n"
           f"Amount: {qty:.6f} {coin}\nBalance: ${p['balance']:,.2f}\n"
           f"SL: ${price*(1-SL_PCT):,.2f} | TP: ${price*(1+TP_PCT):,.2f}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)
 
def c_sell(symbol, price, reason="Signal"):
    p = crypto_paper[symbol]
    if not p["in_trade"]: return
    proceeds = p["coin_held"] * price
    pnl = proceeds - (p["coin_held"] * p["entry_price"])
    p["balance"] += proceeds; p["total_trades"] += 1
    p["coin_held"] = 0.0; p["in_trade"] = False; p["highest_price"] = 0.0
    if pnl >= 0: p["wins"] += 1;  emoji = "✅"; res = f"WIN  +${pnl:.2f}"
    else:        p["losses"] += 1; emoji = "❌"; res = f"LOSS -${abs(pnl):.2f}"
    wr = (p["wins"]/p["total_trades"]*100) if p["total_trades"] > 0 else 0
    coin = symbol.split("/")[0]
    msg = (f"{emoji} <b>SELL {coin}</b> ({reason})\nPrice: ${price:,.2f}\n"
           f"Result: {res}\nBalance: ${p['balance']:,.2f}\n"
           f"Win rate: {wr:.1f}% ({p['wins']}W/{p['losses']}L)")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)
 
def check_c_exits(symbol, price):
    p = crypto_paper[symbol]
    if not p["in_trade"]: return
    if price > p["highest_price"]: p["highest_price"] = price
    if price <= p["highest_price"] * (1 - SL_PCT): c_sell(symbol, price, "Trailing stop")
    elif price >= p["entry_price"] * (1 + TP_PCT):  c_sell(symbol, price, "Take profit")
 
def run_crypto():
    coins = ", ".join(s.split("/")[0] for s in CRYPTO_SYMBOLS)
    msg = (f"🤖 <b>Crypto bot started</b>\nCoins: {coins}\n"
           f"Timeframe: {CRYPTO_TF} | HTF: {CRYPTO_HTF}\n"
           f"EMA: {FAST_EMA}/{SLOW_EMA} | SL: {SL_PCT*100:.0f}% | TP: {TP_PCT*100:.0f}%\n"
           f"Balance: ${CRYPTO_BAL:,.2f} (paper)")
    print("="*58); print(msg.replace("<b>","").replace("</b>","")); print("="*58)
    send_telegram(msg)
 
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking crypto...")
            for symbol in CRYPTO_SYMBOLS:
                coin = symbol.split("/")[0]
                try:
                    df_now = fetch_crypto(symbol, CRYPTO_TF, 5)
                    check_c_exits(symbol, df_now["close"].iloc[-1])
                    trend  = htf_trend(symbol)
                    df     = add_ind(fetch_crypto(symbol, CRYPTO_TF, 200))
                    sig, price, rsi = get_signal(df)
                    st = "IN TRADE" if crypto_paper[symbol]["in_trade"] else "watching"
                    print(f"  {coin:<4} | {sig:<4} | 5m: {trend:<7} | RSI: {rsi:>5.1f} | ${price:>10,.2f} | {st}")
                    if sig == "BUY":
                        if trend == "UP": c_buy(symbol, price)
                        else: print(f"        BUY blocked — 5m: {trend}")
                    elif sig == "SELL": c_sell(symbol, price)
                    time.sleep(2)
                except Exception as e:
                    print(f"  {coin} error: {e}")
        except Exception as e:
            print(f"Crypto loop error: {e}")
        time.sleep(CRYPTO_SLEEP)
 
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
    s: {"balance": STOCK_BAL/len(STOCK_SYMBOLS), "shares_held": 0.0,
        "in_trade": False, "entry_price": 0.0, "highest_price": 0.0,
        "total_trades": 0, "wins": 0, "losses": 0}
    for s in STOCK_SYMBOLS
}
 
# ══════════════════════════════════════════════════════════════
#  STOCKS FUNCTIONS
# ══════════════════════════════════════════════════════════════
 
def s_buy(symbol, price, alpaca):
    p = stock_paper[symbol]
    if p["in_trade"]: return
    spend = p["balance"] * RISK
    if spend < 1: return
    shares = spend / price
    p["balance"] -= spend; p["shares_held"] = shares; p["in_trade"] = True
    p["entry_price"] = price; p["highest_price"] = price
    msg = (f"🟢 <b>BUY {symbol}</b>\nPrice: ${price:,.2f}\nSpent: ${spend:.2f}\n"
           f"Shares: {shares:.4f}\nBalance: ${p['balance']:,.2f}\n"
           f"SL: ${price*(1-SL_PCT):,.2f} | TP: ${price*(1+TP_PCT):,.2f}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)
 
def s_sell(symbol, price, alpaca, reason="Signal"):
    p = stock_paper[symbol]
    if not p["in_trade"]: return
    proceeds = p["shares_held"] * price
    pnl = proceeds - (p["shares_held"] * p["entry_price"])
    p["balance"] += proceeds; p["total_trades"] += 1
    p["shares_held"] = 0.0; p["in_trade"] = False; p["highest_price"] = 0.0
    if pnl >= 0: p["wins"] += 1;  emoji = "✅"; res = f"WIN  +${pnl:.2f}"
    else:        p["losses"] += 1; emoji = "❌"; res = f"LOSS -${abs(pnl):.2f}"
    wr = (p["wins"]/p["total_trades"]*100) if p["total_trades"] > 0 else 0
    msg = (f"{emoji} <b>SELL {symbol}</b> ({reason})\nPrice: ${price:,.2f}\n"
           f"Result: {res}\nBalance: ${p['balance']:,.2f}\n"
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
 
    msg = (f"📈 <b>Stocks bot started</b>\nStocks: {', '.join(STOCK_SYMBOLS)}\n"
           f"EMA: {S_FAST_EMA}/{S_SLOW_EMA} | SL: {SL_PCT*100:.0f}% | TP: {TP_PCT*100:.0f}%\n"
           f"Balance: ${STOCK_BAL:,.2f} (paper) | Mon-Fri 9:30am-4pm ET")
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
 
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Market open — checking stocks...")
 
            for symbol in STOCK_SYMBOLS:
                try:
                    # Fetch daily bars and reset index to avoid indexing errors
                    raw  = alpaca.get_bars(symbol, TimeFrame.Day, limit=200).df
                    bars = raw.copy().reset_index(drop=True)
 
                    if len(bars) < 60:
                        print(f"  {symbol:<4} | only {len(bars)} bars — need 60+, skipping")
                        continue
 
                    bars["ema_fast"] = ta.trend.ema_indicator(bars["close"], window=S_FAST_EMA)
                    bars["ema_slow"] = ta.trend.ema_indicator(bars["close"], window=S_SLOW_EMA)
                    bars["rsi"]      = ta.momentum.rsi(bars["close"], window=RSI_PERIOD)
 
                    prev  = bars.iloc[-2]
                    last  = bars.iloc[-1]
                    price = float(last["close"])
                    rsi   = float(last["rsi"]) if not pd.isna(last["rsi"]) else None
 
                    if rsi is None:
                        print(f"  {symbol:<4} | RSI not ready — skipping")
                        continue
 
                    trend = "UP" if last["ema_fast"] > last["ema_slow"] else "DOWN"
 
                    # Check exits
                    p = stock_paper[symbol]
                    if p["in_trade"]:
                        if price > p["highest_price"]: p["highest_price"] = price
                        if price <= p["highest_price"] * (1 - SL_PCT):
                            s_sell(symbol, price, alpaca, "Trailing stop")
                        elif price >= p["entry_price"] * (1 + TP_PCT):
                            s_sell(symbol, price, alpaca, "Take profit")
 
                    buy  = (prev["ema_fast"] < prev["ema_slow"] and
                            last["ema_fast"] > last["ema_slow"] and rsi < RSI_OB)
                    sell = (prev["ema_fast"] > prev["ema_slow"] and
                            last["ema_fast"] < last["ema_slow"] and rsi > RSI_OS)
 
                    sig    = "BUY" if buy else "SELL" if sell else "HOLD"
                    status = "IN TRADE" if p["in_trade"] else "watching"
 
                    print(f"  {symbol:<4} | {sig:<4} | Trend: {trend:<7} | "
                          f"RSI: {rsi:>5.1f} | ${price:>8,.2f} | {status}")
 
                    if sig == "BUY":
                        if trend == "UP": s_buy(symbol, price, alpaca)
                        else: print(f"        BUY blocked — trend: {trend}")
                    elif sig == "SELL":
                        s_sell(symbol, price, alpaca)
 
                    time.sleep(2)
 
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
