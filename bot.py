import ccxt
import pandas as pd
import ta
import time
import logging
import requests
import os
from datetime import datetime

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    filename="trades.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# ══════════════════════════════════════════════════════════════
#  SETTINGS — edit these
# ══════════════════════════════════════════════════════════════

SYMBOLS        = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # coins to trade
TIMEFRAME      = "15m"          # entry timeframe
HTF_TIMEFRAME  = "1h"          # higher timeframe trend filter
FAST_EMA       = 5             # fast EMA period
SLOW_EMA       = 50            # slow EMA period
RSI_PERIOD     = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30
STOP_LOSS_PCT  = 0.02          # exit if price drops 2% from entry
TAKE_PROFIT_PCT = 0.04         # exit if price rises 4% from entry
PAPER_BALANCE  = 10000.0       # starting fake USD (shared across all coins)
RISK_PER_TRADE = 0.02          # risk 2% of balance per trade
CHECK_INTERVAL = 60 * 15       # check every 1 hour

# Telegram — fill these in (see setup guide below)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ══════════════════════════════════════════════════════════════
#  PAPER TRADING STATE — one entry per coin
# ══════════════════════════════════════════════════════════════

paper = {
    symbol: {
        "balance":      PAPER_BALANCE / len(SYMBOLS),  # split balance across coins
        "coin_held":    0.0,
        "in_trade":     False,
        "entry_price":  0.0,
        "highest_price": 0.0,
        "total_trades": 0,
        "wins":         0,
        "losses":       0,
    }
    for symbol in SYMBOLS
}

# ══════════════════════════════════════════════════════════════
#  EXCHANGE
# ══════════════════════════════════════════════════════════════

exchange = ccxt.kucoin()   # works in the US, no API key needed for price data

# ══════════════════════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════════════════════

def send_telegram(msg):
    """Send a message to your Telegram bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return   # silently skip if not configured
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# ══════════════════════════════════════════════════════════════
#  DATA & INDICATORS
# ══════════════════════════════════════════════════════════════

def fetch_candles(symbol, timeframe, limit=100):
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

def get_htf_trend(symbol):
    """Check 4h chart — returns UP, DOWN, or NEUTRAL."""
    try:
        df = fetch_candles(symbol, HTF_TIMEFRAME, limit=60)
        df = add_indicators(df)
        last = df.iloc[-1]
        if last["ema_fast"] > last["ema_slow"]:
            return "UP"
        elif last["ema_fast"] < last["ema_slow"]:
            return "DOWN"
        return "NEUTRAL"
    except Exception as e:
        logging.error(f"HTF trend error ({symbol}): {e}")
        return "NEUTRAL"

def get_signal(df):
    prev = df.iloc[-2]
    last = df.iloc[-1]

    buy = (
        prev["ema_fast"] < prev["ema_slow"] and
        last["ema_fast"] > last["ema_slow"] and
        last["rsi"] < RSI_OVERBOUGHT and
        last["volume"] > last["vol_avg"] * 0.8    # volume filter
    )

    sell = (
        prev["ema_fast"] > prev["ema_slow"] and
        last["ema_fast"] < last["ema_slow"] and
        last["rsi"] > RSI_OVERSOLD
    )

    if buy:  return "BUY",  last["close"], last["rsi"]
    if sell: return "SELL", last["close"], last["rsi"]
    return "HOLD", last["close"], last["rsi"]

# ══════════════════════════════════════════════════════════════
#  PAPER TRADING EXECUTION
# ══════════════════════════════════════════════════════════════

def paper_buy(symbol, price):
    p = paper[symbol]
    if p["in_trade"]:
        return
    spend    = p["balance"] * RISK_PER_TRADE
    coin_qty = spend / price
    if spend < 1.0:
        return

    p["balance"]       -= spend
    p["coin_held"]      = coin_qty
    p["in_trade"]       = True
    p["entry_price"]    = price
    p["highest_price"]  = price

    coin = symbol.split("/")[0]
    msg = (f"🟢 <b>BUY {coin}</b>\n"
           f"Price: ${price:,.2f}\n"
           f"Spent: ${spend:.2f}\n"
           f"Amount: {coin_qty:.6f} {coin}\n"
           f"Balance: ${p['balance']:,.2f}\n"
           f"SL: ${price*(1-STOP_LOSS_PCT):,.2f} | TP: ${price*(1+TAKE_PROFIT_PCT):,.2f}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    logging.info(msg.replace("<b>","").replace("</b>",""))
    send_telegram(msg)

def paper_sell(symbol, price, reason="Signal"):
    p = paper[symbol]
    if not p["in_trade"]:
        return

    proceeds = p["coin_held"] * price
    pnl      = proceeds - (p["coin_held"] * p["entry_price"])
    p["balance"]  += proceeds
    p["total_trades"] += 1
    p["coin_held"]  = 0.0
    p["in_trade"]   = False
    p["highest_price"] = 0.0

    if pnl >= 0:
        p["wins"] += 1
        emoji  = "✅"
        result = f"WIN  +${pnl:.2f}"
    else:
        p["losses"] += 1
        emoji  = "❌"
        result = f"LOSS -${abs(pnl):.2f}"

    win_rate = (p["wins"] / p["total_trades"] * 100) if p["total_trades"] > 0 else 0
    coin = symbol.split("/")[0]

    msg = (f"{emoji} <b>SELL {coin}</b> ({reason})\n"
           f"Price: ${price:,.2f}\n"
           f"Result: {result}\n"
           f"Balance: ${p['balance']:,.2f}\n"
           f"Win rate: {win_rate:.1f}% ({p['wins']}W / {p['losses']}L)")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    logging.info(msg.replace("<b>","").replace("</b>",""))
    send_telegram(msg)

# ══════════════════════════════════════════════════════════════
#  STOP-LOSS / TAKE-PROFIT / TRAILING STOP CHECK
# ══════════════════════════════════════════════════════════════

def check_exit_conditions(symbol, price):
    """Check stop-loss and take-profit on every tick."""
    p = paper[symbol]
    if not p["in_trade"]:
        return

    # Update trailing high
    if price > p["highest_price"]:
        p["highest_price"] = price

    # Trailing stop — exits if price drops 2% from the highest point reached
    trail_stop = p["highest_price"] * (1 - STOP_LOSS_PCT)

    # Take profit
    take_profit = p["entry_price"] * (1 + TAKE_PROFIT_PCT)

    if price <= trail_stop:
        paper_sell(symbol, price, reason="Trailing stop")
    elif price >= take_profit:
        paper_sell(symbol, price, reason="Take profit")

# ══════════════════════════════════════════════════════════════
#  STATUS PRINT
# ══════════════════════════════════════════════════════════════

def print_portfolio_summary():
    print("\n" + "─" * 55)
    total = 0
    for symbol in SYMBOLS:
        p = paper[symbol]
        coin = symbol.split("/")[0]
        status = "IN TRADE" if p["in_trade"] else "watching"
        total += p["balance"]
        print(f"  {coin:<4} | {status:<10} | "
              f"Balance: ${p['balance']:>9,.2f} | "
              f"Trades: {p['total_trades']} | "
              f"W/L: {p['wins']}/{p['losses']}")
    print(f"  {'TOTAL':<4} | {'':10} | Balance: ${total:>9,.2f}")
    print("─" * 55)

# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run():
    startup_msg = (
        f"🤖 <b>Bot started</b>\n"
        f"Coins: {', '.join(s.split('/')[0] for s in SYMBOLS)}\n"
        f"Timeframe: {TIMEFRAME} | HTF: {HTF_TIMEFRAME}\n"
        f"EMA: {FAST_EMA}/{SLOW_EMA} | RSI: {RSI_PERIOD}\n"
        f"SL: {STOP_LOSS_PCT*100:.0f}% | TP: {TAKE_PROFIT_PCT*100:.0f}%\n"
        f"Balance: ${PAPER_BALANCE:,.2f} (paper)"
    )
    print("=" * 55)
    print(startup_msg.replace("<b>","").replace("</b>",""))
    print("=" * 55)
    send_telegram(startup_msg)

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}] Checking {len(SYMBOLS)} coins...")

        for symbol in SYMBOLS:
            coin = symbol.split("/")[0]
            try:
                # Always check stop-loss / take-profit first
                df_now = fetch_candles(symbol, TIMEFRAME, limit=5)
                current_price = df_now["close"].iloc[-1]
                check_exit_conditions(symbol, current_price)

                # Get higher timeframe trend
                htf = get_htf_trend(symbol)

                # Get 1h signal
                df = fetch_candles(symbol, TIMEFRAME, limit=100)
                df = add_indicators(df)
                signal, price, rsi = get_signal(df)

                status = "IN TRADE" if paper[symbol]["in_trade"] else "watching"
                print(f"  {coin:<4} | {signal:<4} | "
                      f"4h: {htf:<7} | "
                      f"RSI: {rsi:>5.1f} | "
                      f"${price:>10,.2f} | {status}")

                if signal == "BUY":
                    if htf == "UP":
                        paper_buy(symbol, price)
                    else:
                        print(f"        BUY blocked — 4h trend is {htf}")

                elif signal == "SELL":
                    paper_sell(symbol, price)

                time.sleep(2)   # small pause between coins — avoids rate limits

            except Exception as e:
                print(f"  {coin} error: {e}")
                logging.error(f"{coin} error: {e}")

        print_portfolio_summary()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run()