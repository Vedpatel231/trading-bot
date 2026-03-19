import pandas as pd
import ta
import time
import logging
import requests
import os
from datetime import datetime
from alpaca_trade_api.rest import REST, TimeFrame

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    filename="trades_stocks.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# ══════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════

SYMBOLS         = ["VOO", "QQQ", "SPY"]  # stocks to trade (remove any you don't want)
FAST_EMA        = 10         # fast EMA — wider than crypto since stocks move slower
SLOW_EMA        = 50         # slow EMA
RSI_PERIOD      = 14
RSI_OVERBOUGHT  = 70
RSI_OVERSOLD    = 30
STOP_LOSS_PCT   = 0.02       # exit if price drops 2% from highest point
TAKE_PROFIT_PCT = 0.04       # exit if price rises 4% from entry
PAPER_BALANCE   = 10000.0    # starting fake USD
RISK_PER_TRADE  = 0.02       # risk 2% per trade
CHECK_INTERVAL  = 60 * 15    # check every 15 minutes during market hours

# ── Alpaca — set these in Railway environment variables ────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"  # paper trading URL

# ── Telegram — same as crypto bot ─────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ══════════════════════════════════════════════════════════════
#  ALPACA CONNECTION
# ══════════════════════════════════════════════════════════════

alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ══════════════════════════════════════════════════════════════
#  PAPER TRADING STATE
# ══════════════════════════════════════════════════════════════

paper = {
    symbol: {
        "balance":       PAPER_BALANCE / len(SYMBOLS),
        "shares_held":   0.0,
        "in_trade":      False,
        "entry_price":   0.0,
        "highest_price": 0.0,
        "total_trades":  0,
        "wins":          0,
        "losses":        0,
    }
    for symbol in SYMBOLS
}

# ══════════════════════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════════════════════

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
#  MARKET HOURS CHECK
# ══════════════════════════════════════════════════════════════

def is_market_open():
    """Returns True if US stock market is currently open."""
    try:
        clock = alpaca.get_clock()
        return clock.is_open
    except Exception as e:
        logging.error(f"Market clock error: {e}")
        return False

def time_until_open():
    """Returns minutes until market opens."""
    try:
        clock = alpaca.get_clock()
        now       = pd.Timestamp(clock.timestamp)
        next_open = pd.Timestamp(clock.next_open)
        diff      = (next_open - now).total_seconds() / 60
        return int(diff)
    except Exception:
        return 60  # default wait 60 minutes if error

# ══════════════════════════════════════════════════════════════
#  DATA & INDICATORS
# ══════════════════════════════════════════════════════════════

def fetch_candles(symbol, timeframe=TimeFrame.Hour, limit=100):
    """Fetch OHLCV bars from Alpaca."""
    bars = alpaca.get_bars(symbol, timeframe, limit=limit).df
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars

def add_indicators(df):
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["vol_avg"]  = df["volume"].rolling(window=20).mean()
    return df

def get_htf_trend(symbol):
    """Check daily chart for overall trend. Returns UP, DOWN, or NEUTRAL."""
    try:
        df   = fetch_candles(symbol, TimeFrame.Day, limit=60)
        df   = add_indicators(df)
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
    """Return BUY, SELL, or HOLD based on EMA crossover + RSI + volume."""
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
#  PAPER TRADING EXECUTION
# ══════════════════════════════════════════════════════════════

def paper_buy(symbol, price):
    p = paper[symbol]
    if p["in_trade"]:
        return

    spend      = p["balance"] * RISK_PER_TRADE
    shares_qty = spend / price

    if spend < 1.0:
        return

    p["balance"]        -= spend
    p["shares_held"]     = shares_qty
    p["in_trade"]        = True
    p["entry_price"]     = price
    p["highest_price"]   = price

    sl = price * (1 - STOP_LOSS_PCT)
    tp = price * (1 + TAKE_PROFIT_PCT)

    msg = (
        f"🟢 <b>BUY {symbol}</b>\n"
        f"Price:   ${price:,.2f}\n"
        f"Spent:   ${spend:.2f}\n"
        f"Shares:  {shares_qty:.4f}\n"
        f"Balance: ${p['balance']:,.2f}\n"
        f"SL: ${sl:,.2f}  |  TP: ${tp:,.2f}"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"{msg.replace('<b>','').replace('</b>','')}")
    logging.info(msg.replace("<b>", "").replace("</b>", ""))
    send_telegram(msg)

def paper_sell(symbol, price, reason="Signal"):
    p = paper[symbol]
    if not p["in_trade"]:
        return

    proceeds = p["shares_held"] * price
    pnl      = proceeds - (p["shares_held"] * p["entry_price"])

    p["balance"]      += proceeds
    p["total_trades"] += 1
    p["shares_held"]   = 0.0
    p["in_trade"]      = False
    p["highest_price"] = 0.0

    if pnl >= 0:
        p["wins"] += 1
        emoji  = "✅"
        result = f"WIN  +${pnl:.2f}"
    else:
        p["losses"] += 1
        emoji  = "❌"
        result = f"LOSS -${abs(pnl):.2f}"

    win_rate = (
        p["wins"] / p["total_trades"] * 100
        if p["total_trades"] > 0 else 0
    )

    msg = (
        f"{emoji} <b>SELL {symbol}</b>  ({reason})\n"
        f"Price:    ${price:,.2f}\n"
        f"Result:   {result}\n"
        f"Balance:  ${p['balance']:,.2f}\n"
        f"Win rate: {win_rate:.1f}%  "
        f"({p['wins']}W / {p['losses']}L)"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"{msg.replace('<b>','').replace('</b>','')}")
    logging.info(msg.replace("<b>", "").replace("</b>", ""))
    send_telegram(msg)

# ══════════════════════════════════════════════════════════════
#  STOP-LOSS / TAKE-PROFIT CHECK
# ══════════════════════════════════════════════════════════════

def check_exit_conditions(symbol, price):
    p = paper[symbol]
    if not p["in_trade"]:
        return

    if price > p["highest_price"]:
        p["highest_price"] = price

    trail_stop  = p["highest_price"] * (1 - STOP_LOSS_PCT)
    take_profit = p["entry_price"]   * (1 + TAKE_PROFIT_PCT)

    if price <= trail_stop:
        paper_sell(symbol, price, reason="Trailing stop")
    elif price >= take_profit:
        paper_sell(symbol, price, reason="Take profit")

# ══════════════════════════════════════════════════════════════
#  STATUS PRINT
# ══════════════════════════════════════════════════════════════

def print_portfolio_summary():
    print("\n" + "─" * 58)
    total = 0
    for symbol in SYMBOLS:
        p      = paper[symbol]
        status = "IN TRADE" if p["in_trade"] else "watching"
        total += p["balance"]
        print(
            f"  {symbol:<4} | {status:<10} | "
            f"Balance: ${p['balance']:>9,.2f} | "
            f"Trades: {p['total_trades']} | "
            f"W/L: {p['wins']}/{p['losses']}"
        )
    print(f"  {'TOTAL':<4} | {'':10} | Balance: ${total:>9,.2f}")
    print("─" * 58)

# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run():
    startup_msg = (
        f"📈 <b>Stocks bot started</b>\n"
        f"Stocks:    {', '.join(SYMBOLS)}\n"
        f"EMA:       {FAST_EMA}/{SLOW_EMA}  |  RSI: {RSI_PERIOD}\n"
        f"SL: {STOP_LOSS_PCT*100:.0f}%  |  TP: {TAKE_PROFIT_PCT*100:.0f}%\n"
        f"Balance:   ${PAPER_BALANCE:,.2f}  (paper mode)\n"
        f"Hours:     Mon-Fri 9:30am-4pm ET only"
    )

    print("=" * 58)
    print(startup_msg.replace("<b>", "").replace("</b>", ""))
    print("=" * 58)
    send_telegram(startup_msg)

    while True:
        # ── Wait if market is closed ───────────────────────
        if not is_market_open():
            mins = time_until_open()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Market closed — opens in ~{mins} min. Sleeping...")
            # Sleep in 15 min chunks so we catch the open promptly
            time.sleep(min(60 * 15, mins * 60))
            continue

        # ── Market is open — check all stocks ─────────────
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}] Market open — checking {len(SYMBOLS)} stocks...")

        for symbol in SYMBOLS:
            try:
                # 1. Check stop-loss / take-profit
                df_now        = fetch_candles(symbol, TimeFrame.Minute, limit=5)
                current_price = df_now["close"].iloc[-1]
                check_exit_conditions(symbol, current_price)

                # 2. Get daily trend
                htf = get_htf_trend(symbol)

                # 3. Get hourly signal
                df             = fetch_candles(symbol, TimeFrame.Hour, limit=100)
                df             = add_indicators(df)
                signal, price, rsi = get_signal(df)

                status = "IN TRADE" if paper[symbol]["in_trade"] else "watching"
                print(
                    f"  {symbol:<4} | {signal:<4} | "
                    f"Daily: {htf:<7} | "
                    f"RSI: {rsi:>5.1f} | "
                    f"${price:>8,.2f} | {status}"
                )

                # 4. Act on signal — only buy if daily trend is UP
                if signal == "BUY":
                    if htf == "UP":
                        paper_buy(symbol, price)
                    else:
                        print(f"        BUY blocked — daily trend is {htf}")

                elif signal == "SELL":
                    paper_sell(symbol, price)

                time.sleep(2)

            except Exception as e:
                print(f"  {symbol} error: {e}")
                logging.error(f"{symbol} error: {e}")

        print_portfolio_summary()
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run()
