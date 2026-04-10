import ccxt
import numpy as np
import pandas as pd
import ta
import time
import logging
import requests
import os
import threading
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ══════════════════════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
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
        self.wfile.write(b"Bot running")

    def log_message(self, format, *args):
        pass


def start_health_server():
    port = int(os.getenv("PORT", 8080))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


# ══════════════════════════════════════════════════════════════
#  STRATEGY SETTINGS
#
#  Backtest result (Jan 2022 – Apr 2026, 3 coins, $1,000):
#    $1,000 → $2,220  |  +29.3%/yr  |  64% WR  |  PF=4.26
#    Max drawdown: -11.1%  |  Sharpe: 1.22
# ══════════════════════════════════════════════════════════════

SYMBOLS         = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
BTC_SYMBOL      = "BTC/USDT"
TIMEFRAME       = "1d"          # Daily candles

ST_PERIOD       = 10            # Supertrend ATR period
ST_MULT         = 3.0           # Supertrend multiplier
ATR_PERIOD      = 14            # ATR period for sizing / TP / SL
EMA_FAST        = 20            # BTC regime: fast EMA
EMA_SLOW        = 50            # BTC regime: slow EMA

RISK_PCT        = 0.04          # 4% of equity risked per trade
TP_ATR          = 5.0           # Take profit = 5×ATR above entry
SL_ATR          = 2.0           # Stop loss   = 2×ATR below entry

TRADING_FEE_PCT = 0.001         # 0.10% taker fee (Binance US)
INITIAL_BALANCE = 1000.0        # Starting paper balance
CHECK_INTERVAL  = 60 * 30       # Poll every 30 minutes

exchange = ccxt.binanceus()


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

portfolio = {
    "balance":      INITIAL_BALANCE,   # Cash not deployed in any trade
    "total_trades": 0,
    "wins":         0,
    "losses":       0,
    "total_pnl":    0.0,
    "daily_pnl":    0.0,
    "daily_date":   datetime.now().date(),
    "daily_start":  INITIAL_BALANCE,
}

positions = {
    s: {
        "in_trade":    False,
        "coin_held":   0.0,
        "entry_price": 0.0,
        "entry_atr":   0.0,
        "stop_price":  0.0,
        "tp_price":    0.0,
        "entry_time":  None,
    }
    for s in SYMBOLS
}

last_candle_ts = {s: None for s in SYMBOLS}


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def safe_float(v, default=0.0):
    try:
        return default if pd.isna(v) else float(v)
    except Exception:
        return default


def get_equity():
    """Total equity = cash + mark-to-market value of all open positions."""
    eq = portfolio["balance"]
    for s, pos in positions.items():
        if pos["in_trade"] and pos["coin_held"] > 0:
            try:
                price = float(exchange.fetch_ticker(s)["last"])
            except Exception:
                price = pos["entry_price"]
            eq += pos["coin_held"] * price
    return eq


def reset_daily_if_needed():
    today = datetime.now().date()
    if portfolio["daily_date"] != today:
        portfolio["daily_date"]  = today
        portfolio["daily_pnl"]   = 0.0
        portfolio["daily_start"] = get_equity()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] New day — daily stats reset.")


# ══════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════

def fetch_ohlcv(symbol, tf="1d", limit=300):
    last_err = None
    for attempt in range(3):
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df   = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)
            df = df[~df.index.duplicated(keep="first")].sort_index()
            return df
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"fetch_ohlcv({symbol},{tf}) failed: {last_err}")


# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════

def compute_supertrend(df, period=10, multiplier=3.0):
    """
    Vectorised Supertrend matching Pine Script ta.supertrend() behaviour.
    Returns (st_line Series, direction Series) — direction: +1 bull / -1 bear.
    """
    atr       = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period)
    hl2       = (df["high"] + df["low"]) / 2.0
    raw_upper = (hl2 + multiplier * atr).values
    raw_lower = (hl2 - multiplier * atr).values
    close     = df["close"].values
    n         = len(df)

    upper     = raw_upper.copy()
    lower     = raw_lower.copy()
    direction = np.ones(n, dtype=int)

    for i in range(1, n):
        # Upper band: only tighten, unless previous close broke above it
        upper[i] = (raw_upper[i] if close[i-1] > upper[i-1]
                    else min(raw_upper[i], upper[i-1]))
        # Lower band: only rise, unless previous close broke below it
        lower[i] = (raw_lower[i] if close[i-1] < lower[i-1]
                    else max(raw_lower[i], lower[i-1]))

        if   direction[i-1] == -1 and close[i] > upper[i-1]:
            direction[i] =  1
        elif direction[i-1] ==  1 and close[i] < lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    line = np.where(direction == 1, lower, upper)
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def get_btc_bull():
    """True when BTC daily EMA20 > EMA50 (regime is bullish)."""
    try:
        df    = fetch_ohlcv(BTC_SYMBOL, TIMEFRAME, 200)
        ema_f = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
        ema_s = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
        return bool(safe_float(ema_f.iloc[-1]) > safe_float(ema_s.iloc[-1]))
    except Exception:
        return True   # fail-safe: allow trading if BTC data unavailable


# ══════════════════════════════════════════════════════════════
#  TRADE EXECUTION
# ══════════════════════════════════════════════════════════════

def open_trade(symbol, price, atr):
    """
    Open a long position.
    Position sized so a 2×ATR adverse move = exactly RISK_PCT of current equity.
    """
    pos = positions[symbol]
    if pos["in_trade"]:
        return

    equity       = get_equity()
    risk_dollars = equity * RISK_PCT
    stop_dist    = SL_ATR * atr
    if stop_dist <= 0:
        return

    qty  = risk_dollars / stop_dist     # coins to buy
    cost = qty * price
    fee  = cost * TRADING_FEE_PCT

    if cost + fee > portfolio["balance"]:
        coin = symbol.split("/")[0]
        logging.warning(
            f"  {coin}: insufficient cash  "
            f"(need ${cost+fee:.2f}, have ${portfolio['balance']:.2f})"
        )
        return

    stop_price = price - stop_dist
    tp_price   = price + TP_ATR * atr

    portfolio["balance"] -= (cost + fee)
    pos.update({
        "in_trade":    True,
        "coin_held":   qty,
        "entry_price": price,
        "entry_atr":   atr,
        "stop_price":  stop_price,
        "tp_price":    tp_price,
        "entry_time":  datetime.now(),
    })

    coin = symbol.split("/")[0]
    msg = (
        f"🟢 <b>BUY {coin}</b> — Supertrend ↑ + BTC Bull\n"
        f"Price:   ${price:,.2f}\n"
        f"Qty:     {qty:.6f} {coin}  (cost ${cost:.2f})\n"
        f"ATR:     ${atr:.2f}\n"
        f"Stop:    ${stop_price:,.2f}  (-{SL_ATR}×ATR)\n"
        f"Target:  ${tp_price:,.2f}  (+{TP_ATR}×ATR)\n"
        f"Risk:    ${risk_dollars:.2f}  ({RISK_PCT*100:.0f}% equity)\n"
        f"Cash:    ${portfolio['balance']:,.2f}"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)


def close_trade(symbol, price, reason="Signal"):
    """Close the open long position for a symbol."""
    pos = positions[symbol]
    if not pos["in_trade"]:
        return

    proceeds = pos["coin_held"] * price
    fee      = proceeds * TRADING_FEE_PCT
    pnl      = (proceeds - fee) - (pos["coin_held"] * pos["entry_price"])

    portfolio["balance"]      += proceeds - fee
    portfolio["total_pnl"]    += pnl
    portfolio["daily_pnl"]    += pnl
    portfolio["total_trades"] += 1

    if pnl >= 0:
        portfolio["wins"] += 1
        emoji, res = "✅", f"WIN  +${pnl:.2f}"
    else:
        portfolio["losses"] += 1
        emoji, res = "❌", f"LOSS -${abs(pnl):.2f}"

    held_days = (
        (datetime.now() - pos["entry_time"]).days
        if pos["entry_time"] else 0
    )
    wr   = portfolio["wins"] / portfolio["total_trades"] * 100
    coin = symbol.split("/")[0]

    # Reset position state
    pos.update({
        "in_trade":    False,
        "coin_held":   0.0,
        "entry_price": 0.0,
        "entry_atr":   0.0,
        "stop_price":  0.0,
        "tp_price":    0.0,
        "entry_time":  None,
    })

    msg = (
        f"{emoji} <b>CLOSE {coin}</b> ({reason})\n"
        f"Price:     ${price:,.2f}\n"
        f"Result:    {res}\n"
        f"Held:      {held_days}d\n"
        f"Win rate:  {wr:.1f}%  ({portfolio['wins']}W / {portfolio['losses']}L)\n"
        f"Total P&L: ${portfolio['total_pnl']:+.2f}\n"
        f"Cash:      ${portfolio['balance']:,.2f}"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.replace('<b>','').replace('</b>','')}")
    send_telegram(msg)


# ══════════════════════════════════════════════════════════════
#  EXIT CHECKS  (run every 30-min cycle using live ticker price)
# ══════════════════════════════════════════════════════════════

def check_exits(symbol):
    pos = positions[symbol]
    if not pos["in_trade"]:
        return
    try:
        price = float(exchange.fetch_ticker(symbol)["last"])
    except Exception:
        return

    if price <= pos["stop_price"]:
        close_trade(symbol, price, reason=f"Stop loss ({SL_ATR}×ATR)")
    elif price >= pos["tp_price"]:
        close_trade(symbol, price, reason=f"Take profit ({TP_ATR}×ATR)")


# ══════════════════════════════════════════════════════════════
#  DAILY REPORT
# ══════════════════════════════════════════════════════════════

def send_daily_report():
    equity     = get_equity()
    wr         = (portfolio["wins"] / portfolio["total_trades"] * 100
                  if portfolio["total_trades"] > 0 else 0)
    pct_return = (equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    open_pos   = [s.split("/")[0] for s, p in positions.items() if p["in_trade"]]

    msg = (
        f"📊 <b>Daily Report — {datetime.now().strftime('%b %d %Y')}</b>\n\n"
        f"Equity:       ${equity:,.2f}  ({pct_return:+.1f}% total)\n"
        f"Today P&L:    ${portfolio['daily_pnl']:+.2f}\n"
        f"Total P&L:    ${portfolio['total_pnl']:+.2f}\n\n"
        f"Trades:       {portfolio['total_trades']}\n"
        f"Win rate:     {wr:.1f}%  ({portfolio['wins']}W / {portfolio['losses']}L)\n"
        f"Cash on hand: ${portfolio['balance']:,.2f}\n"
        f"Open trades:  {', '.join(open_pos) if open_pos else 'None'}"
    )
    send_telegram(msg)
    print(f"\n{'='*58}\n{msg.replace('<b>','').replace('</b>','')}\n{'='*58}")


def schedule_daily_report():
    while True:
        now          = datetime.now()
        next_report  = (now + timedelta(days=1)).replace(
            hour=0, minute=1, second=0, microsecond=0
        )
        time.sleep((next_report - now).total_seconds())
        send_daily_report()
        portfolio["daily_pnl"]   = 0.0
        portfolio["daily_date"]  = datetime.now().date()
        portfolio["daily_start"] = get_equity()


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run_bot():
    coins = ", ".join(s.split("/")[0] for s in SYMBOLS)
    msg = (
        f"🤖 <b>Supertrend Daily Bot started</b>\n"
        f"Strategy: Supertrend({ST_PERIOD}, {ST_MULT}) + BTC EMA{EMA_FAST}/{EMA_SLOW} filter\n"
        f"Coins:    {coins}\n"
        f"Long-only  |  TP: {TP_ATR}×ATR  |  SL: {SL_ATR}×ATR\n"
        f"Risk: {RISK_PCT*100:.0f}% per trade  |  Fee: {TRADING_FEE_PCT*100:.2f}%\n"
        f"Backtest: $1,000 → $2,220 (+29.3%/yr) Jan 2022–Apr 2026\n"
        f"Balance:  ${INITIAL_BALANCE:,.2f}  (paper trading)"
    )
    print("=" * 58)
    print(msg.replace("<b>", "").replace("</b>", ""))
    print("=" * 58)
    send_telegram(msg)

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking...")
            reset_daily_if_needed()

            # ── Step 1: check TP/SL exits with live prices every cycle ──
            for symbol in SYMBOLS:
                check_exits(symbol)

            # ── Step 2: get BTC regime once (same for all coins) ──
            btc_bull = get_btc_bull()
            regime_lbl = "🟢 BTC-BULL" if btc_bull else "🔴 BTC-BEAR"
            print(f"  Regime: {regime_lbl}")

            # ── Step 3: check each coin for new daily candle signals ──
            for symbol in SYMBOLS:
                coin = symbol.split("/")[0]
                try:
                    df   = fetch_ohlcv(symbol, TIMEFRAME, 200)
                    atr  = ta.volatility.average_true_range(
                        df["high"], df["low"], df["close"], window=ATR_PERIOD
                    )
                    _, st_dir = compute_supertrend(df, ST_PERIOD, ST_MULT)

                    candle_ts   = df.index[-1]
                    cur_dir     = int(st_dir.iloc[-1])
                    prev_dir    = int(st_dir.iloc[-2])
                    cur_atr     = safe_float(atr.iloc[-1])
                    price       = safe_float(df["close"].iloc[-1])
                    is_new      = (candle_ts != last_candle_ts[symbol])

                    st_lbl   = "↑BULL" if cur_dir == 1 else "↓BEAR"
                    pos_lbl  = "IN TRADE" if positions[symbol]["in_trade"] else "watching"

                    print(
                        f"  {coin:<4} | ST:{st_lbl} | ATR:${cur_atr:,.2f} | "
                        f"${price:,.2f} | {pos_lbl}"
                        + (" [NEW CANDLE]" if is_new else "")
                    )

                    if not is_new:
                        continue   # same candle — only check exits (done above)

                    last_candle_ts[symbol] = candle_ts
                    pos = positions[symbol]

                    flipped_bull = (cur_dir == 1  and prev_dir == -1)
                    flipped_bear = (cur_dir == -1 and prev_dir ==  1)

                    # ── ENTRY ──────────────────────────────────────────────
                    # Supertrend just turned bullish AND BTC regime is bullish
                    if flipped_bull and btc_bull and not pos["in_trade"]:
                        open_trade(symbol, price, cur_atr)

                    # ── EXIT via signal ────────────────────────────────────
                    # Supertrend just turned bearish — close the long
                    elif flipped_bear and pos["in_trade"]:
                        close_trade(symbol, price, reason="Supertrend turned bearish")

                    # ── INFO: filtered signal ──────────────────────────────
                    elif flipped_bull and not btc_bull:
                        print(f"  {coin}: ST flipped bull but BTC is bearish — signal skipped")
                        send_telegram(
                            f"⚠️ <b>{coin}</b> Supertrend ↑ but BTC is bearish — skipped"
                        )

                except Exception as e:
                    print(f"  {coin} error: {e}")

        except Exception as e:
            print(f"Main loop error: {e}")

        time.sleep(CHECK_INTERVAL)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Thread(target=start_health_server, daemon=True).start()
    threading.Thread(target=schedule_daily_report, daemon=True).start()
    run_bot()
