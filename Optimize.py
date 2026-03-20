"""
optimize.py — automatically finds the best EMA + SL/TP settings.
Tests every combination and ranks them by profit factor.
Usage: python3 optimize.py
Takes about 2-3 minutes to run.
"""

import pandas as pd
import ta
import yfinance as yf
from itertools import product

# ── Symbols to optimize ────────────────────────────────────
SYMBOLS  = ["BTC-USD", "ETH-USD", "SOL-USD"]
INTERVAL = "1h"
PERIOD   = "1y"

# ── Parameter grid to search ───────────────────────────────
FAST_EMAS  = [5, 7, 9, 12]
SLOW_EMAS  = [18, 21, 26, 34, 50]
SL_PCTS    = [0.003, 0.004, 0.005, 0.006]
TP_PCTS    = [0.006, 0.008, 0.010, 0.012, 0.015]

# Fixed settings
RSI_PERIOD = 14
RSI_OB     = 65
RSI_OS     = 35
RISK       = 0.02
START_BAL  = 10000.0

# ── Download data once ─────────────────────────────────────
print("Downloading historical data...")
data = {}
for sym in SYMBOLS:
    raw = yf.download(sym, period=PERIOD, interval=INTERVAL, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    data[sym] = raw.copy()
    print(f"  {sym}: {len(raw)} candles downloaded")

def run_backtest(df, fast, slow, sl_pct, tp_pct):
    """Run a single backtest with given parameters."""
    df = df.copy().dropna(subset=["close","high","low","volume"])

    try:
        df["ema_fast"]  = ta.trend.ema_indicator(df["close"], window=fast)
        df["ema_slow"]  = ta.trend.ema_indicator(df["close"], window=slow)
        df["rsi"]       = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
        macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
        df["macd_hist"] = macd.macd_diff()
        df["vwap"]      = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["vol_avg"]   = df["volume"].rolling(20).mean()
        df["vol_spike"] = df["volume"] > df["vol_avg"] * 1.2
        df = df.dropna()
    except:
        return None

    balance     = START_BAL
    in_trade    = False
    entry_price = 0.0
    highest     = 0.0
    coin_held   = 0.0
    wins = losses = 0
    gross_wins = gross_losses = 0.0

    for i in range(3, len(df)):
        prev2 = df.iloc[i-2]
        prev  = df.iloc[i-1]
        last  = df.iloc[i]
        price = float(last["close"])

        if in_trade:
            if price > highest: highest = price
            if price <= highest*(1-sl_pct) or price >= entry_price*(1+tp_pct):
                pnl     = coin_held * (price - entry_price)
                balance += coin_held * price
                if pnl > 0: wins += 1;  gross_wins   += pnl
                else:       losses += 1; gross_losses += abs(pnl)
                in_trade = False; coin_held = 0.0; highest = 0.0
            continue

        try:
            rsi  = float(last["rsi"])
            mh   = float(last["macd_hist"])
            vwap = float(last["vwap"])

            cross_up = (float(prev2["ema_fast"]) < float(prev2["ema_slow"]) and
                        float(prev["ema_fast"])  > float(prev["ema_slow"])  and
                        float(last["ema_fast"])  > float(last["ema_slow"]))

            if (cross_up and RSI_OS < rsi < RSI_OB and
                    mh > 0 and price > vwap and bool(last["vol_spike"])):
                spend      = balance * RISK
                coin_held  = spend / price
                balance   -= spend
                entry_price= price
                highest    = price
                in_trade   = True
        except:
            continue

    if in_trade:
        price = float(df.iloc[-1]["close"])
        pnl   = coin_held * (price - entry_price)
        balance += coin_held * price
        if pnl > 0: wins += 1;  gross_wins   += pnl
        else:       losses += 1; gross_losses += abs(pnl)

    total = wins + losses
    if total < 5: return None   # skip if too few trades

    win_rate = wins / total * 100
    pf       = gross_wins / gross_losses if gross_losses > 0 else 0
    pnl      = balance - START_BAL

    return {"wins": wins, "losses": losses, "win_rate": win_rate,
            "profit_factor": pf, "pnl": pnl, "balance": balance}

# ── Run optimization ───────────────────────────────────────
print(f"\nTesting {len(FAST_EMAS)*len(SLOW_EMAS)*len(SL_PCTS)*len(TP_PCTS)} combinations per symbol...")
print("This takes about 2 minutes...\n")

all_results = []
combos = [(f, s, sl, tp) for f, s, sl, tp in product(FAST_EMAS, SLOW_EMAS, SL_PCTS, TP_PCTS)
          if f < s]

for fast, slow, sl_pct, tp_pct in combos:
    combo_results = []
    for sym in SYMBOLS:
        r = run_backtest(data[sym], fast, slow, sl_pct, tp_pct)
        if r:
            combo_results.append(r)

    if len(combo_results) == len(SYMBOLS):
        avg_wr  = sum(r["win_rate"]       for r in combo_results) / len(combo_results)
        avg_pf  = sum(r["profit_factor"]  for r in combo_results) / len(combo_results)
        total_pnl = sum(r["pnl"]          for r in combo_results)
        total_t   = sum(r["wins"]+r["losses"] for r in combo_results)

        all_results.append({
            "fast": fast, "slow": slow,
            "sl": sl_pct, "tp": tp_pct,
            "win_rate": avg_wr,
            "profit_factor": avg_pf,
            "total_pnl": total_pnl,
            "trades": total_t,
        })

# ── Sort and display top results ───────────────────────────
if not all_results:
    print("No valid results — try different parameter ranges")
else:
    df_results = pd.DataFrame(all_results)

    # Score = profit_factor * win_rate/100 (balances both metrics)
    df_results["score"] = df_results["profit_factor"] * (df_results["win_rate"] / 100)
    df_results = df_results.sort_values("score", ascending=False)

    print("="*65)
    print("  TOP 10 SETTINGS (sorted by combined score)")
    print("="*65)
    print(f"  {'EMA':<10} {'SL%':<7} {'TP%':<7} {'WinRate':<10} {'ProfFactor':<13} {'P&L':<10} {'Trades'}")
    print(f"  {'-'*60}")

    for _, row in df_results.head(10).iterrows():
        ema_str = f"{int(row['fast'])}/{int(row['slow'])}"
        print(f"  {ema_str:<10} "
              f"{row['sl']*100:.1f}%  "
              f"  {row['tp']*100:.1f}%  "
              f"  {row['win_rate']:>6.1f}%  "
              f"  {row['profit_factor']:>8.2f}      "
              f"  ${row['total_pnl']:>+7.2f}   "
              f"  {int(row['trades'])}")

    best = df_results.iloc[0]
    print(f"\n{'='*65}")
    print(f"  BEST SETTINGS FOUND:")
    print(f"{'='*65}")
    print(f"  FAST_EMA = {int(best['fast'])}")
    print(f"  SLOW_EMA = {int(best['slow'])}")
    print(f"  SL_PCT   = {best['sl']}")
    print(f"  TP_PCT   = {best['tp']}")
    print(f"\n  Win rate:      {best['win_rate']:.1f}%")
    print(f"  Profit factor: {best['profit_factor']:.2f}")
    print(f"  Total P&L:     ${best['total_pnl']:+.2f}")
    print(f"\n  Copy these values into both bot.py and backtest.py")
    print("="*65)