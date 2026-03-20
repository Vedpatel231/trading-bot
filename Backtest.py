"""
backtest.py — run this locally to test your strategy on historical data.
Usage:  python3 backtest.py
No API keys needed — uses free Yahoo Finance data.
"""

import pandas as pd
import ta
import yfinance as yf
from datetime import datetime

# ══════════════════════════════════════════════════════════════
#  SETTINGS — must match bot.py
# ══════════════════════════════════════════════════════════════

SYMBOLS     = ["ETH-USD", "SOL-USD"]  # Yahoo Finance format
INTERVAL    = "1h"       # 1h candles (max 2 years on Yahoo)
PERIOD      = "1y"       # how far back to test

FAST_EMA    = 7
SLOW_EMA    = 18
RSI_PERIOD  = 14
BB_PERIOD   = 20
BB_STD      = 2.0
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIG    = 9
RSI_OB      = 65
RSI_OS      = 35
SL_PCT      = 0.003
TP_PCT      = 0.015
START_BAL   = 10000.0
RISK        = 0.02

# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════

def add_indicators(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df["ema_fast"]  = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"]  = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]       = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    bb = ta.volatility.BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()

    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST,
                          window_slow=MACD_SLOW, window_sign=MACD_SIG)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["vwap"]      = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_avg"] * 1.2
    df["atr"]       = ta.volatility.average_true_range(
                        df["high"], df["low"], df["close"], window=14)
    return df

# ══════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

def backtest(symbol):
    print(f"\n{'='*55}")
    print(f"  Backtesting {symbol} | {INTERVAL} | {PERIOD}")
    print(f"{'='*55}")

    # Download data
    raw = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)
    if raw.empty:
        print(f"  No data for {symbol}")
        return None

    df = add_indicators(raw)
    df = df.dropna()

    balance    = START_BAL
    in_trade   = False
    entry_price= 0.0
    highest    = 0.0
    coin_held  = 0.0
    trades     = []

    for i in range(3, len(df)):
        prev2 = df.iloc[i-2]
        prev  = df.iloc[i-1]
        last  = df.iloc[i]
        price = float(last["close"])

        # Check exits first
        if in_trade:
            if price > highest: highest = price
            trail_sl    = highest * (1 - SL_PCT)
            take_profit = entry_price * (1 + TP_PCT)

            if price <= trail_sl or price >= take_profit:
                pnl      = coin_held * (price - entry_price)
                balance += coin_held * price
                reason   = "TP" if price >= take_profit else "SL"
                trades.append({
                    "exit_time":  last.name,
                    "entry":      entry_price,
                    "exit":       price,
                    "pnl":        pnl,
                    "result":     "WIN" if pnl > 0 else "LOSS",
                    "reason":     reason,
                    "balance":    balance,
                })
                in_trade    = False
                coin_held   = 0.0
                highest     = 0.0
                entry_price = 0.0
                continue

        # Skip if already in trade
        if in_trade:
            continue

        # Signal
        try:
            rsi  = float(last["rsi"])
            macd_h = float(last["macd_hist"])
            vwap   = float(last["vwap"])
            atr    = float(last["atr"])

            ema_cross_up = (float(prev2["ema_fast"]) < float(prev2["ema_slow"]) and
                            float(prev["ema_fast"])  > float(prev["ema_slow"])  and
                            float(last["ema_fast"])  > float(last["ema_slow"]))

            macd_bullish = macd_h > 0
            above_vwap   = price > vwap
            vol_spike    = bool(last["vol_spike"])

            buy = (ema_cross_up and
                   RSI_OS < rsi < RSI_OB and
                   macd_bullish and above_vwap and vol_spike)

            if buy:
                spend      = balance * RISK
                coin_held  = spend / price
                balance   -= spend
                entry_price= price
                highest    = price
                in_trade   = True
        except:
            continue

    # Close any open trade at end
    if in_trade:
        price = float(df.iloc[-1]["close"])
        pnl   = coin_held * (price - entry_price)
        balance += coin_held * price
        trades.append({
            "exit_time": df.index[-1],
            "entry":     entry_price,
            "exit":      price,
            "pnl":       pnl,
            "result":    "WIN" if pnl > 0 else "LOSS",
            "reason":    "END",
            "balance":   balance,
        })

    # ── Results ────────────────────────────────────────────
    if not trades:
        print(f"  No trades generated — try different settings")
        return None

    results_df = pd.DataFrame(trades)
    wins       = results_df[results_df["result"] == "WIN"]
    losses     = results_df[results_df["result"] == "LOSS"]
    win_rate   = len(wins) / len(results_df) * 100
    total_pnl  = results_df["pnl"].sum()
    avg_win    = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_loss   = losses["pnl"].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")

    # Max drawdown
    peak     = START_BAL
    max_dd   = 0.0
    for bal in results_df["balance"]:
        if bal > peak: peak = bal
        dd = (peak - bal) / peak
        if dd > max_dd: max_dd = dd

    print(f"\n  Results for {symbol}:")
    print(f"  ─────────────────────────────────────")
    print(f"  Starting balance: ${START_BAL:,.2f}")
    print(f"  Final balance:    ${balance:,.2f}")
    print(f"  Total P&L:        ${total_pnl:+.2f} ({total_pnl/START_BAL*100:+.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Total trades:     {len(results_df)}")
    print(f"  Win rate:         {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win:          ${avg_win:+.2f}")
    print(f"  Avg loss:         ${avg_loss:+.2f}")
    print(f"  Profit factor:    {profit_factor:.2f}")
    print(f"  Max drawdown:     {max_dd*100:.1f}%")
    print(f"  ─────────────────────────────────────")

    # Interpretation
    print(f"\n  Interpretation:")
    if win_rate >= 55:
        print(f"  ✅ Win rate {win_rate:.1f}% is solid")
    elif win_rate >= 45:
        print(f"  ⚠️  Win rate {win_rate:.1f}% is borderline — needs bigger wins than losses")
    else:
        print(f"  ❌ Win rate {win_rate:.1f}% is low — adjust settings")

    if profit_factor >= 1.5:
        print(f"  ✅ Profit factor {profit_factor:.2f} is good (above 1.5)")
    elif profit_factor >= 1.0:
        print(f"  ⚠️  Profit factor {profit_factor:.2f} is marginal (aim for 1.5+)")
    else:
        print(f"  ❌ Profit factor {profit_factor:.2f} — losing strategy")

    if max_dd <= 0.15:
        print(f"  ✅ Max drawdown {max_dd*100:.1f}% is acceptable")
    elif max_dd <= 0.25:
        print(f"  ⚠️  Max drawdown {max_dd*100:.1f}% is moderate")
    else:
        print(f"  ❌ Max drawdown {max_dd*100:.1f}% is too high — tighten stop loss")

    return {
        "symbol":         symbol,
        "final_balance":  balance,
        "total_pnl":      total_pnl,
        "win_rate":       win_rate,
        "trades":         len(results_df),
        "profit_factor":  profit_factor,
        "max_drawdown":   max_dd,
    }

# ══════════════════════════════════════════════════════════════
#  RUN ALL SYMBOLS
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  BACKTEST ENGINE")
    print(f"  Strategy: EMA {FAST_EMA}/{SLOW_EMA} + RSI + MACD + BB + VWAP")
    print(f"  SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%  |  Risk: {RISK*100:.0f}%")
    print(f"  Period: {PERIOD}  |  Interval: {INTERVAL}")
    print("="*55)

    all_results = []
    for symbol in SYMBOLS:
        result = backtest(symbol)
        if result:
            all_results.append(result)

    if all_results:
        print(f"\n{'='*55}")
        print(f"  COMBINED SUMMARY")
        print(f"{'='*55}")
        total_final = sum(r["final_balance"] for r in all_results)
        total_pnl   = sum(r["total_pnl"]     for r in all_results)
        avg_wr      = sum(r["win_rate"]       for r in all_results) / len(all_results)
        total_trades= sum(r["trades"]         for r in all_results)
        avg_pf      = sum(r["profit_factor"]  for r in all_results) / len(all_results)
        max_dd      = max(r["max_drawdown"]   for r in all_results)

        print(f"  Total final balance: ${total_final:,.2f}")
        print(f"  Total P&L:           ${total_pnl:+.2f}")
        print(f"  Average win rate:    {avg_wr:.1f}%")
        print(f"  Total trades:        {total_trades}")
        print(f"  Avg profit factor:   {avg_pf:.2f}")
        print(f"  Worst drawdown:      {max_dd*100:.1f}%")
        print(f"\n  {'✅ STRATEGY LOOKS VIABLE' if avg_wr > 50 and avg_pf > 1.2 else '⚠️  NEEDS TUNING BEFORE GOING LIVE'}")
        print("="*55)

    print(f"\n  Tip: change FAST_EMA, SLOW_EMA, SL_PCT, TP_PCT at the")
    print(f"  top of backtest.py and re-run to find optimal settings.\n")