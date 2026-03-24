"""
═══════════════════════════════════════════════════════════
  MACD-ONLY BACKTEST — 15m candles
  
  Tests MACD as the SOLE indicator on ETH + SOL
  with 1:3 risk-reward ratio (SL 0.3% / TP 0.9%)
  
  Uses ccxt to pull MAX available 15m data from Binance US.
  SOL launched ~2020, ETH has data since ~2017.
  15m candles available: ~1-2 years on Binance US.
  
  For longer history, also tests on 1h and daily candles
  to show MACD performance across all available timeframes.
  
  Run:  python3 macd_only_backtest.py
  Takes about 3-5 minutes.
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import ccxt
import time as time_module
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

SYMBOLS      = ["ETH/USDT", "SOL/USDT"]
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIG     = 9

# SL/TP ratio
SL_PCT       = 0.003    # 0.3% stop loss
TP_PCT       = 0.015    # 1.5% take profit

RISK         = 0.02
START_BAL    = 10_000.0

exchange = ccxt.binanceus()


def fetch_all_candles(symbol, timeframe, days_back):
    """Fetch historical OHLCV data with pagination."""
    all_candles = []
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%dT%H:%M:%S')
    )
    limit = 1000
    print(f"    Fetching {symbol} {timeframe} ({days_back} days)...", end="", flush=True)
    retries = 0

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            retries += 1
            if retries > 5:
                print(f" ERROR: {e}")
                break
            time_module.sleep(2)
            continue

        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < limit:
            break
        time_module.sleep(0.3)

    if not all_candles:
        print(f" no data!")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    days = (df.index[-1] - df.index[0]).days
    print(f" {len(df)} candles ({days} days / {days/365:.1f} years)")
    return df


def add_macd(df):
    """Add ONLY MACD — no other indicators."""
    df = df.copy()
    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST,
                          window_slow=MACD_SLOW, window_sign=MACD_SIG)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    return df


def run_backtest(df, label=""):
    """
    MACD-only strategy:
      BUY:  MACD line crosses above signal line (histogram goes from - to +)
      SELL: MACD line crosses below signal line (histogram goes from + to -)
      Exits: trailing SL 0.3% or TP 0.9% (1:3 ratio)
    """
    df = add_macd(df.copy())
    df = df.dropna()

    balance     = START_BAL
    in_trade    = False
    entry_price = 0.0
    highest     = 0.0
    coin_held   = 0.0
    wins = losses = 0
    gross_wins = gross_losses = 0.0
    trade_list  = []

    for i in range(2, len(df)):
        prev = df.iloc[i - 1]
        last = df.iloc[i]
        price = float(last["close"])

        # Check exits
        if in_trade:
            if price > highest:
                highest = price
            trail_sl    = highest * (1 - SL_PCT)
            take_profit = entry_price * (1 + TP_PCT)

            if price <= trail_sl or price >= take_profit:
                pnl      = coin_held * (price - entry_price)
                balance += coin_held * price
                if pnl > 0:
                    wins += 1; gross_wins += pnl
                else:
                    losses += 1; gross_losses += abs(pnl)
                trade_list.append({
                    "pnl": pnl, "balance": balance,
                    "reason": "TP" if price >= take_profit else "SL"
                })
                in_trade = False; coin_held = 0.0; highest = 0.0
            continue

        # MACD crossover signal — ONLY indicator
        try:
            prev_hist = float(prev["macd_hist"])
            curr_hist = float(last["macd_hist"])
            macd_val  = float(last["macd"])
            macd_sig  = float(last["macd_sig"])

            if pd.isna(prev_hist) or pd.isna(curr_hist):
                continue

            # BUY: histogram crosses from negative to positive
            # (MACD line crosses above signal line)
            macd_cross_up = (prev_hist <= 0 and curr_hist > 0)

            if macd_cross_up:
                spend       = balance * RISK
                coin_held   = spend / price
                balance    -= spend
                entry_price = price
                highest     = price
                in_trade    = True

        except:
            continue

    # Close any open trade
    if in_trade:
        price = float(df.iloc[-1]["close"])
        pnl   = coin_held * (price - entry_price)
        balance += coin_held * price
        if pnl > 0: wins += 1; gross_wins += pnl
        else: losses += 1; gross_losses += abs(pnl)
        trade_list.append({"pnl": pnl, "balance": balance, "reason": "END"})

    total = wins + losses
    if total < 1:
        return None

    # Max drawdown
    peak = START_BAL; max_dd = 0.0
    for t in trade_list:
        if t["balance"] > peak: peak = t["balance"]
        dd = (peak - t["balance"]) / peak
        if dd > max_dd: max_dd = dd

    # Trade breakdown
    tp_count = sum(1 for t in trade_list if t["reason"] == "TP")
    sl_count = sum(1 for t in trade_list if t["reason"] == "SL")

    return {
        "wins": wins, "losses": losses, "total": total,
        "win_rate": wins / total * 100,
        "pf": gross_wins / gross_losses if gross_losses > 0 else 999,
        "pnl": balance - START_BAL,
        "balance": balance,
        "max_dd": max_dd,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "avg_win": gross_wins / wins if wins > 0 else 0,
        "avg_loss": gross_losses / losses if losses > 0 else 0,
    }


def print_result(label, r):
    if r is None:
        print(f"\n  {label}")
        print(f"  {'─'*60}")
        print(f"  No trades generated")
        return

    pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
    print(f"\n  {label}")
    print(f"  {'─'*60}")
    print(f"  Balance:       ${r['balance']:,.2f}  (P&L: ${r['pnl']:+.2f})")
    print(f"  Total trades:  {r['total']}  ({r['tp_count']} TP / {r['sl_count']} SL)")
    print(f"  Win rate:      {r['win_rate']:.1f}%  ({r['wins']}W / {r['losses']}L)")
    print(f"  Avg win:       ${r['avg_win']:.2f}")
    print(f"  Avg loss:      ${r['avg_loss']:.2f}")
    print(f"  Profit factor: {pf_str}")
    print(f"  Max drawdown:  {r['max_dd']*100:.1f}%")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  MACD-ONLY BACKTEST")
    print(f"  Signal: MACD crossover ONLY (no EMA, no RSI, no volume)")
    print(f"  MACD:   {MACD_FAST}/{MACD_SLOW}/{MACD_SIG}")
    print(f"  SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%  (1:{TP_PCT/SL_PCT:.0f} ratio)")
    print(f"  Risk: {RISK*100:.0f}% per trade")
    print(f"  Data: up to 4 years (max available from Binance US)")
    print(f"  Coins: {', '.join(s.split('/')[0] for s in SYMBOLS)}")
    print("=" * 65)

    # ── Download data ─────────────────────────────────
    # All timeframes — request 4 years, exchange returns max available
    # Binance US typically has: 1m ~30d, 5m ~1yr, 15m ~1yr, 30m ~2yr, 1h ~4yr
    # Script fetches whatever is available — no error if less than 4 years
    timeframes = [
        ("1m",  1460, "1m candles"),
        ("5m",  1460, "5m candles"),
        ("15m", 1460, "15m candles"),
        ("30m", 1460, "30m candles"),
        ("1h",  1460, "1h candles"),
    ]

    all_data = {}
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        print(f"\n  {coin}:")
        all_data[sym] = {}
        for tf, days, desc in timeframes:
            df = fetch_all_candles(sym, tf, days)
            all_data[sym][tf] = df

    # ── Run backtests ─────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS — MACD ONLY")
    print(f"  SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%  (1:{TP_PCT/SL_PCT:.0f} ratio)")
    print(f"{'='*65}")

    all_results = {}
    for tf, days, desc in timeframes:
        print(f"\n{'─'*65}")
        print(f"  TIMEFRAME: {tf.upper()} CANDLES")
        print(f"{'─'*65}")

        combined_trades = 0
        combined_wins   = 0
        combined_pnl    = 0
        combined_pf_sum = 0
        sym_count       = 0

        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            df = all_data[sym][tf]
            if len(df) < 50:
                print(f"\n  {coin} — not enough data"); continue

            days_actual = (df.index[-1] - df.index[0]).days
            r = run_backtest(df)
            key = f"{coin}_{tf}"
            all_results[key] = r

            print_result(f"{coin} on {tf} ({days_actual} days / {days_actual/365:.1f} years)", r)

            if r:
                combined_trades += r["total"]
                combined_wins   += r["wins"]
                combined_pnl    += r["pnl"]
                combined_pf_sum += r["pf"]
                sym_count       += 1

        if sym_count > 0 and combined_trades > 0:
            avg_wr = combined_wins / combined_trades * 100
            avg_pf = combined_pf_sum / sym_count
            print(f"\n  COMBINED {tf.upper()}:")
            print(f"  Total trades: {combined_trades}  |  WR: {avg_wr:.1f}%  |  "
                  f"Avg PF: {avg_pf:.2f}  |  Total P&L: ${combined_pnl:+.2f}")

    # ── Summary comparison ────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY — MACD ONLY ACROSS ALL TIMEFRAMES")
    print(f"  SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%  (1:{TP_PCT/SL_PCT:.0f} ratio)")
    print(f"{'='*65}")
    print(f"  {'Config':<25} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>11} {'DD':>6}")
    print(f"  {'─'*65}")

    for tf, _, _ in timeframes:
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            key = f"{coin}_{tf}"
            r = all_results.get(key)
            if r:
                pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
                print(f"  {coin:<4} {tf:<20} {r['total']:>7} {r['win_rate']:>6.1f}% "
                      f"{pf_str:>8} ${r['pnl']:>+9.2f} {r['max_dd']*100:>5.1f}%")

    # ── Analysis ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ANALYSIS")
    print(f"{'='*65}")

    # Best config
    viable = {k: v for k, v in all_results.items() if v and v["total"] >= 10}
    if viable:
        best_key = max(viable, key=lambda k: viable[k]["pf"])
        best = viable[best_key]
        worst_key = min(viable, key=lambda k: viable[k]["pf"])
        worst = viable[worst_key]

        print(f"\n  BEST:  {best_key.replace('_', ' on ')}")
        pf_str = f"{best['pf']:.2f}" if best['pf'] < 999 else "inf"
        print(f"    PF: {pf_str}  |  WR: {best['win_rate']:.1f}%  |  "
              f"Trades: {best['total']}  |  P&L: ${best['pnl']:+.2f}")

        print(f"\n  WORST: {worst_key.replace('_', ' on ')}")
        print(f"    PF: {worst['pf']:.2f}  |  WR: {worst['win_rate']:.1f}%  |  "
              f"Trades: {worst['total']}  |  P&L: ${worst['pnl']:+.2f}")

        # Compare 15m specifically since that's what you asked about
        print(f"\n  15m CANDLE RESULTS (what you asked about):")
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            key = f"{coin}_15m"
            r = all_results.get(key)
            if r:
                pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
                verdict = "✅ PROFITABLE" if r['pf'] > 1.0 else "❌ LOSING"
                print(f"    {coin}: PF {pf_str}  |  WR {r['win_rate']:.1f}%  |  "
                      f"{r['total']} trades  |  ${r['pnl']:+.2f}  |  {verdict}")

        # Risk-reward analysis
        print(f"\n  RISK-REWARD ANALYSIS:")
        print(f"    SL: {SL_PCT*100:.1f}% (${START_BAL * RISK * SL_PCT:.2f} risk per trade)")
        print(f"    TP: {TP_PCT*100:.1f}% (${START_BAL * RISK * TP_PCT:.2f} reward per trade)")
        print(f"    Ratio: 1:{TP_PCT/SL_PCT:.0f}")
        print(f"    Breakeven win rate needed: {100/(1+TP_PCT/SL_PCT):.1f}%")

    # ── Final verdict ─────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  VERDICT: Which timeframe is best for MACD-only?")
    print(f"{'='*65}")

    # Find best timeframe across both coins combined
    tf_scores = {}
    for tf_name in ["1m", "5m", "15m", "30m", "1h"]:
        tf_trades = 0
        tf_pnl    = 0
        tf_pf_sum = 0
        tf_count  = 0
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            key = f"{coin}_{tf_name}"
            r = all_results.get(key)
            if r and r["total"] >= 5:
                tf_trades += r["total"]
                tf_pnl    += r["pnl"]
                tf_pf_sum += r["pf"]
                tf_count  += 1
        if tf_count > 0:
            tf_scores[tf_name] = {
                "trades": tf_trades,
                "pnl": tf_pnl,
                "avg_pf": tf_pf_sum / tf_count,
            }

    if tf_scores:
        print(f"\n  {'TF':<6} {'Trades':>8} {'Avg PF':>8} {'Total P&L':>12}")
        print(f"  {'─'*36}")
        for tf_name in ["1m", "5m", "15m", "30m", "1h"]:
            if tf_name in tf_scores:
                s = tf_scores[tf_name]
                pf_str = f"{s['avg_pf']:.2f}" if s['avg_pf'] < 999 else "inf"
                print(f"  {tf_name:<6} {s['trades']:>8} {pf_str:>8} ${s['pnl']:>+10.2f}")

        best_tf = max(tf_scores, key=lambda k: tf_scores[k]["avg_pf"])
        best = tf_scores[best_tf]
        print(f"\n  BEST TIMEFRAME: {best_tf}")
        pf_str = f"{best['avg_pf']:.2f}" if best['avg_pf'] < 999 else "inf"
        print(f"    PF: {pf_str}  |  Trades: {best['trades']}  |  P&L: ${best['pnl']:+.2f}")

        print(f"\n  Compare to your current setup:")
        print(f"    Current (30m + EMA + MACD + Vol + 1h filter): PF 1.84")
        print(f"    Best MACD-only: {best_tf} PF {pf_str}")
        if best["avg_pf"] > 1.84:
            print(f"    → MACD-only on {best_tf} is BETTER")
        else:
            print(f"    → Current setup has better quality per trade")
            print(f"      But MACD-only gives more total trades")

    print(f"\n{'='*65}")
    print(f"  Done!")
    print(f"{'='*65}\n")
