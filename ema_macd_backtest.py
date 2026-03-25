"""
═══════════════════════════════════════════════════════════
  EMA + MACD BACKTEST — 2 indicators only
  
  BUY:  EMA 7/18 crossover UP + MACD bullish
  SELL: Trailing SL 0.3% or TP 1.5%
  
  Coins: BTC, ETH, SOL
  Timeframes: 5m, 15m, 30m, 1h
  Data: 4 years from Binance US
  
  Run:  python3 ema_macd_backtest.py
  Takes about 10-15 minutes (lots of data to download).
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import ccxt
import time as time_module
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

SYMBOLS      = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
FAST_EMA     = 7
SLOW_EMA     = 18
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIG     = 9
SL_PCT       = 0.003
TP_PCT       = 0.015
RISK         = 0.02
START_BAL    = 10_000.0

exchange = ccxt.binanceus()

TIMEFRAMES = [
    ("5m",  1460),
    ("15m", 1460),
    ("30m", 1460),
    ("1h",  1460),
]


def fetch_all_candles(symbol, timeframe, days_back):
    all_candles = []
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%dT%H:%M:%S')
    )
    limit = 1000
    coin = symbol.split("/")[0]
    print(f"    {coin} {timeframe}...", end="", flush=True)
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
    print(f" {len(df)} candles ({days}d / {days/365:.1f}yr)")
    return df


def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST,
                          window_slow=MACD_SLOW, window_sign=MACD_SIG)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["rsi"]       = ta.momentum.rsi(df["close"], window=14)
    return df


def run_backtest(df):
    df = add_indicators(df.copy())
    df = df.dropna()

    balance     = START_BAL
    in_trade    = False
    entry_price = 0.0
    highest     = 0.0
    coin_held   = 0.0
    wins = losses = 0
    gross_wins = gross_losses = 0.0
    tp_count = sl_count = 0
    trade_list = []

    for i in range(2, len(df)):
        prev = df.iloc[i - 1]
        last = df.iloc[i]
        price = float(last["close"])

        if in_trade:
            if price > highest:
                highest = price
            trail_sl    = highest * (1 - SL_PCT)
            take_profit = entry_price * (1 + TP_PCT)
            if price <= trail_sl or price >= take_profit:
                pnl = coin_held * (price - entry_price)
                balance += coin_held * price
                if pnl > 0:
                    wins += 1; gross_wins += pnl
                else:
                    losses += 1; gross_losses += abs(pnl)
                if price >= take_profit:
                    tp_count += 1
                else:
                    sl_count += 1
                trade_list.append({"pnl": pnl, "balance": balance})
                in_trade = False; coin_held = 0.0; highest = 0.0
            continue

        try:
            macd_h = float(last["macd_hist"])
            macd_v = float(last["macd"])
            macd_s = float(last["macd_sig"])
            if pd.isna(macd_h):
                continue

            ema_cross = (float(prev["ema_fast"]) < float(prev["ema_slow"]) and
                         float(last["ema_fast"]) > float(last["ema_slow"]))
            if not ema_cross:
                continue

            macd_bullish = (macd_v > macd_s and macd_h > 0)
            if not macd_bullish:
                continue

            spend       = balance * RISK
            coin_held   = spend / price
            balance    -= spend
            entry_price = price
            highest     = price
            in_trade    = True
        except:
            continue

    if in_trade:
        price = float(df.iloc[-1]["close"])
        pnl = coin_held * (price - entry_price)
        balance += coin_held * price
        if pnl > 0: wins += 1; gross_wins += pnl
        else: losses += 1; gross_losses += abs(pnl)
        trade_list.append({"pnl": pnl, "balance": balance})

    total = wins + losses
    if total < 1:
        return None

    peak = START_BAL; max_dd = 0.0
    for t in trade_list:
        if t["balance"] > peak: peak = t["balance"]
        dd = (peak - t["balance"]) / peak
        if dd > max_dd: max_dd = dd

    return {
        "wins": wins, "losses": losses, "total": total,
        "win_rate": wins / total * 100,
        "pf": gross_wins / gross_losses if gross_losses > 0 else 999,
        "pnl": balance - START_BAL,
        "balance": balance,
        "max_dd": max_dd,
        "tp": tp_count, "sl": sl_count,
        "avg_win": gross_wins / wins if wins > 0 else 0,
        "avg_loss": gross_losses / losses if losses > 0 else 0,
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  EMA + MACD BACKTEST (2 indicators only)")
    print(f"  BUY:  EMA {FAST_EMA}/{SLOW_EMA} crossover + MACD bullish")
    print(f"  SELL: Trailing SL {SL_PCT*100:.1f}% or TP {TP_PCT*100:.1f}%")
    print(f"  Risk: {RISK*100:.0f}% per trade")
    print(f"  Coins: {', '.join(s.split('/')[0] for s in SYMBOLS)}")
    print(f"  Timeframes: 5m, 15m, 30m, 1h")
    print(f"  Data: 4 years from Binance US")
    print("=" * 70)

    # Download all data
    print("\nDownloading data (this takes 10-15 min for all combos)...\n")

    all_data = {}
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        print(f"  {coin}:")
        all_data[sym] = {}
        for tf, days in TIMEFRAMES:
            df = fetch_all_candles(sym, tf, days)
            all_data[sym][tf] = df
        print()

    # Run backtests
    all_results = {}

    for tf, _ in TIMEFRAMES:
        print(f"\n{'─'*70}")
        print(f"  TIMEFRAME: {tf.upper()}")
        print(f"{'─'*70}")

        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            df = all_data[sym][tf]
            if len(df) < 50:
                print(f"\n  {coin} — not enough data")
                continue

            days_actual = (df.index[-1] - df.index[0]).days
            r = run_backtest(df)
            key = f"{coin}_{tf}"
            all_results[key] = r

            if r:
                pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
                print(f"\n  {coin} on {tf} ({days_actual}d / {days_actual/365:.1f}yr)")
                print(f"  {'─'*55}")
                print(f"  Balance:      ${r['balance']:,.2f}  (P&L: ${r['pnl']:+.2f})")
                print(f"  Trades:       {r['total']}  ({r['tp']} TP / {r['sl']} SL)")
                print(f"  Win rate:     {r['win_rate']:.1f}%  ({r['wins']}W / {r['losses']}L)")
                print(f"  Avg win:      ${r['avg_win']:.2f}")
                print(f"  Avg loss:     ${r['avg_loss']:.2f}")
                print(f"  Profit factor: {pf_str}")
                print(f"  Max drawdown: {r['max_dd']*100:.1f}%")
            else:
                print(f"\n  {coin} on {tf} — no trades")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY — EMA {FAST_EMA}/{SLOW_EMA} + MACD")
    print(f"  SL: {SL_PCT*100:.1f}% | TP: {TP_PCT*100:.1f}%")
    print(f"{'='*70}")
    print(f"  {'Coin':<5} {'TF':<5} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>11} {'DD':>6} {'Data':>6}")
    print(f"  {'─'*60}")

    for tf, _ in TIMEFRAMES:
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            key = f"{coin}_{tf}"
            r = all_results.get(key)
            if r:
                pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
                df = all_data[sym][tf]
                days = (df.index[-1] - df.index[0]).days if len(df) > 0 else 0
                print(f"  {coin:<5} {tf:<5} {r['total']:>7} {r['win_rate']:>6.1f}% "
                      f"{pf_str:>8} ${r['pnl']:>+9.2f} {r['max_dd']*100:>5.1f}% {days:>4}d")

    # Combined per timeframe
    print(f"\n{'='*70}")
    print(f"  COMBINED PER TIMEFRAME (all 3 coins)")
    print(f"{'='*70}")
    print(f"  {'TF':<6} {'Trades':>8} {'Avg WR':>8} {'Avg PF':>8} {'Total P&L':>12} {'Worst DD':>9}")
    print(f"  {'─'*55}")

    for tf, _ in TIMEFRAMES:
        tf_trades = 0; tf_wins = 0; tf_pnl = 0; tf_pf_sum = 0
        tf_count = 0; tf_max_dd = 0
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            key = f"{coin}_{tf}"
            r = all_results.get(key)
            if r and r["total"] >= 5:
                tf_trades += r["total"]
                tf_wins   += r["wins"]
                tf_pnl    += r["pnl"]
                tf_pf_sum += r["pf"]
                tf_count  += 1
                if r["max_dd"] > tf_max_dd:
                    tf_max_dd = r["max_dd"]

        if tf_count > 0 and tf_trades > 0:
            avg_wr = tf_wins / tf_trades * 100
            avg_pf = tf_pf_sum / tf_count
            pf_str = f"{avg_pf:.2f}" if avg_pf < 999 else "inf"
            print(f"  {tf:<6} {tf_trades:>8} {avg_wr:>7.1f}% {pf_str:>8} "
                  f"${tf_pnl:>+10.2f} {tf_max_dd*100:>8.1f}%")

    # Per coin analysis
    print(f"\n{'='*70}")
    print(f"  BEST TIMEFRAME PER COIN")
    print(f"{'='*70}")

    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        best_key = None
        best_pf  = 0
        for tf, _ in TIMEFRAMES:
            key = f"{coin}_{tf}"
            r = all_results.get(key)
            if r and r["total"] >= 10 and r["pf"] > best_pf:
                best_pf  = r["pf"]
                best_key = key

        if best_key:
            r = all_results[best_key]
            tf = best_key.split("_")[1]
            pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
            verdict = "✅ PROFITABLE" if r['pf'] > 1.0 else "❌ LOSING"
            print(f"\n  {coin}: Best on {tf}")
            print(f"    PF: {pf_str} | WR: {r['win_rate']:.1f}% | "
                  f"Trades: {r['total']} | P&L: ${r['pnl']:+.2f} | {verdict}")

    # Final recommendation
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")

    # Find overall best
    viable = {k: v for k, v in all_results.items() if v and v["total"] >= 20}
    if viable:
        best_key = max(viable, key=lambda k: viable[k]["pf"])
        best = viable[best_key]
        parts = best_key.split("_")
        coin = parts[0]
        tf   = parts[1]
        pf_str = f"{best['pf']:.2f}" if best['pf'] < 999 else "inf"

        print(f"\n  OVERALL BEST: {coin} on {tf}")
        print(f"    PF: {pf_str} | WR: {best['win_rate']:.1f}% | "
              f"Trades: {best['total']} | P&L: ${best['pnl']:+.2f}")

        # Compare timeframes fairly
        print(f"\n  TIMEFRAME RANKING (by avg PF across all coins):")
        tf_ranking = []
        for tf, _ in TIMEFRAMES:
            pfs = []
            for sym in SYMBOLS:
                key = f"{sym.split('/')[0]}_{tf}"
                r = all_results.get(key)
                if r and r["total"] >= 10:
                    pfs.append(r["pf"])
            if pfs:
                tf_ranking.append((tf, sum(pfs)/len(pfs), sum(
                    all_results.get(f"{s.split('/')[0]}_{tf}", {}).get("pnl", 0)
                    for s in SYMBOLS)))

        tf_ranking.sort(key=lambda x: x[1], reverse=True)
        for i, (tf, avg_pf, total_pnl) in enumerate(tf_ranking):
            pf_str = f"{avg_pf:.2f}" if avg_pf < 999 else "inf"
            marker = " ◀ BEST" if i == 0 else ""
            print(f"    {i+1}. {tf:<5} Avg PF: {pf_str}  |  Total P&L: ${total_pnl:+.2f}{marker}")

        print(f"\n  Compare to your current bot:")
        print(f"    Current (30m + EMA + MACD + 1h filter): PF 1.62")
        print(f"    Best EMA+MACD only: {tf_ranking[0][0]} PF {tf_ranking[0][1]:.2f}")
        if tf_ranking[0][1] > 1.62:
            print(f"    → EMA+MACD on {tf_ranking[0][0]} beats current setup!")
        else:
            print(f"    → Current setup with 1h filter is still better quality")

    print(f"\n{'='*70}")
    print(f"  Done!")
    print(f"{'='*70}\n")
