"""
═══════════════════════════════════════════════════════════
  LONG-TERM FAIR TIMEFRAME TEST
  
  Uses ccxt to download 1 YEAR of 5m, 15m, 1h data directly
  from Binance US (free, no API key needed for historical data).
  
  ALL timeframes tested on the SAME 1-year period.
  Real candle data — not resampled.
  
  Strategy: EMA 7/18 + MACD + Volume (optimized)
  Symbols: ETH + SOL
  
  Run:  python3 fair_timeframe_test.py
  Takes about 3-5 minutes (downloading + backtesting).
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import ccxt
import time as time_module
import math
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

SYMBOLS      = ["ETH/USDT", "SOL/USDT"]
FAST_EMA     = 7
SLOW_EMA     = 18
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIG     = 9
SL_PCT       = 0.003
TP_PCT       = 0.015
RISK         = 0.02
START_BAL    = 10_000.0
MIN_VOL_MULT = 1.2

# How far back to fetch (in days)
LOOKBACK_DAYS = 365

exchange = ccxt.binanceus()


def fetch_all_candles(symbol, timeframe, days_back):
    """Fetch historical OHLCV data with pagination."""
    all_candles = []
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%dT%H:%M:%S')
    )
    limit = 1000

    print(f"    Fetching {symbol} {timeframe}...", end="", flush=True)
    retries = 0

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            retries += 1
            if retries > 5:
                print(f" ERROR: {e}")
                break
            print(f" (retry {retries})...", end="", flush=True)
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
    print(f" {len(df)} candles ({days} days)")
    return df


def resample(df, tf):
    return df.resample(tf).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()


def add_indicators(df):
    df = df.copy()
    df["ema_fast"]  = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"]  = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST,
                          window_slow=MACD_SLOW, window_sign=MACD_SIG)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_avg"] * MIN_VOL_MULT
    df["rsi"]       = ta.momentum.rsi(df["close"], window=14)
    df["atr"]       = ta.volatility.average_true_range(
                        df["high"], df["low"], df["close"], window=14)
    return df


def compute_htf_trend(entry_df, htf_df):
    htf = add_indicators(htf_df.copy())
    htf["htf_trend"] = "NEUTRAL"
    htf.loc[htf["ema_fast"] > htf["ema_slow"], "htf_trend"] = "UP"
    htf.loc[htf["ema_fast"] < htf["ema_slow"], "htf_trend"] = "DOWN"
    trends = []
    htf_times  = htf.index
    htf_trends = htf["htf_trend"].values
    for ts in entry_df.index:
        mask = htf_times <= ts
        if mask.any():
            trends.append(htf_trends[mask.sum() - 1])
        else:
            trends.append("NEUTRAL")
    return pd.Series(trends, index=entry_df.index)


def run_backtest(df, htf_trends=None):
    df = add_indicators(df.copy())
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

        if in_trade:
            if price > highest: highest = price
            trail_sl    = highest * (1 - SL_PCT)
            take_profit = entry_price * (1 + TP_PCT)
            if price <= trail_sl or price >= take_profit:
                pnl      = coin_held * (price - entry_price)
                balance += coin_held * price
                if pnl > 0: wins += 1; gross_wins += pnl
                else: losses += 1; gross_losses += abs(pnl)
                trade_list.append({"pnl": pnl, "balance": balance})
                in_trade = False; coin_held = 0.0; highest = 0.0
            continue

        try:
            macd_h = float(last["macd_hist"])
            macd_v = float(last["macd"])
            macd_s = float(last["macd_sig"])
            if pd.isna(macd_h): continue

            ema_cross = (float(prev["ema_fast"]) < float(prev["ema_slow"]) and
                         float(last["ema_fast"]) > float(last["ema_slow"]))
            if not ema_cross: continue

            macd_bullish = (macd_v > macd_s and macd_h > 0)
            vol_spike    = bool(last["vol_spike"])
            if not (macd_bullish and vol_spike): continue

            if htf_trends is not None:
                ts = df.index[i]
                if ts in htf_trends.index:
                    trend = htf_trends.loc[ts]
                else:
                    mask = htf_trends.index <= ts
                    trend = htf_trends[mask].iloc[-1] if mask.any() else "NEUTRAL"
                if trend != "UP": continue

            spend       = balance * RISK
            coin_held   = spend / price
            balance    -= spend
            entry_price = price
            highest     = price
            in_trade    = True
        except: continue

    if in_trade:
        price = float(df.iloc[-1]["close"])
        pnl   = coin_held * (price - entry_price)
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
        "pnl": balance - START_BAL, "max_dd": max_dd,
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  LONG-TERM FAIR TIMEFRAME TEST")
    print(f"  ALL timeframes tested on the SAME {LOOKBACK_DAYS}-day period")
    print(f"  Coins: {', '.join(s.split('/')[0] for s in SYMBOLS)}")
    print(f"  Strategy: EMA {FAST_EMA}/{SLOW_EMA} + MACD + Volume")
    print(f"  SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%")
    print(f"  Data source: Binance US via ccxt (real candles)")
    print("=" * 70)

    print(f"\nDownloading {LOOKBACK_DAYS} days of REAL candle data...")
    print("  (takes 2-3 min — paginating through Binance API)\n")

    timeframes_to_fetch = ["5m", "15m", "1h"]

    all_data = {}
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        print(f"  {coin}:")
        all_data[sym] = {}

        for tf in timeframes_to_fetch:
            df = fetch_all_candles(sym, tf, LOOKBACK_DAYS)
            all_data[sym][tf] = df

        # Resample for filters not directly available
        if len(all_data[sym]["15m"]) > 0:
            all_data[sym]["30m"] = resample(all_data[sym]["15m"], "30min")
            print(f"    30m: {len(all_data[sym]['30m'])} candles (resampled from 15m)")

        if len(all_data[sym]["1h"]) > 0:
            all_data[sym]["2h"] = resample(all_data[sym]["1h"], "2h")
            all_data[sym]["4h"] = resample(all_data[sym]["1h"], "4h")
            print(f"    2h:  {len(all_data[sym]['2h'])} candles (resampled from 1h)")
            print(f"    4h:  {len(all_data[sym]['4h'])} candles (resampled from 1h)")
        print()

    # Trim to common date range
    print("Aligning all data to the same date range...")
    latest_start = None
    earliest_end = None
    for sym in SYMBOLS:
        for tf in timeframes_to_fetch:
            df = all_data[sym][tf]
            if len(df) == 0: continue
            s, e = df.index[0], df.index[-1]
            if latest_start is None or s > latest_start: latest_start = s
            if earliest_end is None or e < earliest_end: earliest_end = e

    days_common = 0
    if latest_start and earliest_end:
        days_common = (earliest_end - latest_start).days
        print(f"  Common range: {latest_start.strftime('%Y-%m-%d')} to "
              f"{earliest_end.strftime('%Y-%m-%d')} ({days_common} days)")
        for sym in SYMBOLS:
            for tf_key in list(all_data[sym].keys()):
                df = all_data[sym][tf_key]
                if len(df) > 0:
                    all_data[sym][tf_key] = df[(df.index >= latest_start) & (df.index <= earliest_end)]

    configs = [
        ("5m entry, NO filter",     "5m",  None),
        ("5m entry, 15m filter",    "5m",  "15m"),
        ("5m entry, 30m filter",    "5m",  "30m"),
        ("5m entry, 1h filter",     "5m",  "1h"),
        ("15m entry, NO filter",    "15m", None),
        ("15m entry, 1h filter",    "15m", "1h"),
        ("15m entry, 2h filter",    "15m", "2h"),
        ("30m entry, NO filter",    "30m", None),
        ("30m entry, 1h filter",    "30m", "1h"),
        ("30m entry, 2h filter",    "30m", "2h"),
        ("1h entry, NO filter",     "1h",  None),
        ("1h entry, 2h filter",     "1h",  "2h"),
        ("1h entry, 4h filter",     "1h",  "4h"),
    ]

    print("\nComputing HTF trends...")
    htf_cache = {}
    for sym in SYMBOLS:
        d = all_data[sym]
        htf_cache[sym] = {}
        pairs = [
            ("5m", "15m"), ("5m", "30m"), ("5m", "1h"),
            ("15m", "1h"), ("15m", "2h"),
            ("30m", "1h"), ("30m", "2h"),
            ("1h", "2h"), ("1h", "4h"),
        ]
        for entry_key, htf_key in pairs:
            if entry_key in d and htf_key in d:
                if len(d[entry_key]) > 50 and len(d[htf_key]) > 10:
                    cache_key = f"{entry_key}_{htf_key}"
                    htf_cache[sym][cache_key] = compute_htf_trend(d[entry_key], d[htf_key])
    print("  Done.\n")

    print("=" * 70)
    print(f"  RUNNING 13 CONFIGS (all on same {days_common}-day period)")
    print("=" * 70)

    results = []
    for label, entry_tf, htf_tf in configs:
        sym_results = {}
        for sym in SYMBOLS:
            d = all_data[sym]
            if entry_tf not in d or len(d[entry_tf]) < 50:
                continue
            entry_df = d[entry_tf]
            htf_trends = None
            if htf_tf:
                cache_key = f"{entry_tf}_{htf_tf}"
                if cache_key in htf_cache.get(sym, {}):
                    htf_trends = htf_cache[sym][cache_key]
            r = run_backtest(entry_df, htf_trends)
            if r: sym_results[sym] = r

        if not sym_results:
            results.append({"label": label, "trades": 0, "win_rate": 0,
                            "pf": 0, "pnl": 0, "max_dd": 0, "per_sym": {}})
            continue

        total_trades = sum(r["total"] for r in sym_results.values())
        total_wins   = sum(r["wins"]  for r in sym_results.values())
        total_pnl    = sum(r["pnl"]   for r in sym_results.values())
        avg_pf       = sum(r["pf"]    for r in sym_results.values()) / len(sym_results)
        avg_wr       = total_wins / total_trades * 100 if total_trades > 0 else 0
        max_dd       = max(r["max_dd"] for r in sym_results.values())

        results.append({
            "label": label, "trades": total_trades,
            "win_rate": avg_wr, "pf": avg_pf, "pnl": total_pnl,
            "max_dd": max_dd, "per_sym": sym_results,
        })

    for r in results:
        pf_str = f"{r['pf']:.2f}" if 0 < r['pf'] < 999 else ("inf" if r['pf'] >= 999 else "0.00")
        print(f"\n  {r['label']}")
        print(f"  {'─'*55}")
        if r["trades"] == 0:
            print(f"  No trades"); continue
        print(f"  P&L: ${r['pnl']:+.2f}  |  Trades: {r['trades']}  |  "
              f"WR: {r['win_rate']:.1f}%  |  PF: {pf_str}  |  DD: {r['max_dd']*100:.1f}%")
        for sym, sr in r.get("per_sym", {}).items():
            coin = sym.split("/")[0]
            print(f"    {coin:<4}  {sr['total']:>4} trades  WR:{sr['win_rate']:>5.1f}%  "
                  f"PF:{sr['pf']:>6.2f}  P&L:${sr['pnl']:>+9.2f}")

    print(f"\n{'='*70}")
    print(f"  FAIR HEAD-TO-HEAD ({days_common} days, same data)")
    print(f"{'='*70}")
    print(f"  {'Config':<30} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>11} {'DD':>6}")
    print(f"  {'─'*70}")

    for r in sorted(results, key=lambda x: x["pnl"], reverse=True):
        if r["trades"] == 0:
            print(f"  {r['label']:<30} {'no trades':>40}"); continue
        pf_str = f"{r['pf']:.2f}" if 0 < r['pf'] < 999 else "inf"
        current = " ◀ CURRENT" if "1h entry, 2h filter" in r["label"] else ""
        print(f"  {r['label']:<30} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"{pf_str:>8} ${r['pnl']:>+9.2f} {r['max_dd']*100:>5.1f}%{current}")

    print(f"\n{'='*70}")
    print(f"  BEST CONFIG PER ENTRY TIMEFRAME")
    print(f"{'='*70}")
    for tf in ["5m", "15m", "30m", "1h"]:
        tf_results = [r for r in results if r["label"].startswith(tf) and r["trades"] >= 5]
        if tf_results:
            best = max(tf_results, key=lambda r: r["pf"])
            pf_str = f"{best['pf']:.2f}" if best['pf'] < 999 else "inf"
            print(f"\n  Best {tf:<4} config: {best['label']}")
            print(f"    PF: {pf_str}  |  Trades: {best['trades']}  |  "
                  f"WR: {best['win_rate']:.1f}%  |  P&L: ${best['pnl']:+.2f}")

    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")

    viable = [r for r in results if r["trades"] >= 10]
    if viable:
        best_pf  = max(viable, key=lambda r: r["pf"])
        best_pnl = max(viable, key=lambda r: r["pnl"])

        print(f"\n  HIGHEST PROFIT FACTOR (10+ trades):")
        print(f"    {best_pf['label']}")
        pf1 = f"{best_pf['pf']:.2f}" if best_pf['pf'] < 999 else "inf"
        print(f"    PF: {pf1}  |  Trades: {best_pf['trades']}  |  "
              f"WR: {best_pf['win_rate']:.1f}%  |  P&L: ${best_pf['pnl']:+.2f}")

        print(f"\n  HIGHEST TOTAL P&L:")
        print(f"    {best_pnl['label']}")
        pf2 = f"{best_pnl['pf']:.2f}" if best_pnl['pf'] < 999 else "inf"
        print(f"    PF: {pf2}  |  Trades: {best_pnl['trades']}  |  "
              f"WR: {best_pnl['win_rate']:.1f}%  |  P&L: ${best_pnl['pnl']:+.2f}")

        reliable = [r for r in viable if r["trades"] >= 20]
        winner = max(reliable, key=lambda r: r["pf"]) if reliable else best_pf

        print(f"\n  {'─'*55}")
        print(f"  WINNER (best PF with 20+ trades):")
        print(f"    {winner['label']}")
        pf_w = f"{winner['pf']:.2f}" if winner['pf'] < 999 else "inf"
        print(f"    PF: {pf_w}  |  Trades: {winner['trades']}  |  "
              f"WR: {winner['win_rate']:.1f}%  |  P&L: ${winner['pnl']:+.2f}")

        entry_tf = winner["label"].split(" entry")[0]
        if "NO filter" in winner["label"]:
            htf = None
        else:
            htf = winner["label"].split(", ")[1].replace(" filter", "")

        print(f"\n  WHAT TO CHANGE IN bot.py:")
        print(f'    CRYPTO_TF      = "{entry_tf}"')
        if htf:
            print(f'    CRYPTO_HTF     = "{htf}"')
        else:
            print(f'    Remove HTF filter from run_crypto()')

        if entry_tf == "5m":
            print(f'    CHECK_INTERVAL = 60          # every 1 min')
        elif entry_tf == "15m":
            print(f'    CHECK_INTERVAL = 60 * 3      # every 3 min')
        elif entry_tf == "30m":
            print(f'    CHECK_INTERVAL = 60 * 5      # every 5 min')
        else:
            print(f'    CHECK_INTERVAL = 60 * 5      # every 5 min')

    print(f"\n{'='*70}")
    print(f"  Done! {days_common} days of REAL candle data from Binance.")
    print(f"{'='*70}\n")