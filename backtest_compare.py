"""
═══════════════════════════════════════════════════════════
  BACKTEST COMPARISON — 15m/1h vs 1h/2h
  
  Tests your strategy on FASTER timeframes to see if
  15m entry + 1h filter beats your current 1h entry + 2h filter.
  
  Config A:  15m entry, NO filter         (pure 15m signals)
  Config B:  15m entry, 1h filter         (what you asked for)
  Config C:  15m entry, 30m filter        (middle ground)
  Config D:  1h entry, 2h filter          (current best from last test)
  Config E:  1h entry, NO filter          (1h baseline)
  
  Symbols: BTC, ETH, SOL (combined results)
  
  Run:  python3 backtest_compare.py
  Takes about 1-2 minutes.
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ── Strategy settings (matching your bot.py exactly) ────
FAST_EMA     = 7
SLOW_EMA     = 18
RSI_PERIOD   = 14
BB_PERIOD    = 20
BB_STD       = 2.0
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIG     = 9
RSI_OB       = 65
RSI_OS       = 35
SL_PCT       = 0.003    # 0.3%
TP_PCT       = 0.015    # 1.5%
RISK         = 0.02     # 2% per trade
START_BAL    = 10_000.0
MIN_VOL_MULT = 1.2

# ── Symbols to test ────────────────────────────────────
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]


# ═══════════════════════════════════════════════════════
#  DOWNLOAD DATA
# ═══════════════════════════════════════════════════════

def download(symbol, period, interval):
    """Download data from Yahoo Finance, clean columns."""
    raw = yf.download(symbol, period=period, interval=interval, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw.dropna(subset=["close", "high", "low", "volume"])
    return raw


def resample(df, tf):
    """Resample OHLCV data to a larger timeframe."""
    return df.resample(tf).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()


# ═══════════════════════════════════════════════════════
#  ADD INDICATORS (same as bot.py)
# ═══════════════════════════════════════════════════════

def add_indicators(df):
    """Add all 6 indicators matching bot.py exactly."""
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    bb = ta.volatility.BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST,
                          window_slow=MACD_SLOW, window_sign=MACD_SIG)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["vwap"]      = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_avg"] * MIN_VOL_MULT
    df["atr"]       = ta.volatility.average_true_range(
                        df["high"], df["low"], df["close"], window=14)
    return df


# ═══════════════════════════════════════════════════════
#  HTF TREND CALCULATOR
# ═══════════════════════════════════════════════════════

def compute_htf_trend_series(entry_df, htf_df):
    """
    For each bar in entry_df, look up the most recent HTF bar
    and determine if ema_fast > ema_slow on that HTF bar.
    Returns a Series of 'UP' / 'DOWN' / 'NEUTRAL'.
    """
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
            idx = mask.sum() - 1
            trends.append(htf_trends[idx])
        else:
            trends.append("NEUTRAL")

    return pd.Series(trends, index=entry_df.index)


# ═══════════════════════════════════════════════════════
#  BACKTEST ENGINE (matches bot.py logic exactly)
# ═══════════════════════════════════════════════════════

def run_backtest(df, htf_trends=None):
    """
    Run backtest on df with optional HTF trend filter.
    Returns (trades_list, final_balance).
    """
    df = add_indicators(df.copy())
    df = df.dropna()

    balance     = START_BAL
    in_trade    = False
    entry_price = 0.0
    highest     = 0.0
    coin_held   = 0.0
    trades      = []

    for i in range(3, len(df)):
        prev2 = df.iloc[i - 2]
        prev  = df.iloc[i - 1]
        last  = df.iloc[i]
        price = float(last["close"])

        # ── CHECK EXITS FIRST ────────────────────────
        if in_trade:
            if price > highest:
                highest = price
            trail_sl    = highest * (1 - SL_PCT)
            take_profit = entry_price * (1 + TP_PCT)

            if price <= trail_sl or price >= take_profit:
                pnl      = coin_held * (price - entry_price)
                balance += coin_held * price
                reason   = "TP" if price >= take_profit else "SL"
                trades.append({
                    "entry": entry_price, "exit": price,
                    "pnl": pnl, "result": "WIN" if pnl > 0 else "LOSS",
                    "reason": reason, "balance": balance,
                })
                in_trade    = False
                coin_held   = 0.0
                highest     = 0.0
                entry_price = 0.0
            continue

        # ── CHECK BUY SIGNAL ─────────────────────────
        try:
            rsi    = float(last["rsi"])
            macd_h = float(last["macd_hist"])
            macd_v = float(last["macd"])
            macd_s = float(last["macd_sig"])
            vwap   = float(last["vwap"])

            if pd.isna(rsi) or pd.isna(macd_h) or pd.isna(last["bb_lower"]):
                continue

            # 2-candle EMA crossover confirmation
            ema_cross_up = (float(prev2["ema_fast"]) < float(prev2["ema_slow"]) and
                            float(prev["ema_fast"])  > float(prev["ema_slow"])  and
                            float(last["ema_fast"])  > float(last["ema_slow"]))

            # MACD confirmation
            macd_bullish = (macd_v > macd_s and macd_h > 0)

            above_vwap = price > vwap
            vol_spike  = bool(last["vol_spike"])

            buy = (ema_cross_up   and
                   RSI_OS < rsi < RSI_OB and
                   macd_bullish   and
                   above_vwap     and
                   vol_spike)

            # ── HTF FILTER ───────────────────────────
            if buy and htf_trends is not None:
                ts = df.index[i]
                if ts in htf_trends.index:
                    trend = htf_trends.loc[ts]
                else:
                    mask = htf_trends.index <= ts
                    if mask.any():
                        trend = htf_trends[mask].iloc[-1]
                    else:
                        trend = "NEUTRAL"
                if trend != "UP":
                    buy = False

            if buy:
                spend       = balance * RISK
                coin_held   = spend / price
                balance    -= spend
                entry_price = price
                highest     = price
                in_trade    = True

        except Exception:
            continue

    # Close any open trade at end
    if in_trade:
        price    = float(df.iloc[-1]["close"])
        pnl      = coin_held * (price - entry_price)
        balance += coin_held * price
        trades.append({
            "entry": entry_price, "exit": price,
            "pnl": pnl, "result": "WIN" if pnl > 0 else "LOSS",
            "reason": "END", "balance": balance,
        })

    return trades, balance


# ═══════════════════════════════════════════════════════
#  RESULTS HELPERS
# ═══════════════════════════════════════════════════════

def calc_stats(trades):
    """Calculate stats from a list of trade dicts."""
    if not trades:
        return {"trades": 0, "win_rate": 0, "profit_factor": 0,
                "pnl": 0, "max_dd": 0, "avg_win": 0, "avg_loss": 0}
    tdf     = pd.DataFrame(trades)
    wins    = tdf[tdf["result"] == "WIN"]
    losses  = tdf[tdf["result"] == "LOSS"]
    wr      = len(wins) / len(tdf) * 100
    pnl     = tdf["pnl"].sum()
    avg_w   = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_l   = losses["pnl"].mean() if len(losses) > 0 else 0
    gross_w = wins["pnl"].sum()    if len(wins)   > 0 else 0
    gross_l = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf      = gross_w / gross_l if gross_l > 0 else float("inf")
    tp_ct   = len(tdf[tdf["reason"] == "TP"])
    sl_ct   = len(tdf[tdf["reason"] == "SL"])

    peak   = START_BAL
    max_dd = 0.0
    for bal in tdf["balance"]:
        if bal > peak: peak = bal
        dd = (peak - bal) / peak
        if dd > max_dd: max_dd = dd

    return {"trades": len(tdf), "win_rate": wr, "profit_factor": pf,
            "pnl": pnl, "max_dd": max_dd, "avg_win": avg_w,
            "avg_loss": avg_l, "tp": tp_ct, "sl": sl_ct}


def print_config_results(label, all_symbol_stats):
    """Print combined results for one config across all symbols."""
    total_trades = sum(s["trades"] for s in all_symbol_stats.values())
    if total_trades == 0:
        print(f"\n  {label}")
        print(f"  {'─'*55}")
        print(f"  No trades generated across any symbol.")
        return {"label": label, "trades": 0, "win_rate": 0,
                "profit_factor": 0, "pnl": 0, "max_dd": 0}

    total_pnl    = sum(s["pnl"]     for s in all_symbol_stats.values())
    total_tp     = sum(s["tp"]      for s in all_symbol_stats.values())
    total_sl     = sum(s["sl"]      for s in all_symbol_stats.values())
    max_dd       = max(s["max_dd"]  for s in all_symbol_stats.values())

    # Weighted averages
    all_wins   = sum(int(s["trades"] * s["win_rate"] / 100) for s in all_symbol_stats.values())
    all_losses = total_trades - all_wins
    wr         = all_wins / total_trades * 100 if total_trades > 0 else 0

    # Combined profit factor
    all_gross_w = sum(s["avg_win"]  * int(s["trades"] * s["win_rate"] / 100)
                      for s in all_symbol_stats.values() if s["trades"] > 0)
    all_gross_l = sum(abs(s["avg_loss"]) * (s["trades"] - int(s["trades"] * s["win_rate"] / 100))
                      for s in all_symbol_stats.values() if s["trades"] > 0)
    pf = all_gross_w / all_gross_l if all_gross_l > 0 else float("inf")

    print(f"\n  {label}")
    print(f"  {'─'*55}")
    print(f"  Total P&L:      ${total_pnl:+.2f}  across {len(all_symbol_stats)} coins")
    print(f"  Total trades:   {total_trades}  ({total_tp} TP / {total_sl} SL / "
          f"{total_trades - total_tp - total_sl} other)")
    print(f"  Win rate:       {wr:.1f}%  ({all_wins}W / {all_losses}L)")
    print(f"  Profit factor:  {pf:.2f}")
    print(f"  Max drawdown:   {max_dd*100:.1f}%")

    # Per-symbol breakdown
    for sym, st in all_symbol_stats.items():
        coin = sym.replace("-USD", "")
        if st["trades"] > 0:
            print(f"    {coin:<4}  {st['trades']:>3} trades  "
                  f"WR:{st['win_rate']:>5.1f}%  "
                  f"PF:{st['profit_factor']:>5.2f}  "
                  f"P&L:${st['pnl']:>+7.2f}")
        else:
            print(f"    {coin:<4}  no trades")

    return {"label": label, "trades": total_trades, "win_rate": wr,
            "profit_factor": pf, "pnl": total_pnl, "max_dd": max_dd}


# ═══════════════════════════════════════════════════════
#  MAIN — RUN ALL CONFIGS ON ALL SYMBOLS
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  BACKTEST COMPARISON — 15m/1h vs 1h/2h")
    print(f"  Coins: {', '.join(s.replace('-USD','') for s in SYMBOLS)}")
    print(f"  Strategy: EMA {FAST_EMA}/{SLOW_EMA} + RSI + MACD + BB + VWAP")
    print(f"  SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%  |  Risk: {RISK*100:.0f}%")
    print("=" * 62)

    # ── Download data for all symbols ──────────────────
    print("\nDownloading data from Yahoo Finance...")
    print("  (15m data = ~60 days  |  1h data = ~2 years)")

    all_data = {}
    for sym in SYMBOLS:
        coin = sym.replace("-USD", "")
        print(f"\n  {coin}:")

        # 15m candles (~60 days max on Yahoo)
        df_15m = download(sym, period="60d", interval="15m")
        print(f"    15m: {len(df_15m)} candles ({len(df_15m)//96:.0f} days)")

        # 1h candles (~2 years max on Yahoo)
        df_1h = download(sym, period="2y", interval="1h")
        print(f"    1h:  {len(df_1h)} candles ({len(df_1h)//24:.0f} days)")

        # Resample 15m → 30m and 15m → 1h for HTF filters
        df_30m = resample(df_15m, "30min") if len(df_15m) > 0 else pd.DataFrame()
        df_1h_from_15m = resample(df_15m, "1h") if len(df_15m) > 0 else pd.DataFrame()

        # Resample 1h → 2h for the current best config
        df_2h = resample(df_1h, "2h") if len(df_1h) > 0 else pd.DataFrame()

        if len(df_15m) > 0:
            print(f"    30m: {len(df_30m)} candles (resampled)")
            print(f"    1h from 15m: {len(df_1h_from_15m)} candles (resampled)")
        if len(df_1h) > 0:
            print(f"    2h:  {len(df_2h)} candles (resampled)")

        all_data[sym] = {
            "15m": df_15m,
            "30m": df_30m,
            "1h_from_15m": df_1h_from_15m,
            "1h": df_1h,
            "2h": df_2h,
        }

    # ── Compute HTF trends ────────────────────────────
    print("\nComputing HTF trends for all configs...")

    htf_cache = {}
    for sym in SYMBOLS:
        d = all_data[sym]
        htf_cache[sym] = {}

        # 15m entry → 1h filter
        if len(d["15m"]) > 100 and len(d["1h_from_15m"]) > 30:
            htf_cache[sym]["15m_1h"] = compute_htf_trend_series(d["15m"], d["1h_from_15m"])

        # 15m entry → 30m filter
        if len(d["15m"]) > 100 and len(d["30m"]) > 30:
            htf_cache[sym]["15m_30m"] = compute_htf_trend_series(d["15m"], d["30m"])

        # 1h entry → 2h filter (current best)
        if len(d["1h"]) > 100 and len(d["2h"]) > 30:
            htf_cache[sym]["1h_2h"] = compute_htf_trend_series(d["1h"], d["2h"])

    print("  Done.\n")

    # ── Define configs ────────────────────────────────
    configs = [
        ("CONFIG A — 15m entry, NO filter",         "15m", None),
        ("CONFIG B — 15m entry, 1h filter",          "15m", "15m_1h"),
        ("CONFIG C — 15m entry, 30m filter",         "15m", "15m_30m"),
        ("CONFIG D — 1h entry, 2h filter (CURRENT)", "1h",  "1h_2h"),
        ("CONFIG E — 1h entry, NO filter",           "1h",  None),
    ]

    # ── Run all configs on all symbols ────────────────
    print("=" * 62)
    print("  RUNNING 5 CONFIGURATIONS x 3 COINS...")
    print("=" * 62)

    all_results = []

    for config_label, entry_tf, htf_key in configs:
        symbol_stats = {}

        for sym in SYMBOLS:
            d = all_data[sym]

            # Pick entry dataframe
            if entry_tf == "15m":
                entry_df = d["15m"]
            else:
                entry_df = d["1h"]

            if len(entry_df) < 100:
                symbol_stats[sym] = {"trades": 0, "win_rate": 0,
                    "profit_factor": 0, "pnl": 0, "max_dd": 0,
                    "avg_win": 0, "avg_loss": 0, "tp": 0, "sl": 0}
                continue

            # Pick HTF trends (or None)
            htf_trends = None
            if htf_key and htf_key in htf_cache.get(sym, {}):
                htf_trends = htf_cache[sym][htf_key]

            trades, balance = run_backtest(entry_df, htf_trends=htf_trends)
            symbol_stats[sym] = calc_stats(trades)

        result = print_config_results(config_label, symbol_stats)
        all_results.append(result)

    # ── COMPARISON TABLE ──────────────────────────────
    print(f"\n{'='*62}")
    print(f"  HEAD-TO-HEAD COMPARISON — ALL COINS COMBINED")
    print(f"{'='*62}")
    print(f"  {'Config':<42} {'Trades':>7} {'WR':>7} {'PF':>7} {'P&L':>10} {'DD':>6}")
    print(f"  {'─'*80}")

    for r in all_results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 999 else "inf"
        print(f"  {r['label'][:42]:<42} {r['trades']:>7} "
              f"{r['win_rate']:>6.1f}% {pf_str:>7} "
              f"${r['pnl']:>+8.2f} {r['max_dd']*100:>5.1f}%")

    # ── RECOMMENDATION ────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  RECOMMENDATION")
    print(f"{'='*62}")

    valid = [r for r in all_results if r["trades"] >= 5]
    if not valid:
        print("\n  Not enough trades in any config. Try looser settings.")
    else:
        # Find best by profit factor
        best = max(valid, key=lambda r: r["profit_factor"])

        # Get specific configs for comparison
        a = next((r for r in all_results if "CONFIG A" in r["label"]), None)
        b = next((r for r in all_results if "CONFIG B" in r["label"]), None)
        d = next((r for r in all_results if "CONFIG D" in r["label"]), None)
        e = next((r for r in all_results if "CONFIG E" in r["label"]), None)

        # 15m vs 1h comparison
        best_15m = max([r for r in valid if "15m" in r["label"]],
                       key=lambda r: r["profit_factor"], default=None)
        best_1h  = max([r for r in valid if "1h" in r["label"]],
                       key=lambda r: r["profit_factor"], default=None)

        print(f"\n  BEST OVERALL: {best['label']}")
        print(f"  PF: {best['profit_factor']:.2f}  |  WR: {best['win_rate']:.1f}%  |  "
              f"Trades: {best['trades']}  |  P&L: ${best['pnl']:+.2f}")

        if best_15m and best_1h:
            print(f"\n  15m vs 1h TIMEFRAME COMPARISON:")
            print(f"    Best 15m config: {best_15m['label']}")
            print(f"      PF: {best_15m['profit_factor']:.2f}  |  "
                  f"Trades: {best_15m['trades']}  |  P&L: ${best_15m['pnl']:+.2f}")
            print(f"    Best 1h config:  {best_1h['label']}")
            print(f"      PF: {best_1h['profit_factor']:.2f}  |  "
                  f"Trades: {best_1h['trades']}  |  P&L: ${best_1h['pnl']:+.2f}")

            if best_15m["profit_factor"] > best_1h["profit_factor"] * 1.15:
                print(f"\n  ✅ 15m is SIGNIFICANTLY BETTER than 1h")
                print(f"     Switch bot.py to 15m entry timeframe.")
            elif best_1h["profit_factor"] > best_15m["profit_factor"] * 1.15:
                print(f"\n  ✅ 1h is SIGNIFICANTLY BETTER than 15m")
                print(f"     Keep bot.py on 1h entry timeframe.")
            else:
                print(f"\n  🟡 Similar performance — 15m gives more trades, 1h is calmer")
                print(f"     15m = more active, more fees, faster signals")
                print(f"     1h  = fewer trades, proven over 2 years of data")

        # Data coverage warning
        print(f"\n  ⚠️  DATA COVERAGE NOTE:")
        print(f"     15m configs tested on ~60 days of data")
        print(f"     1h  configs tested on ~2 years of data")
        print(f"     1h results are MORE RELIABLE due to larger sample size.")
        print(f"     15m results may not hold up long-term.")

        # Final action
        print(f"\n  {'─'*55}")
        print(f"  WHAT TO CHANGE IN bot.py:")
        if "15m" in best["label"] and best["profit_factor"] > 1.5:
            if "1h filter" in best["label"]:
                print(f'  1. CRYPTO_TF   = "15m"')
                print(f'  2. CRYPTO_HTF  = "1h"')
                print(f'  3. CHECK_INTERVAL = 60 * 5   # check every 5 min')
            elif "30m filter" in best["label"]:
                print(f'  1. CRYPTO_TF   = "15m"')
                print(f'  2. CRYPTO_HTF  = "30m"  # note: ccxt may not support 30m')
                print(f'  3. CHECK_INTERVAL = 60 * 5')
            elif "NO filter" in best["label"]:
                print(f'  1. CRYPTO_TF   = "15m"')
                print(f'  2. Remove HTF filter from run_crypto()')
                print(f'  3. CHECK_INTERVAL = 60 * 5')
        elif "1h" in best["label"]:
            if "2h filter" in best["label"]:
                print(f'  Keep current settings — 1h entry + 2h filter is best.')
                print(f'  CRYPTO_TF  = "1h"')
                print(f'  CRYPTO_HTF = "2h"')
            elif "NO filter" in best["label"]:
                print(f'  1. CRYPTO_TF  = "1h"')
                print(f'  2. Remove HTF filter from run_crypto()')
        else:
            print(f"  No clear winner — keep current settings and monitor.")

    print(f"\n{'='*62}")
    print(f"  Done! Compare results and update bot.py accordingly.")
    print(f"{'='*62}\n")