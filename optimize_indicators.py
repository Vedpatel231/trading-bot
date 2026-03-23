"""
═══════════════════════════════════════════════════════════
  INDICATOR COMBO OPTIMIZER
  
  Tests every combination of indicators to find which
  ones actually help and which ones are blocking good trades.
  
  Tests: EMA (always on) + optional RSI / MACD / VWAP / Volume / HTF
  Also tests different RSI ranges (35-65, 25-75, 20-80, off)
  
  Symbols: ETH + SOL (matching your bot)
  Timeframe: 1h with 2h HTF (matching your bot)
  
  Run:  python3 optimize_indicators.py
  Takes about 2-3 minutes.
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import yfinance as yf
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ── Fixed settings ──────────────────────────────────────
SYMBOLS      = ["ETH-USD", "SOL-USD"]
FAST_EMA     = 7
SLOW_EMA     = 18
BB_PERIOD    = 20
BB_STD       = 2.0
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIG     = 9
SL_PCT       = 0.003
TP_PCT       = 0.015
RISK         = 0.02
START_BAL    = 10_000.0
MIN_VOL_MULT = 1.2


# ── Download data ───────────────────────────────────────

def download(symbol, period, interval):
    raw = yf.download(symbol, period=period, interval=interval, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    return raw.dropna(subset=["close", "high", "low", "volume"])


def resample(df, tf):
    return df.resample(tf).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()


# ── Add indicators ──────────────────────────────────────

def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=14)

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


# ── Backtest engine ─────────────────────────────────────

def run_backtest(df, htf_trends, config):
    """
    config dict:
      use_rsi:    True/False
      rsi_low:    lower bound (e.g. 25)
      rsi_high:   upper bound (e.g. 75)
      use_macd:   True/False
      use_vwap:   True/False
      use_vol:    True/False
      use_htf:    True/False
      confirm_2:  True/False (2-candle confirmation)
    """
    df = add_indicators(df.copy())
    df = df.dropna()

    balance     = START_BAL
    in_trade    = False
    entry_price = 0.0
    highest     = 0.0
    coin_held   = 0.0
    wins        = 0
    losses      = 0
    gross_wins  = 0.0
    gross_losses = 0.0

    start_idx = 3 if config["confirm_2"] else 2

    for i in range(start_idx, len(df)):
        prev2 = df.iloc[i - 2]
        prev  = df.iloc[i - 1]
        last  = df.iloc[i]
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
                in_trade = False; coin_held = 0.0; highest = 0.0
            continue

        # Check buy signal
        try:
            rsi    = float(last["rsi"])
            macd_h = float(last["macd_hist"])
            macd_v = float(last["macd"])
            macd_s = float(last["macd_sig"])
            vwap   = float(last["vwap"])

            if pd.isna(rsi) or pd.isna(macd_h):
                continue

            # EMA crossover (always required)
            if config["confirm_2"]:
                ema_cross = (float(prev2["ema_fast"]) < float(prev2["ema_slow"]) and
                             float(prev["ema_fast"])  > float(prev["ema_slow"])  and
                             float(last["ema_fast"])  > float(last["ema_slow"]))
            else:
                ema_cross = (float(prev["ema_fast"])  < float(prev["ema_slow"]) and
                             float(last["ema_fast"])  > float(last["ema_slow"]))

            if not ema_cross:
                continue

            # Optional filters
            if config["use_rsi"]:
                if not (config["rsi_low"] < rsi < config["rsi_high"]):
                    continue

            if config["use_macd"]:
                if not (macd_v > macd_s and macd_h > 0):
                    continue

            if config["use_vwap"]:
                if not (price > vwap):
                    continue

            if config["use_vol"]:
                if not bool(last["vol_spike"]):
                    continue

            if config["use_htf"] and htf_trends is not None:
                ts = df.index[i]
                if ts in htf_trends.index:
                    trend = htf_trends.loc[ts]
                else:
                    mask = htf_trends.index <= ts
                    trend = htf_trends[mask].iloc[-1] if mask.any() else "NEUTRAL"
                if trend != "UP":
                    continue

            # All filters passed — buy
            spend       = balance * RISK
            coin_held   = spend / price
            balance    -= spend
            entry_price = price
            highest     = price
            in_trade    = True

        except Exception:
            continue

    # Close open trade
    if in_trade:
        price    = float(df.iloc[-1]["close"])
        pnl      = coin_held * (price - entry_price)
        balance += coin_held * price
        if pnl > 0:
            wins += 1; gross_wins += pnl
        else:
            losses += 1; gross_losses += abs(pnl)

    total = wins + losses
    if total < 3:
        return None

    return {
        "wins": wins, "losses": losses,
        "total": total,
        "win_rate": wins / total * 100,
        "pf": gross_wins / gross_losses if gross_losses > 0 else 999,
        "pnl": balance - START_BAL,
        "balance": balance,
    }


# ── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  INDICATOR COMBO OPTIMIZER")
    print(f"  Coins: {', '.join(s.replace('-USD','') for s in SYMBOLS)}")
    print(f"  EMA: {FAST_EMA}/{SLOW_EMA}  |  SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%")
    print(f"  Timeframe: 1h  |  HTF: 2h")
    print("=" * 65)

    # Download data
    print("\nDownloading data...")
    all_data = {}
    for sym in SYMBOLS:
        coin = sym.replace("-USD", "")
        df_1h = download(sym, period="2y", interval="1h")
        df_2h = resample(df_1h, "2h") if len(df_1h) > 0 else pd.DataFrame()
        htf   = compute_htf_trend(df_1h, df_2h) if len(df_2h) > 30 else None
        all_data[sym] = {"1h": df_1h, "htf": htf}
        print(f"  {coin}: {len(df_1h)} candles")

    # Define all configurations to test
    rsi_options = [
        {"use_rsi": False, "rsi_low": 0, "rsi_high": 100, "rsi_label": "OFF"},
        {"use_rsi": True,  "rsi_low": 20, "rsi_high": 80, "rsi_label": "20-80"},
        {"use_rsi": True,  "rsi_low": 25, "rsi_high": 75, "rsi_label": "25-75"},
        {"use_rsi": True,  "rsi_low": 30, "rsi_high": 70, "rsi_label": "30-70"},
        {"use_rsi": True,  "rsi_low": 35, "rsi_high": 65, "rsi_label": "35-65"},
    ]

    macd_options   = [True, False]
    vwap_options   = [True, False]
    vol_options    = [True, False]
    htf_options    = [True, False]
    confirm_options = [True, False]

    total_combos = (len(rsi_options) * len(macd_options) * len(vwap_options) *
                    len(vol_options) * len(htf_options) * len(confirm_options))

    print(f"\nTesting {total_combos} indicator combinations...")
    print("This takes about 2-3 minutes...\n")

    all_results = []
    tested = 0

    for rsi_opt in rsi_options:
        for use_macd in macd_options:
            for use_vwap in vwap_options:
                for use_vol in vol_options:
                    for use_htf in htf_options:
                        for confirm_2 in confirm_options:
                            config = {
                                "use_rsi":   rsi_opt["use_rsi"],
                                "rsi_low":   rsi_opt["rsi_low"],
                                "rsi_high":  rsi_opt["rsi_high"],
                                "use_macd":  use_macd,
                                "use_vwap":  use_vwap,
                                "use_vol":   use_vol,
                                "use_htf":   use_htf,
                                "confirm_2": confirm_2,
                            }

                            sym_results = []
                            for sym in SYMBOLS:
                                d = all_data[sym]
                                r = run_backtest(d["1h"], d["htf"], config)
                                if r:
                                    sym_results.append(r)

                            if len(sym_results) == len(SYMBOLS):
                                total_trades = sum(r["total"] for r in sym_results)
                                total_wins   = sum(r["wins"]  for r in sym_results)
                                total_losses = sum(r["losses"] for r in sym_results)
                                total_pnl    = sum(r["pnl"]   for r in sym_results)
                                total_gw     = sum(r["wins"] * (r["pnl"] / r["total"] if r["total"] > 0 else 0)
                                                   for r in sym_results)
                                avg_wr       = total_wins / total_trades * 100 if total_trades > 0 else 0
                                avg_pf       = sum(r["pf"] for r in sym_results) / len(sym_results)

                                # Build label
                                parts = ["EMA"]
                                if rsi_opt["use_rsi"]:
                                    parts.append(f"RSI({rsi_opt['rsi_label']})")
                                if use_macd:
                                    parts.append("MACD")
                                if use_vwap:
                                    parts.append("VWAP")
                                if use_vol:
                                    parts.append("Vol")
                                if use_htf:
                                    parts.append("2h")
                                if confirm_2:
                                    parts.append("2bar")

                                label = " + ".join(parts)

                                all_results.append({
                                    "label":    label,
                                    "trades":   total_trades,
                                    "win_rate": avg_wr,
                                    "pf":       avg_pf,
                                    "pnl":      total_pnl,
                                    "config":   config,
                                    "rsi_label": rsi_opt["rsi_label"],
                                })

                            tested += 1
                            if tested % 50 == 0:
                                print(f"  Tested {tested}/{total_combos} combos...")

    print(f"  Done! Tested {tested} combinations.\n")

    if not all_results:
        print("No valid results found.")
        exit()

    # Sort by score = profit_factor * sqrt(trades)
    # This balances quality (PF) with quantity (enough trades to be reliable)
    import math
    for r in all_results:
        r["score"] = r["pf"] * math.sqrt(r["trades"]) if r["trades"] >= 5 else 0

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("score", ascending=False)

    # ── TOP 20 RESULTS ────────────────────────────────
    print("=" * 90)
    print("  TOP 20 INDICATOR COMBINATIONS (sorted by score = PF × √trades)")
    print("=" * 90)
    print(f"  {'#':<3} {'Indicators':<45} {'Trades':>7} {'WR':>7} {'PF':>7} {'P&L':>10} {'Score':>7}")
    print(f"  {'─'*85}")

    for i, (_, row) in enumerate(df_results.head(20).iterrows()):
        pf_str = f"{row['pf']:.2f}" if row['pf'] < 999 else "inf"
        marker = " ◀ CURRENT" if row["label"] == "EMA + RSI(35-65) + MACD + VWAP + Vol + 2h + 2bar" else ""
        print(f"  {i+1:<3} {row['label']:<45} {row['trades']:>7} "
              f"{row['win_rate']:>6.1f}% {pf_str:>7} "
              f"${row['pnl']:>+8.2f} {row['score']:>7.1f}{marker}")

    # ── Find current settings ─────────────────────────
    current_label = "EMA + RSI(35-65) + MACD + VWAP + Vol + 2h + 2bar"
    current = df_results[df_results["label"] == current_label]

    print(f"\n{'='*90}")
    print(f"  YOUR CURRENT SETTINGS vs BEST FOUND")
    print(f"{'='*90}")

    if not current.empty:
        cur = current.iloc[0]
        print(f"\n  CURRENT:  {cur['label']}")
        print(f"            Trades: {cur['trades']}  |  WR: {cur['win_rate']:.1f}%  |  "
              f"PF: {cur['pf']:.2f}  |  P&L: ${cur['pnl']:+.2f}")
    else:
        print(f"\n  CURRENT:  (exact match not found in results)")

    best = df_results.iloc[0]
    print(f"\n  BEST:     {best['label']}")
    print(f"            Trades: {best['trades']}  |  WR: {best['win_rate']:.1f}%  |  "
          f"PF: {best['pf']:.2f}  |  P&L: ${best['pnl']:+.2f}")

    # ── RECOMMENDATION ────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  RECOMMENDATION")
    print(f"{'='*90}")

    cfg = best["config"]
    print(f"\n  Use these indicators:")
    print(f"    EMA {FAST_EMA}/{SLOW_EMA} crossover     — ALWAYS ON (core signal)")

    if cfg["confirm_2"]:
        print(f"    2-candle confirmation   — ON  (wait for crossover to hold)")
    else:
        print(f"    2-candle confirmation   — OFF (trade on first crossover)")

    if cfg["use_rsi"]:
        print(f"    RSI filter              — ON  (range: {cfg['rsi_low']}-{cfg['rsi_high']})")
    else:
        print(f"    RSI filter              — OFF (removed)")

    if cfg["use_macd"]:
        print(f"    MACD confirmation       — ON")
    else:
        print(f"    MACD confirmation       — OFF (removed)")

    if cfg["use_vwap"]:
        print(f"    VWAP filter             — ON  (price must be above VWAP)")
    else:
        print(f"    VWAP filter             — OFF (removed)")

    if cfg["use_vol"]:
        print(f"    Volume spike filter     — ON  (volume > 1.2x avg)")
    else:
        print(f"    Volume spike filter     — OFF (removed)")

    if cfg["use_htf"]:
        print(f"    2h trend filter         — ON  (2h EMA must be UP)")
    else:
        print(f"    2h trend filter         — OFF (removed)")

    # ── Code changes ──────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  WHAT TO CHANGE IN bot.py get_signal() function:")
    print()

    if not cfg["use_rsi"]:
        print(f"    Remove:  RSI_OS < rsi < RSI_OB")
    elif cfg["rsi_low"] != 35 or cfg["rsi_high"] != 65:
        print(f"    Change:  RSI_OB = {cfg['rsi_high']}  (was 65)")
        print(f"    Change:  RSI_OS = {cfg['rsi_low']}   (was 35)")

    if not cfg["use_macd"]:
        print(f"    Remove:  macd_bullish condition from buy signal")

    if not cfg["use_vwap"]:
        print(f"    Remove:  above_vwap condition from buy signal")

    if not cfg["use_vol"]:
        print(f"    Remove:  vol_spike condition from buy signal")

    if not cfg["use_htf"]:
        print(f"    Remove:  trend == 'UP' check in run_crypto()")

    if not cfg["confirm_2"]:
        print(f"    Change:  Use 1-candle crossover (remove prev2 check)")

    # ── Analysis: which indicators help vs hurt ───────
    print(f"\n{'='*90}")
    print(f"  INDICATOR IMPACT ANALYSIS")
    print(f"  (average PF when indicator is ON vs OFF)")
    print(f"{'='*90}")

    for indicator, key in [("RSI (any range)", "use_rsi"),
                            ("MACD", "use_macd"),
                            ("VWAP", "use_vwap"),
                            ("Volume spike", "use_vol"),
                            ("2h HTF filter", "use_htf"),
                            ("2-candle confirm", "confirm_2")]:
        on_results  = [r for r in all_results if r["config"][key] == True and r["trades"] >= 5]
        off_results = [r for r in all_results if r["config"][key] == False and r["trades"] >= 5]

        avg_pf_on  = sum(r["pf"] for r in on_results)  / len(on_results)  if on_results  else 0
        avg_pf_off = sum(r["pf"] for r in off_results) / len(off_results) if off_results else 0
        avg_trades_on  = sum(r["trades"] for r in on_results)  / len(on_results)  if on_results  else 0
        avg_trades_off = sum(r["trades"] for r in off_results) / len(off_results) if off_results else 0

        if avg_pf_on > avg_pf_off * 1.05:
            verdict = "HELPS  ✅"
        elif avg_pf_off > avg_pf_on * 1.05:
            verdict = "HURTS  ❌"
        else:
            verdict = "NEUTRAL ─"

        print(f"  {indicator:<22}  ON: PF {avg_pf_on:>5.2f} ({avg_trades_on:>5.0f} trades)  |  "
              f"OFF: PF {avg_pf_off:>5.2f} ({avg_trades_off:>5.0f} trades)  |  {verdict}")

    # RSI range comparison
    print(f"\n  RSI RANGE COMPARISON:")
    for rsi_label in ["OFF", "20-80", "25-75", "30-70", "35-65"]:
        rsi_results = [r for r in all_results if r["rsi_label"] == rsi_label and r["trades"] >= 5]
        if rsi_results:
            avg_pf = sum(r["pf"] for r in rsi_results) / len(rsi_results)
            avg_tr = sum(r["trades"] for r in rsi_results) / len(rsi_results)
            avg_pnl = sum(r["pnl"] for r in rsi_results) / len(rsi_results)
            marker = " ◀ current" if rsi_label == "35-65" else ""
            print(f"    RSI {rsi_label:<6}  PF: {avg_pf:>5.2f}  |  "
                  f"Avg trades: {avg_tr:>5.0f}  |  Avg P&L: ${avg_pnl:>+7.2f}{marker}")

    print(f"\n{'='*90}")
    print(f"  Done! Update bot.py based on the recommendation above.")
    print(f"{'='*90}\n")
