"""
═══════════════════════════════════════════════════════════
  STOCKS INDICATOR COMBO OPTIMIZER
  
  Tests every combination of indicators for SPY + QQQ
  to find which ones help and which ones block good trades.
  
  Tests: EMA (always on) + optional RSI / MACD / VWAP / 
         Volume / 200EMA / 2-candle confirm
  Also tests different RSI ranges and EMA combos
  
  Run:  python3 optimize_stocks.py
  Takes about 2-3 minutes.
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import yfinance as yf
import math
import warnings
warnings.filterwarnings("ignore")

# ── Fixed settings ──────────────────────────────────────
SYMBOLS      = ["SPY", "QQQ"]
SL_PCT       = 0.003
TP_PCT       = 0.015
RISK         = 0.02
START_BAL    = 10_000.0
MIN_VOL_MULT = 1.2
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIG     = 9


# ── Download data ───────────────────────────────────────

def download(symbol):
    raw = yf.download(symbol, period="5y", interval="1d", progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    return raw.dropna(subset=["close", "high", "low", "volume"])


# ── Add indicators ──────────────────────────────────────

def add_indicators(df, fast_ema, slow_ema):
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=fast_ema)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=slow_ema)
    df["ema_200"]  = ta.trend.ema_indicator(df["close"], window=200)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=14)

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


# ── Backtest engine ─────────────────────────────────────

def run_backtest(df, fast_ema, slow_ema, config):
    """
    config dict:
      use_rsi, rsi_low, rsi_high
      use_macd, use_vwap, use_vol
      use_200ema      — price must be above 200 EMA
      confirm_2       — 2-candle confirmation
    """
    df = add_indicators(df.copy(), fast_ema, slow_ema)
    df = df.dropna()

    balance      = START_BAL
    in_trade     = False
    entry_price  = 0.0
    highest      = 0.0
    shares_held  = 0.0
    wins         = 0
    losses       = 0
    gross_wins   = 0.0
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
                pnl      = shares_held * (price - entry_price)
                balance += shares_held * price
                if pnl > 0:
                    wins += 1; gross_wins += pnl
                else:
                    losses += 1; gross_losses += abs(pnl)
                in_trade = False; shares_held = 0.0; highest = 0.0
            continue

        # Check buy signal
        try:
            rsi    = float(last["rsi"])
            macd_h = float(last["macd_hist"])
            macd_v = float(last["macd"])
            macd_s = float(last["macd_sig"])
            e200   = float(last["ema_200"])

            if pd.isna(rsi) or pd.isna(macd_h) or pd.isna(e200):
                continue

            # EMA crossover (always required)
            if config["confirm_2"]:
                ema_cross = (float(prev2["ema_fast"]) < float(prev2["ema_slow"]) and
                             float(prev["ema_fast"])  > float(prev["ema_slow"])  and
                             float(last["ema_fast"])  > float(last["ema_slow"]))
            else:
                ema_cross = (float(prev["ema_fast"]) < float(prev["ema_slow"]) and
                             float(last["ema_fast"]) > float(last["ema_slow"]))

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
                if not (price > float(last["vwap"])):
                    continue

            if config["use_vol"]:
                if not bool(last["vol_spike"]):
                    continue

            if config["use_200ema"]:
                if not (price > e200):
                    continue

            # All filters passed — buy
            spend       = balance * RISK
            shares_held = spend / price
            balance    -= spend
            entry_price = price
            highest     = price
            in_trade    = True

        except Exception:
            continue

    # Close open trade
    if in_trade:
        price    = float(df.iloc[-1]["close"])
        pnl      = shares_held * (price - entry_price)
        balance += shares_held * price
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
    }


# ── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  STOCKS INDICATOR COMBO OPTIMIZER")
    print(f"  Stocks: {', '.join(SYMBOLS)}")
    print(f"  SL: {SL_PCT*100:.1f}%  |  TP: {TP_PCT*100:.1f}%")
    print(f"  Data: Daily candles, 5 years")
    print("=" * 65)

    # Download data
    print("\nDownloading data...")
    all_data = {}
    for sym in SYMBOLS:
        df = download(sym)
        all_data[sym] = df
        print(f"  {sym}: {len(df)} candles ({len(df)//252:.0f} years)")

    # Define all configurations to test
    ema_options = [
        (10, 50, "10/50"),
        (7, 18, "7/18"),
        (9, 21, "9/21"),
        (12, 26, "12/26"),
        (5, 20, "5/20"),
    ]

    rsi_options = [
        {"use_rsi": False, "rsi_low": 0, "rsi_high": 100, "rsi_label": "OFF"},
        {"use_rsi": True,  "rsi_low": 25, "rsi_high": 75, "rsi_label": "25-75"},
        {"use_rsi": True,  "rsi_low": 35, "rsi_high": 65, "rsi_label": "35-65"},
    ]

    macd_options    = [True, False]
    vwap_options    = [True, False]
    vol_options     = [True, False]
    ema200_options  = [True, False]
    confirm_options = [True, False]

    total_combos = (len(ema_options) * len(rsi_options) * len(macd_options) *
                    len(vwap_options) * len(vol_options) * len(ema200_options) *
                    len(confirm_options))

    print(f"\nTesting {total_combos} combinations...")
    print("This takes about 3-5 minutes...\n")

    all_results = []
    tested = 0

    for fast_ema, slow_ema, ema_label in ema_options:
        for rsi_opt in rsi_options:
            for use_macd in macd_options:
                for use_vwap in vwap_options:
                    for use_vol in vol_options:
                        for use_200ema in ema200_options:
                            for confirm_2 in confirm_options:
                                config = {
                                    "use_rsi":    rsi_opt["use_rsi"],
                                    "rsi_low":    rsi_opt["rsi_low"],
                                    "rsi_high":   rsi_opt["rsi_high"],
                                    "use_macd":   use_macd,
                                    "use_vwap":   use_vwap,
                                    "use_vol":    use_vol,
                                    "use_200ema": use_200ema,
                                    "confirm_2":  confirm_2,
                                }

                                sym_results = []
                                for sym in SYMBOLS:
                                    r = run_backtest(all_data[sym], fast_ema, slow_ema, config)
                                    if r:
                                        sym_results.append(r)

                                if len(sym_results) == len(SYMBOLS):
                                    total_trades = sum(r["total"] for r in sym_results)
                                    total_wins   = sum(r["wins"]  for r in sym_results)
                                    total_pnl    = sum(r["pnl"]   for r in sym_results)
                                    avg_wr       = total_wins / total_trades * 100 if total_trades > 0 else 0
                                    avg_pf       = sum(r["pf"] for r in sym_results) / len(sym_results)

                                    # Build label
                                    parts = [f"EMA({ema_label})"]
                                    if rsi_opt["use_rsi"]:
                                        parts.append(f"RSI({rsi_opt['rsi_label']})")
                                    if use_macd:
                                        parts.append("MACD")
                                    if use_vwap:
                                        parts.append("VWAP")
                                    if use_vol:
                                        parts.append("Vol")
                                    if use_200ema:
                                        parts.append("200EMA")
                                    if confirm_2:
                                        parts.append("2bar")

                                    label = " + ".join(parts)

                                    all_results.append({
                                        "label":     label,
                                        "ema_label": ema_label,
                                        "fast_ema":  fast_ema,
                                        "slow_ema":  slow_ema,
                                        "trades":    total_trades,
                                        "win_rate":  avg_wr,
                                        "pf":        avg_pf,
                                        "pnl":       total_pnl,
                                        "config":    config,
                                        "rsi_label": rsi_opt["rsi_label"],
                                    })

                                tested += 1
                                if tested % 100 == 0:
                                    print(f"  Tested {tested}/{total_combos} combos...")

    print(f"  Done! Tested {tested} combinations.\n")

    if not all_results:
        print("No valid results found.")
        exit()

    # Score = PF * sqrt(trades)
    for r in all_results:
        r["score"] = r["pf"] * math.sqrt(r["trades"]) if r["trades"] >= 5 else 0

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("score", ascending=False)

    # ── TOP 20 RESULTS ────────────────────────────────
    print("=" * 95)
    print("  TOP 20 INDICATOR COMBINATIONS (sorted by score = PF × √trades)")
    print("=" * 95)
    print(f"  {'#':<3} {'Indicators':<50} {'Trades':>7} {'WR':>7} {'PF':>7} {'P&L':>10} {'Score':>7}")
    print(f"  {'─'*90}")

    for i, (_, row) in enumerate(df_results.head(20).iterrows()):
        pf_str = f"{row['pf']:.2f}" if row['pf'] < 999 else "inf"
        marker = " ◀ CURRENT" if "10/50" in row["label"] and "RSI(35-65)" in row["label"] and "MACD" in row["label"] and "200EMA" in row["label"] and "Vol" in row["label"] else ""
        print(f"  {i+1:<3} {row['label']:<50} {row['trades']:>7} "
              f"{row['win_rate']:>6.1f}% {pf_str:>7} "
              f"${row['pnl']:>+8.2f} {row['score']:>7.1f}{marker}")

    # ── Find current settings ─────────────────────────
    best = df_results.iloc[0]

    print(f"\n{'='*95}")
    print(f"  YOUR CURRENT SETTINGS vs BEST FOUND")
    print(f"{'='*95}")

    current_match = df_results[
        (df_results["label"].str.contains("10/50")) &
        (df_results["label"].str.contains("RSI")) &
        (df_results["label"].str.contains("MACD")) &
        (df_results["label"].str.contains("200EMA")) &
        (df_results["label"].str.contains("Vol"))
    ]
    if not current_match.empty:
        cur = current_match.iloc[0]
        print(f"\n  CURRENT:  {cur['label']}")
        print(f"            Trades: {cur['trades']}  |  WR: {cur['win_rate']:.1f}%  |  "
              f"PF: {cur['pf']:.2f}  |  P&L: ${cur['pnl']:+.2f}")
    else:
        print(f"\n  CURRENT:  EMA 10/50 + RSI(35-65) + MACD + Vol + 200EMA")
        print(f"            (exact match not found)")

    print(f"\n  BEST:     {best['label']}")
    print(f"            Trades: {best['trades']}  |  WR: {best['win_rate']:.1f}%  |  "
          f"PF: {best['pf']:.2f}  |  P&L: ${best['pnl']:+.2f}")

    # ── RECOMMENDATION ────────────────────────────────
    cfg = best["config"]
    print(f"\n{'='*95}")
    print(f"  RECOMMENDATION FOR STOCKS BOT")
    print(f"{'='*95}")

    print(f"\n  Best EMA:  {best['fast_ema']}/{best['slow_ema']}")
    print(f"  Use these indicators:")
    print(f"    EMA {best['fast_ema']}/{best['slow_ema']} crossover  — ALWAYS ON")

    if cfg["confirm_2"]:
        print(f"    2-candle confirmation    — ON")
    else:
        print(f"    2-candle confirmation    — OFF")

    if cfg["use_rsi"]:
        print(f"    RSI filter               — ON  (range: {cfg['rsi_low']}-{cfg['rsi_high']})")
    else:
        print(f"    RSI filter               — OFF")

    if cfg["use_macd"]:
        print(f"    MACD confirmation        — ON")
    else:
        print(f"    MACD confirmation        — OFF")

    if cfg["use_vwap"]:
        print(f"    VWAP filter              — ON")
    else:
        print(f"    VWAP filter              — OFF")

    if cfg["use_vol"]:
        print(f"    Volume spike filter      — ON")
    else:
        print(f"    Volume spike filter      — OFF")

    if cfg["use_200ema"]:
        print(f"    200-day EMA filter       — ON")
    else:
        print(f"    200-day EMA filter       — OFF")

    # ── Code changes ──────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  WHAT TO CHANGE IN bot.py (stocks section):")

    if best['fast_ema'] != 10 or best['slow_ema'] != 50:
        print(f"    S_FAST_EMA = {best['fast_ema']}   (was 10)")
        print(f"    S_SLOW_EMA = {best['slow_ema']}   (was 50)")

    if not cfg["use_rsi"]:
        print(f"    Remove:  RSI filter from stocks buy signal")
    elif cfg["rsi_low"] != 35 or cfg["rsi_high"] != 65:
        print(f"    Change RSI range to {cfg['rsi_low']}-{cfg['rsi_high']}")

    if not cfg["use_macd"]:
        print(f"    Remove:  MACD condition from stocks buy signal")
    if not cfg["use_vwap"]:
        print(f"    Remove:  VWAP condition from stocks buy signal")
    if not cfg["use_vol"]:
        print(f"    Remove:  Volume spike condition from stocks buy signal")
    if not cfg["use_200ema"]:
        print(f"    Remove:  200 EMA condition from stocks buy signal")
    if not cfg["confirm_2"]:
        print(f"    Change:  Use 1-candle crossover")

    # ── Indicator impact analysis ─────────────────────
    print(f"\n{'='*95}")
    print(f"  INDICATOR IMPACT ANALYSIS (stocks)")
    print(f"  (average PF when indicator is ON vs OFF)")
    print(f"{'='*95}")

    for indicator, key in [("RSI (any range)", "use_rsi"),
                            ("MACD", "use_macd"),
                            ("VWAP", "use_vwap"),
                            ("Volume spike", "use_vol"),
                            ("200-day EMA", "use_200ema"),
                            ("2-candle confirm", "confirm_2")]:
        on_results  = [r for r in all_results if r["config"][key] == True and r["trades"] >= 5]
        off_results = [r for r in all_results if r["config"][key] == False and r["trades"] >= 5]

        avg_pf_on  = sum(r["pf"] for r in on_results)  / len(on_results)  if on_results  else 0
        avg_pf_off = sum(r["pf"] for r in off_results) / len(off_results) if off_results else 0
        avg_t_on   = sum(r["trades"] for r in on_results)  / len(on_results)  if on_results  else 0
        avg_t_off  = sum(r["trades"] for r in off_results) / len(off_results) if off_results else 0

        if avg_pf_on > avg_pf_off * 1.05:
            verdict = "HELPS  ✅"
        elif avg_pf_off > avg_pf_on * 1.05:
            verdict = "HURTS  ❌"
        else:
            verdict = "NEUTRAL ─"

        print(f"  {indicator:<22}  ON: PF {avg_pf_on:>5.2f} ({avg_t_on:>5.0f} trades)  |  "
              f"OFF: PF {avg_pf_off:>5.2f} ({avg_t_off:>5.0f} trades)  |  {verdict}")

    # EMA comparison
    print(f"\n  EMA PAIR COMPARISON:")
    for ema_label in ["5/20", "7/18", "9/21", "10/50", "12/26"]:
        ema_results = [r for r in all_results if r["ema_label"] == ema_label and r["trades"] >= 5]
        if ema_results:
            avg_pf  = sum(r["pf"] for r in ema_results) / len(ema_results)
            avg_tr  = sum(r["trades"] for r in ema_results) / len(ema_results)
            avg_pnl = sum(r["pnl"] for r in ema_results) / len(ema_results)
            marker  = " ◀ current" if ema_label == "10/50" else ""
            print(f"    EMA {ema_label:<6}  PF: {avg_pf:>5.2f}  |  "
                  f"Avg trades: {avg_tr:>5.0f}  |  Avg P&L: ${avg_pnl:>+8.2f}{marker}")

    print(f"\n{'='*95}")
    print(f"  Done! Update bot.py stocks section based on the results above.")
    print(f"{'='*95}\n")
