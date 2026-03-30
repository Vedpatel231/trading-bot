"""
═══════════════════════════════════════════════════════════
  REGIME FLIP STRATEGY BACKTEST
  
  Tests a new 4th entry strategy: RegimeFlip
  
  Current bot misses trades when:
    1. EMA crosses while regime is still DOWN
    2. Regime flips to UP later — but EMA already crossed
    3. No new crossover = no entry = missed trade
  
  RegimeFlip fixes this:
    BUY when:  regime JUST flipped to UP (was not UP on prev candle)
               + EMA fast > slow (already crossed)
               + MACD bullish
    
    SHORT when: regime JUST flipped to DOWN (was not DOWN on prev candle)
                + EMA fast < slow (already crossed)
                + MACD bearish
  
  Tests 3 modes:
    A. Current bot only (Trend + Breakout + Pullback)
    B. RegimeFlip only (new strategy alone)
    C. Current + RegimeFlip (all 4 strategies combined)
  
  Coins: BTC, ETH, SOL
  Timeframes: 15m (with 1h regime), 30m, 1h
  Data: 4 years
  
  Run:  python3 regimeflip_backtest.py
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import ccxt
import time as time_module
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

SYMBOLS          = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
FAST_EMA         = 7
SLOW_EMA         = 18
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIG         = 9
ATR_PERIOD       = 14

STOP_ATR         = 1.2
TP_ATR           = 2.5
BREAKOUT_LOOKBACK = 10
BREAKOUT_VOL_MULT = 1.5
STRONG_BODY_MULT = 1.2
REGIME_MIN_ATR_PCT = 0.002
PULLBACK_BUFFER  = 0.003

RISK             = 0.02
START_BAL        = 10_000.0
LOOKBACK         = 1460

exchange = ccxt.binanceus()


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default


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
        print(" no data!")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    days = (df.index[-1] - df.index[0]).days
    print(f" {len(df)} candles ({days}d / {days/365:.1f}y)")
    return df


def resample(df, tf):
    return df.resample(tf).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()


def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST,
                          window_slow=MACD_SLOW, window_sign=MACD_SIG)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    df["atr"]     = ta.volatility.average_true_range(
                      df["high"], df["low"], df["close"], window=ATR_PERIOD)
    df["atr_pct"] = df["atr"] / df["close"]

    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_avg"] * BREAKOUT_VOL_MULT

    df["body"]           = (df["close"] - df["open"]).abs()
    df["body_avg"]       = df["body"].rolling(20).mean()
    df["bullish_candle"] = df["close"] > df["open"]
    df["bearish_candle"] = df["close"] < df["open"]
    df["strong_bullish"] = df["bullish_candle"] & (df["body"] > df["body_avg"] * STRONG_BODY_MULT)
    df["strong_bearish"] = df["bearish_candle"] & (df["body"] > df["body_avg"] * STRONG_BODY_MULT)

    df["recent_high"] = df["high"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["recent_low"]  = df["low"].shift(1).rolling(BREAKOUT_LOOKBACK).min()
    return df


def compute_regime_series(entry_df, htf_df):
    htf = add_indicators(htf_df.copy())
    htf["fast_above"]  = htf["ema_fast"] > htf["ema_slow"]
    htf["fast_below"]  = htf["ema_fast"] < htf["ema_slow"]
    htf["slow_rising"] = htf["ema_slow"] > htf["ema_slow"].shift(1)
    htf["slow_falling"] = htf["ema_slow"] < htf["ema_slow"].shift(1)
    htf["regime_up"]   = htf["fast_above"] & htf["slow_rising"] & (htf["atr_pct"] >= REGIME_MIN_ATR_PCT)
    htf["regime_down"] = htf["fast_below"] & htf["slow_falling"] & (htf["atr_pct"] >= REGIME_MIN_ATR_PCT)
    htf["regime_not_bearish"] = htf["fast_above"] | htf["slow_rising"]
    htf["regime_not_bullish"] = htf["fast_below"] | htf["slow_falling"]

    results = []
    htf_times = htf.index
    for ts in entry_df.index:
        mask = htf_times <= ts
        if mask.any():
            idx = mask.sum() - 1
            row = htf.iloc[idx]
            results.append({
                "up": bool(row["regime_up"]),
                "down": bool(row["regime_down"]),
                "not_bearish": bool(row["regime_not_bearish"]),
                "not_bullish": bool(row["regime_not_bullish"]),
            })
        else:
            results.append({"up": False, "down": False, "not_bearish": False, "not_bullish": False})

    return pd.DataFrame(results, index=entry_df.index)


def calc_position_size(balance, price, atr):
    risk_amount = balance * RISK
    if atr > 0:
        stop_distance = atr * STOP_ATR
        qty = risk_amount / stop_distance if stop_distance > 0 else 0
        spend = min(qty * price, balance * 0.25)
        spend = max(spend, 1.0)
    else:
        spend = max(risk_amount, 1.0)
    return round(spend, 2)


def run_backtest(df, regime_df, mode="current"):
    """
    mode:
      "current"    = Trend + Breakout + Pullback (existing bot)
      "flip_only"  = RegimeFlip only (new strategy alone)
      "combined"   = Current + RegimeFlip (all 4 strategies)
      "all_both"   = Current + RegimeFlip + Short versions of all
    """
    df = add_indicators(df.copy())
    df = df.dropna()

    balance     = START_BAL
    in_trade    = False
    trade_dir   = None
    entry_price = 0.0
    stop_price  = 0.0
    tp_price    = 0.0
    highest     = 0.0
    lowest      = 999999.0
    coin_held   = 0.0
    entry_strat = ""

    wins = losses = 0
    gross_wins = gross_losses = 0.0
    trade_list = []
    strat_counts = {}

    prev_regime_up = False
    prev_regime_down = False

    min_bars = max(MACD_SLOW + 10, BREAKOUT_LOOKBACK + 5)

    for i in range(min_bars, len(df)):
        prev = df.iloc[i - 1]
        last = df.iloc[i]
        price = safe_float(last["close"])
        atr   = safe_float(last["atr"])

        # Get regime
        ts = df.index[i]
        if ts in regime_df.index:
            regime = regime_df.loc[ts]
        else:
            mask = regime_df.index <= ts
            if mask.any():
                regime = regime_df[mask].iloc[-1]
            else:
                prev_regime_up = False
                prev_regime_down = False
                continue

        regime_up = bool(regime["up"])
        regime_down = bool(regime["down"])
        regime_not_bearish = bool(regime["not_bearish"])
        regime_not_bullish = bool(regime["not_bullish"])

        # Detect regime flip
        regime_just_flipped_up = regime_up and not prev_regime_up
        regime_just_flipped_down = regime_down and not prev_regime_down

        # Exit logic
        if in_trade:
            if trade_dir == "long":
                if price > highest: highest = price
                if price <= stop_price or price >= tp_price:
                    pnl = coin_held * (price - entry_price)
                    balance += coin_held * price
                    if pnl > 0: wins += 1; gross_wins += pnl
                    else: losses += 1; gross_losses += abs(pnl)
                    strat_counts.setdefault(entry_strat, [0, 0])
                    strat_counts[entry_strat][0] += 1
                    if pnl > 0: strat_counts[entry_strat][1] += 1
                    trade_list.append({"pnl": pnl, "balance": balance, "strat": entry_strat})
                    in_trade = False; coin_held = 0.0; highest = 0.0

            elif trade_dir == "short":
                if price < lowest: lowest = price
                if price >= stop_price or price <= tp_price:
                    pnl = coin_held * (entry_price - price)
                    balance += coin_held * entry_price + pnl
                    if pnl > 0: wins += 1; gross_wins += pnl
                    else: losses += 1; gross_losses += abs(pnl)
                    strat_counts.setdefault(entry_strat, [0, 0])
                    strat_counts[entry_strat][0] += 1
                    if pnl > 0: strat_counts[entry_strat][1] += 1
                    trade_list.append({"pnl": pnl, "balance": balance, "strat": entry_strat})
                    in_trade = False; coin_held = 0.0; lowest = 999999.0

            if in_trade:
                prev_regime_up = regime_up
                prev_regime_down = regime_down
                continue

        # Signal logic
        try:
            ema_cross_up = (safe_float(prev["ema_fast"]) < safe_float(prev["ema_slow"]) and
                            safe_float(last["ema_fast"]) > safe_float(last["ema_slow"]))
            ema_cross_down = (safe_float(prev["ema_fast"]) > safe_float(prev["ema_slow"]) and
                              safe_float(last["ema_fast"]) < safe_float(last["ema_slow"]))

            ema_already_above = safe_float(last["ema_fast"]) > safe_float(last["ema_slow"])
            ema_already_below = safe_float(last["ema_fast"]) < safe_float(last["ema_slow"])

            macd_bullish = (safe_float(last["macd"]) > safe_float(last["macd_signal"]) and
                            safe_float(last["macd_hist"]) > 0)
            macd_bearish = (safe_float(last["macd"]) < safe_float(last["macd_signal"]) and
                            safe_float(last["macd_hist"]) < 0)
            macd_improving = safe_float(last["macd_hist"]) > safe_float(prev["macd_hist"])
            macd_worsening = safe_float(last["macd_hist"]) < safe_float(prev["macd_hist"])

            strategy = None
            direction = None

            # ═══ CURRENT STRATEGIES (long) ════════════
            if mode in ("current", "combined", "all_both"):
                if ema_cross_up and macd_bullish and regime_up:
                    strategy = "Trend"; direction = "long"

                if not strategy:
                    if (price > safe_float(last["recent_high"]) and
                        bool(last["strong_bullish"]) and
                        bool(last["vol_spike"]) and regime_not_bearish):
                        strategy = "Breakout"; direction = "long"

                if not strategy:
                    prev_near_ema = safe_float(prev["close"]) <= safe_float(prev["ema_fast"]) * (1 + PULLBACK_BUFFER)
                    rebound = (bool(last["bullish_candle"]) and
                               price > safe_float(last["ema_fast"]) and
                               price > safe_float(prev["high"]))
                    if regime_up and prev_near_ema and rebound and macd_improving:
                        strategy = "Pullback"; direction = "long"

            # ═══ REGIME FLIP (long) ═══════════════════
            if mode in ("flip_only", "combined", "all_both") and not strategy:
                if regime_just_flipped_up and ema_already_above and macd_bullish:
                    strategy = "RegimeFlip"; direction = "long"

            # ═══ SHORT STRATEGIES ═════════════════════
            if mode == "all_both" and not strategy:
                # Short Trend
                if ema_cross_down and macd_bearish and regime_down:
                    strategy = "ShortTrend"; direction = "short"

                # Short Breakout
                if not strategy:
                    if (price < safe_float(last["recent_low"]) and
                        bool(last["strong_bearish"]) and
                        bool(last["vol_spike"]) and regime_not_bullish):
                        strategy = "ShortBreakout"; direction = "short"

                # Short Pullback
                if not strategy:
                    prev_near_above = safe_float(prev["close"]) >= safe_float(prev["ema_fast"]) * (1 - PULLBACK_BUFFER)
                    rejection = (bool(last["bearish_candle"]) and
                                 price < safe_float(last["ema_fast"]) and
                                 price < safe_float(prev["low"]))
                    if regime_down and prev_near_above and rejection and macd_worsening:
                        strategy = "ShortPullback"; direction = "short"

                # Short RegimeFlip
                if not strategy:
                    if regime_just_flipped_down and ema_already_below and macd_bearish:
                        strategy = "ShortRegimeFlip"; direction = "short"

            # Execute
            if strategy and direction and atr > 0:
                spend = calc_position_size(balance, price, atr)
                if spend >= 1.0 and spend <= balance:
                    coin_held   = spend / price
                    entry_price = price
                    entry_strat = strategy
                    trade_dir   = direction
                    in_trade    = True

                    if direction == "long":
                        balance    -= spend
                        highest     = price
                        stop_price  = price - (atr * STOP_ATR)
                        tp_price    = price + (atr * TP_ATR)
                    else:
                        balance    -= spend
                        lowest      = price
                        stop_price  = price + (atr * STOP_ATR)
                        tp_price    = price - (atr * TP_ATR)

        except:
            pass

        prev_regime_up = regime_up
        prev_regime_down = regime_down

    # Close open trade
    if in_trade:
        price = safe_float(df.iloc[-1]["close"])
        if trade_dir == "long":
            pnl = coin_held * (price - entry_price)
            balance += coin_held * price
        else:
            pnl = coin_held * (entry_price - price)
            balance += coin_held * entry_price + pnl
        if pnl > 0: wins += 1; gross_wins += pnl
        else: losses += 1; gross_losses += abs(pnl)
        trade_list.append({"pnl": pnl, "balance": balance, "strat": entry_strat})

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
        "strat_counts": strat_counts,
    }


def print_result(label, r):
    if r is None:
        print(f"\n  {label}")
        print(f"  {'─'*65}")
        print(f"  No trades")
        return
    pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
    print(f"\n  {label}")
    print(f"  {'─'*65}")
    print(f"  P&L: ${r['pnl']:+.2f}  |  PF: {pf_str}  |  WR: {r['win_rate']:.1f}%  |  DD: {r['max_dd']*100:.1f}%")
    print(f"  Trades: {r['total']}  ({r['wins']}W / {r['losses']}L)")
    for strat, (count, w) in sorted(r['strat_counts'].items()):
        wr = w / count * 100 if count > 0 else 0
        print(f"    {strat:<18} {count:>4} trades  |  WR: {wr:.1f}%")


if __name__ == "__main__":
    htf_map = {"15m": "1h", "30m": "1h", "1h": "4h"}
    timeframes = ["15m", "30m", "1h"]
    modes = [
        ("current",  "Current bot (Trend+Breakout+Pullback)"),
        ("flip_only", "RegimeFlip ONLY"),
        ("combined", "Current + RegimeFlip"),
        ("all_both", "ALL strategies + shorts"),
    ]

    print("\n" + "=" * 75)
    print("  REGIME FLIP STRATEGY BACKTEST")
    print(f"  New strategy: enter when regime JUST flips + EMA already crossed")
    print(f"  Coins: {', '.join(s.split('/')[0] for s in SYMBOLS)}")
    print(f"  Data: {LOOKBACK} days ({LOOKBACK/365:.1f} years)")
    print("=" * 75)

    # Download
    print(f"\nDownloading data...\n")
    all_data = {}
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        print(f"  {coin}:")
        all_data[sym] = {}
        for tf in timeframes:
            df = fetch_all_candles(sym, tf, LOOKBACK)
            all_data[sym][tf] = df
        if len(all_data[sym]["1h"]) > 0:
            all_data[sym]["4h"] = resample(all_data[sym]["1h"], "4h")
            print(f"    4h (resampled): {len(all_data[sym]['4h'])} candles")
        print()

    # Compute regimes
    print("Computing regime filters...")
    regime_cache = {}
    for sym in SYMBOLS:
        d = all_data[sym]
        regime_cache[sym] = {}
        for entry_tf, htf_tf in htf_map.items():
            if entry_tf in d and htf_tf in d:
                if len(d[entry_tf]) > 50 and len(d[htf_tf]) > 20:
                    key = f"{entry_tf}_{htf_tf}"
                    regime_cache[sym][key] = compute_regime_series(d[entry_tf], d[htf_tf])
    print("  Done.\n")

    # Run all
    all_results = {}
    config_num = 0
    total_configs = len(timeframes) * len(SYMBOLS) * len(modes)

    for tf in timeframes:
        htf_tf = htf_map[tf]
        print(f"{'='*75}")
        print(f"  TIMEFRAME: {tf.upper()} entry  |  {htf_tf.upper()} regime")
        print(f"{'='*75}")

        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            d = all_data[sym]
            if tf not in d or len(d[tf]) < 50:
                continue

            regime_key = f"{tf}_{htf_tf}"
            regime_df = regime_cache.get(sym, {}).get(regime_key)
            if regime_df is None:
                continue

            for mode_key, mode_label in modes:
                config_num += 1
                pct = config_num / total_configs * 100
                bar_len = 30
                filled = int(bar_len * config_num / total_configs)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"  [{bar}] {pct:>5.1f}%  {coin} {tf} {mode_key}", flush=True)

                r = run_backtest(d[tf], regime_df, mode_key)
                key = f"{coin}_{tf}_{mode_key}"
                all_results[key] = r

        # Print per TF
        for mode_key, mode_label in modes:
            print(f"\n  --- {mode_label} ---")
            for sym in SYMBOLS:
                coin = sym.split("/")[0]
                key = f"{coin}_{tf}_{mode_key}"
                r = all_results.get(key)
                print_result(f"  {coin} {tf}", r)

    # ── HEAD-TO-HEAD ──────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Config':<40} {'Trades':>7} {'WR':>7} {'PF':>7} {'P&L':>11} {'DD':>6}")
    print(f"  {'─'*80}")

    for tf in timeframes:
        for mode_key, mode_label in modes:
            trades = 0; pnl = 0; pf_sum = 0; count = 0; w = 0
            max_dd = 0
            for sym in SYMBOLS:
                coin = sym.split("/")[0]
                key = f"{coin}_{tf}_{mode_key}"
                r = all_results.get(key)
                if r:
                    trades += r["total"]; pnl += r["pnl"]
                    pf_sum += r["pf"]; w += r["wins"]; count += 1
                    max_dd = max(max_dd, r["max_dd"])
            if count > 0 and trades > 0:
                avg_pf = pf_sum / count
                avg_wr = w / trades * 100
                pf_str = f"{avg_pf:.2f}" if avg_pf < 999 else "inf"
                marker = " ◀ CURRENT" if mode_key == "current" and tf == "15m" else ""
                print(f"  {tf} {mode_label[:35]:<35} {trades:>7} {avg_wr:>6.1f}% "
                      f"{pf_str:>7} ${pnl:>+9.2f} {max_dd*100:>5.1f}%{marker}")
        print(f"  {'─'*80}")

    # ── REGIME FLIP IMPACT ────────────────────────────
    print(f"\n{'='*75}")
    print(f"  DOES REGIME FLIP HELP?")
    print(f"{'='*75}")

    for tf in timeframes:
        current_pnl = 0; combined_pnl = 0; flip_pnl = 0; all_pnl = 0
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            rc = all_results.get(f"{coin}_{tf}_current")
            rf = all_results.get(f"{coin}_{tf}_flip_only")
            rb = all_results.get(f"{coin}_{tf}_combined")
            ra = all_results.get(f"{coin}_{tf}_all_both")
            if rc: current_pnl += rc["pnl"]
            if rf: flip_pnl += rf["pnl"]
            if rb: combined_pnl += rb["pnl"]
            if ra: all_pnl += ra["pnl"]

        print(f"\n  {tf.upper()}:")
        print(f"    Current bot:           ${current_pnl:>+10.2f}")
        print(f"    RegimeFlip only:       ${flip_pnl:>+10.2f}")
        print(f"    Current + RegimeFlip:  ${combined_pnl:>+10.2f}")
        print(f"    ALL + shorts:          ${all_pnl:>+10.2f}")

        if combined_pnl > current_pnl:
            print(f"    ✅ RegimeFlip HELPS (+${combined_pnl - current_pnl:.2f})")
        else:
            print(f"    ❌ RegimeFlip HURTS (-${current_pnl - combined_pnl:.2f})")

        if all_pnl > combined_pnl:
            print(f"    ✅ Adding shorts HELPS (+${all_pnl - combined_pnl:.2f} more)")

    # ── FINAL RECOMMENDATION ──────────────────────────
    print(f"\n{'='*75}")
    print(f"  RECOMMENDATION")
    print(f"{'='*75}")

    best_tf = None
    best_mode = None
    best_pnl = -999999
    for tf in timeframes:
        for mode_key, mode_label in modes:
            pnl = 0
            for sym in SYMBOLS:
                coin = sym.split("/")[0]
                r = all_results.get(f"{coin}_{tf}_{mode_key}")
                if r: pnl += r["pnl"]
            if pnl > best_pnl:
                best_pnl = pnl
                best_tf = tf
                best_mode = mode_key

    best_label = dict(modes).get(best_mode, best_mode)
    print(f"\n  BEST OVERALL: {best_tf} + {best_label}")
    print(f"  Total P&L: ${best_pnl:+.2f}")

    # Show strategy breakdown for the winner
    print(f"\n  Strategy breakdown for {best_tf} {best_label}:")
    combined_strats = {}
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        r = all_results.get(f"{coin}_{best_tf}_{best_mode}")
        if r:
            for strat, (count, w) in r["strat_counts"].items():
                if strat not in combined_strats:
                    combined_strats[strat] = [0, 0]
                combined_strats[strat][0] += count
                combined_strats[strat][1] += w

    for strat, (count, w) in sorted(combined_strats.items()):
        wr = w / count * 100 if count > 0 else 0
        print(f"    {strat:<18} {count:>5} trades  |  WR: {wr:.1f}%")

    print(f"\n{'='*75}")
    print(f"  Done!")
    print(f"{'='*75}\n")
