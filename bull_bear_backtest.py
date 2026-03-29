"""
═══════════════════════════════════════════════════════════
  BULLISH + BEARISH STRATEGY BACKTEST
  
  Tests your current bullish bot PLUS bearish (short) strategies
  on the same data to see combined performance.
  
  BULLISH (current bot):
    - Trend:    EMA cross UP + MACD bullish + regime UP
    - Breakout: Price > recent high + strong candle + vol spike
    - Pullback: Regime UP + near EMA + rebound + MACD improving
  
  BEARISH (new — mirror of bullish):
    - Short Trend:    EMA cross DOWN + MACD bearish + regime DOWN
    - Short Breakout: Price < recent low + strong bearish + vol spike
    - Short Pullback: Regime DOWN + price rejected at EMA + MACD worsening
  
  Exits: ATR-based (same as bullish but reversed for shorts)
  
  Coins: BTC, ETH, SOL
  Timeframes: 15m, 30m, 1h
  Data: 4 years from Binance US
  
  Run:  python3 bull_bear_backtest.py
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


# ═══════════════════════════════════════════════════════
#  BACKTEST — runs bullish only, bearish only, or both
# ═══════════════════════════════════════════════════════

def run_backtest(df, regime_df, mode="both"):
    """
    mode: "bull" = long only, "bear" = short only, "both" = combined
    """
    df = add_indicators(df.copy())
    df = df.dropna()

    balance     = START_BAL
    in_trade    = False
    trade_dir   = None  # "long" or "short"
    entry_price = 0.0
    stop_price  = 0.0
    tp_price    = 0.0
    highest     = 0.0
    lowest      = 999999.0
    coin_held   = 0.0
    entry_strat = ""

    wins = losses = 0
    gross_wins = gross_losses = 0.0
    long_trades = long_wins = 0
    short_trades = short_wins = 0
    trade_list = []

    bull_strats = {"Trend": [0, 0], "Breakout": [0, 0], "Pullback": [0, 0]}
    bear_strats = {"ShortTrend": [0, 0], "ShortBreakout": [0, 0], "ShortPullback": [0, 0]}

    min_bars = max(MACD_SLOW + 10, BREAKOUT_LOOKBACK + 5)

    for i in range(min_bars, len(df)):
        prev = df.iloc[i - 1]
        last = df.iloc[i]
        price = safe_float(last["close"])
        atr   = safe_float(last["atr"])

        # ── Exit logic ────────────────────────────────
        if in_trade:
            if trade_dir == "long":
                if price > highest:
                    highest = price
                if price <= stop_price or price >= tp_price:
                    pnl = coin_held * (price - entry_price)
                    balance += coin_held * price
                    if pnl > 0:
                        wins += 1; gross_wins += pnl; long_wins += 1
                        if entry_strat in bull_strats: bull_strats[entry_strat][1] += 1
                    else:
                        losses += 1; gross_losses += abs(pnl)
                    trade_list.append({"pnl": pnl, "balance": balance, "dir": "long", "strat": entry_strat})
                    in_trade = False; coin_held = 0.0; highest = 0.0

            elif trade_dir == "short":
                if price < lowest:
                    lowest = price
                # Short: profit when price goes DOWN
                # Stop = price goes UP above stop
                # TP = price goes DOWN below tp
                if price >= stop_price or price <= tp_price:
                    pnl = coin_held * (entry_price - price)  # reversed for short
                    balance += coin_held * entry_price + pnl  # return original + profit/loss
                    if pnl > 0:
                        wins += 1; gross_wins += pnl; short_wins += 1
                        if entry_strat in bear_strats: bear_strats[entry_strat][1] += 1
                    else:
                        losses += 1; gross_losses += abs(pnl)
                    trade_list.append({"pnl": pnl, "balance": balance, "dir": "short", "strat": entry_strat})
                    in_trade = False; coin_held = 0.0; lowest = 999999.0

            if in_trade:
                continue
            # Fall through to check for new entries after exit

        # ── Get regime ────────────────────────────────
        ts = df.index[i]
        if ts in regime_df.index:
            regime = regime_df.loc[ts]
        else:
            mask = regime_df.index <= ts
            if mask.any():
                regime = regime_df[mask].iloc[-1]
            else:
                continue

        regime_up = bool(regime["up"])
        regime_down = bool(regime["down"])
        regime_not_bearish = bool(regime["not_bearish"])
        regime_not_bullish = bool(regime["not_bullish"])

        # ── Signal logic ──────────────────────────────
        try:
            ema_cross_up = (safe_float(prev["ema_fast"]) < safe_float(prev["ema_slow"]) and
                            safe_float(last["ema_fast"]) > safe_float(last["ema_slow"]))
            ema_cross_down = (safe_float(prev["ema_fast"]) > safe_float(prev["ema_slow"]) and
                              safe_float(last["ema_fast"]) < safe_float(last["ema_slow"]))

            macd_bullish = (safe_float(last["macd"]) > safe_float(last["macd_signal"]) and
                            safe_float(last["macd_hist"]) > 0)
            macd_bearish = (safe_float(last["macd"]) < safe_float(last["macd_signal"]) and
                            safe_float(last["macd_hist"]) < 0)
            macd_improving = safe_float(last["macd_hist"]) > safe_float(prev["macd_hist"])
            macd_worsening = safe_float(last["macd_hist"]) < safe_float(prev["macd_hist"])

            strategy = None
            direction = None

            # ══ BULLISH STRATEGIES ════════════════════
            if mode in ("bull", "both"):
                # A. Trend long
                if ema_cross_up and macd_bullish and regime_up:
                    strategy = "Trend"; direction = "long"

                # B. Breakout long
                if not strategy:
                    if (price > safe_float(last["recent_high"]) and
                        bool(last["strong_bullish"]) and
                        bool(last["vol_spike"]) and
                        regime_not_bearish):
                        strategy = "Breakout"; direction = "long"

                # C. Pullback long
                if not strategy:
                    prev_near_ema = safe_float(prev["close"]) <= safe_float(prev["ema_fast"]) * (1 + PULLBACK_BUFFER)
                    rebound = (bool(last["bullish_candle"]) and
                               price > safe_float(last["ema_fast"]) and
                               price > safe_float(prev["high"]))
                    if regime_up and prev_near_ema and rebound and macd_improving:
                        strategy = "Pullback"; direction = "long"

            # ══ BEARISH STRATEGIES (mirror) ═══════════
            if mode in ("bear", "both") and not strategy:
                # A. Short Trend
                if ema_cross_down and macd_bearish and regime_down:
                    strategy = "ShortTrend"; direction = "short"

                # B. Short Breakout
                if not strategy:
                    if (price < safe_float(last["recent_low"]) and
                        bool(last["strong_bearish"]) and
                        bool(last["vol_spike"]) and
                        regime_not_bullish):
                        strategy = "ShortBreakout"; direction = "short"

                # C. Short Pullback
                if not strategy:
                    prev_near_ema_above = safe_float(prev["close"]) >= safe_float(prev["ema_fast"]) * (1 - PULLBACK_BUFFER)
                    rejection = (bool(last["bearish_candle"]) and
                                 price < safe_float(last["ema_fast"]) and
                                 price < safe_float(prev["low"]))
                    if regime_down and prev_near_ema_above and rejection and macd_worsening:
                        strategy = "ShortPullback"; direction = "short"

            # ── Execute trade ─────────────────────────
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
                        long_trades += 1
                        if strategy in bull_strats: bull_strats[strategy][0] += 1

                    elif direction == "short":
                        balance    -= spend  # collateral
                        lowest      = price
                        stop_price  = price + (atr * STOP_ATR)  # stop ABOVE for short
                        tp_price    = price - (atr * TP_ATR)    # TP BELOW for short
                        short_trades += 1
                        if strategy in bear_strats: bear_strats[strategy][0] += 1

        except:
            continue

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
        trade_list.append({"pnl": pnl, "balance": balance, "dir": trade_dir, "strat": entry_strat})

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
        "long_trades": long_trades, "long_wins": long_wins,
        "short_trades": short_trades, "short_wins": short_wins,
        "bull_strats": bull_strats, "bear_strats": bear_strats,
    }


def print_result(label, r):
    if r is None:
        print(f"\n  {label}")
        print(f"  {'─'*65}")
        print(f"  No trades")
        return
    pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
    long_wr = r['long_wins'] / r['long_trades'] * 100 if r['long_trades'] > 0 else 0
    short_wr = r['short_wins'] / r['short_trades'] * 100 if r['short_trades'] > 0 else 0

    print(f"\n  {label}")
    print(f"  {'─'*65}")
    print(f"  P&L: ${r['pnl']:+.2f}  |  PF: {pf_str}  |  WR: {r['win_rate']:.1f}%  |  DD: {r['max_dd']*100:.1f}%")
    print(f"  Total: {r['total']} trades  ({r['wins']}W / {r['losses']}L)")
    if r['long_trades'] > 0:
        print(f"  Longs:  {r['long_trades']} trades  |  WR: {long_wr:.1f}%")
    if r['short_trades'] > 0:
        print(f"  Shorts: {r['short_trades']} trades  |  WR: {short_wr:.1f}%")

    # Strategy breakdown
    for name, (count, w) in r['bull_strats'].items():
        if count > 0:
            print(f"    {name:<14} {count:>4} trades  |  WR: {w/count*100:.1f}%")
    for name, (count, w) in r['bear_strats'].items():
        if count > 0:
            print(f"    {name:<14} {count:>4} trades  |  WR: {w/count*100:.1f}%")


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    htf_map = {
        "15m": "1h",
        "30m": "1h",
        "1h":  "4h",
    }
    timeframes = ["15m", "30m", "1h"]

    print("\n" + "=" * 75)
    print("  BULLISH + BEARISH STRATEGY BACKTEST")
    print(f"  Bull: Trend + Breakout + Pullback (current bot)")
    print(f"  Bear: ShortTrend + ShortBreakout + ShortPullback (mirror)")
    print(f"  Exits: ATR stop ({STOP_ATR}) | ATR TP ({TP_ATR})")
    print(f"  Risk: {RISK*100:.0f}% per trade | Data: 4 years")
    print(f"  Coins: {', '.join(s.split('/')[0] for s in SYMBOLS)}")
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

    # ── Run 3 modes per coin per TF ───────────────────
    config_num = 0
    total_configs = len(timeframes) * len(SYMBOLS) * 3  # 3 modes
    all_results = {}

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

            for mode in ["bull", "bear", "both"]:
                config_num += 1
                pct = config_num / total_configs * 100
                bar_len = 30
                filled = int(bar_len * config_num / total_configs)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"  [{bar}] {pct:>5.1f}%  {coin} {tf} {mode}", flush=True)

                r = run_backtest(d[tf], regime_df, mode)
                key = f"{coin}_{tf}_{mode}"
                all_results[key] = r

        # Print results for this timeframe
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            for mode in ["bull", "bear", "both"]:
                key = f"{coin}_{tf}_{mode}"
                r = all_results.get(key)
                mode_label = {"bull": "LONG ONLY", "bear": "SHORT ONLY", "both": "BULL+BEAR"}[mode]
                print_result(f"{coin} {tf} — {mode_label}", r)

    # ── COMPARISON TABLE ──────────────────────────────
    print(f"\n{'='*75}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Config':<30} {'Trades':>7} {'WR':>7} {'PF':>7} {'P&L':>11} {'DD':>6}")
    print(f"  {'─'*70}")

    for tf in timeframes:
        for mode in ["bull", "bear", "both"]:
            trades = 0; pnl = 0; pf_sum = 0; count = 0; wins = 0
            for sym in SYMBOLS:
                coin = sym.split("/")[0]
                key = f"{coin}_{tf}_{mode}"
                r = all_results.get(key)
                if r:
                    trades += r["total"]; pnl += r["pnl"]
                    pf_sum += r["pf"]; wins += r["wins"]; count += 1
            if count > 0 and trades > 0:
                avg_pf = pf_sum / count
                avg_wr = wins / trades * 100
                mode_label = {"bull": "LONG", "bear": "SHORT", "both": "COMBINED"}[mode]
                pf_str = f"{avg_pf:.2f}" if avg_pf < 999 else "inf"
                max_dd = max(all_results.get(f"{s.split('/')[0]}_{tf}_{mode}", {}).get("max_dd", 0)
                             for s in SYMBOLS if all_results.get(f"{s.split('/')[0]}_{tf}_{mode}"))
                marker = " ◀ CURRENT" if mode == "bull" and tf == "15m" else ""
                print(f"  {tf} {mode_label:<25} {trades:>7} {avg_wr:>6.1f}% "
                      f"{pf_str:>7} ${pnl:>+9.2f} {max_dd*100:>5.1f}%{marker}")
        print(f"  {'─'*70}")

    # ── VALUE OF ADDING SHORTS ────────────────────────
    print(f"\n{'='*75}")
    print(f"  DOES ADDING SHORTS HELP?")
    print(f"{'='*75}")

    for tf in timeframes:
        bull_pnl = 0; bear_pnl = 0; both_pnl = 0
        bull_trades = 0; bear_trades = 0; both_trades = 0
        bull_pf = 0; bear_pf = 0; both_pf = 0
        cnt = 0
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            rb = all_results.get(f"{coin}_{tf}_bull")
            rs = all_results.get(f"{coin}_{tf}_bear")
            rc = all_results.get(f"{coin}_{tf}_both")
            if rb: bull_pnl += rb["pnl"]; bull_trades += rb["total"]; bull_pf += rb["pf"]
            if rs: bear_pnl += rs["pnl"]; bear_trades += rs["total"]; bear_pf += rs["pf"]
            if rc: both_pnl += rc["pnl"]; both_trades += rc["total"]; both_pf += rc["pf"]
            cnt += 1

        if cnt > 0:
            print(f"\n  {tf.upper()}:")
            print(f"    Bull only:  ${bull_pnl:>+9.2f}  |  {bull_trades} trades  |  PF {bull_pf/cnt:.2f}")
            print(f"    Bear only:  ${bear_pnl:>+9.2f}  |  {bear_trades} trades  |  PF {bear_pf/cnt:.2f}")
            print(f"    Combined:   ${both_pnl:>+9.2f}  |  {both_trades} trades  |  PF {both_pf/cnt:.2f}")

            if both_pnl > bull_pnl:
                print(f"    ✅ Adding shorts HELPS (+${both_pnl - bull_pnl:.2f})")
            else:
                print(f"    ❌ Adding shorts HURTS (-${bull_pnl - both_pnl:.2f})")

    # ── FINAL RECOMMENDATION ──────────────────────────
    print(f"\n{'='*75}")
    print(f"  FINAL RECOMMENDATION")
    print(f"{'='*75}")

    # Find the best mode per timeframe
    for tf in timeframes:
        modes = {}
        for mode in ["bull", "bear", "both"]:
            pnl = 0; pf_sum = 0; cnt = 0
            for sym in SYMBOLS:
                coin = sym.split("/")[0]
                r = all_results.get(f"{coin}_{tf}_{mode}")
                if r:
                    pnl += r["pnl"]; pf_sum += r["pf"]; cnt += 1
            if cnt > 0:
                modes[mode] = {"pnl": pnl, "avg_pf": pf_sum / cnt}

        if modes:
            best_mode = max(modes, key=lambda k: modes[k]["pnl"])
            best = modes[best_mode]
            mode_label = {"bull": "LONG ONLY", "bear": "SHORT ONLY", "both": "BULL+BEAR"}[best_mode]
            pf_str = f"{best['avg_pf']:.2f}" if best['avg_pf'] < 999 else "inf"
            print(f"\n  {tf}: Best = {mode_label}")
            print(f"    P&L: ${best['pnl']:+.2f}  |  Avg PF: {pf_str}")

    print(f"\n{'='*75}")
    print(f"  Done!")
    print(f"{'='*75}\n")
