"""
═══════════════════════════════════════════════════════════
  CURRENT BOT BACKTEST — exact replica of bot.py logic
  
  3 entry strategies:
    A. Trend:     EMA crossover + MACD bullish + regime UP
    B. Breakout:  Price > recent high + strong candle + vol spike + regime not bearish
    C. Pullback:  Regime UP + price near EMA + rebound candle + MACD improving
  
  Exits: ATR-based stop (1.2 ATR) and take profit (2.5 ATR)
  
  Coins: BTC, ETH, SOL
  Timeframes: 5m, 15m, 30m, 1h
  Data: 4 years from Binance US
  
  Run:  python3 current_bot_backtest.py
  Takes about 10-15 minutes.
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import ccxt
import time as time_module
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ── Settings (exact copy from bot.py) ───────────────────
SYMBOLS          = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
FAST_EMA         = 7
SLOW_EMA         = 18
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIG         = 9
ATR_PERIOD       = 14

CRYPTO_STOP_ATR  = 1.2
CRYPTO_TP_ATR    = 2.5
BREAKOUT_LOOKBACK = 10
BREAKOUT_VOL_MULT = 1.5
STRONG_BODY_MULT = 1.2
REGIME_MIN_ATR_PCT = 0.002
PULLBACK_BUFFER  = 0.003

RISK             = 0.02
START_BAL        = 10_000.0
LOOKBACK         = 1460  # 4 years

exchange = ccxt.binanceus()


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default


# ═══════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════
#  INDICATORS (exact copy from bot.py)
# ═══════════════════════════════════════════════════════

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
    df["strong_bullish"] = df["bullish_candle"] & (df["body"] > df["body_avg"] * STRONG_BODY_MULT)

    df["recent_high"] = df["high"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["recent_low"]  = df["low"].shift(1).rolling(BREAKOUT_LOOKBACK).min()
    return df


# ═══════════════════════════════════════════════════════
#  REGIME (simulated from HTF data)
# ═══════════════════════════════════════════════════════

def compute_regime_series(entry_df, htf_df):
    """
    For each bar in entry_df, compute the regime from the HTF.
    Returns a DataFrame with columns: up, not_bearish, atr_pct
    """
    htf = add_indicators(htf_df.copy())

    htf["fast_above"]  = htf["ema_fast"] > htf["ema_slow"]
    htf["slow_rising"] = htf["ema_slow"] > htf["ema_slow"].shift(1)
    htf["regime_up"]   = htf["fast_above"] & htf["slow_rising"] & (htf["atr_pct"] >= REGIME_MIN_ATR_PCT)
    htf["regime_not_bearish"] = htf["fast_above"] | htf["slow_rising"]

    # Map HTF regime to entry timeframe
    results = []
    htf_times = htf.index
    for ts in entry_df.index:
        mask = htf_times <= ts
        if mask.any():
            idx = mask.sum() - 1
            row = htf.iloc[idx]
            results.append({
                "up": bool(row["regime_up"]),
                "not_bearish": bool(row["regime_not_bearish"]),
                "atr_pct": safe_float(row["atr_pct"]),
            })
        else:
            results.append({"up": False, "not_bearish": False, "atr_pct": 0.0})

    return pd.DataFrame(results, index=entry_df.index)


# ═══════════════════════════════════════════════════════
#  POSITION SIZING (exact copy from bot.py)
# ═══════════════════════════════════════════════════════

def calc_position_size(balance, price, atr):
    risk_amount = balance * RISK
    if atr > 0:
        stop_distance = atr * CRYPTO_STOP_ATR
        qty = risk_amount / stop_distance if stop_distance > 0 else 0
        spend = min(qty * price, balance * 0.25)
        spend = max(spend, 1.0)
    else:
        spend = max(risk_amount, 1.0)
    return round(spend, 2)


# ═══════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════

def run_backtest(df, regime_df):
    """
    Exact replica of bot.py signal logic:
    - Trend:    EMA cross + MACD bullish + regime UP
    - Breakout: Price > recent high + strong candle + vol spike + not bearish
    - Pullback: Regime UP + near EMA + rebound + MACD improving
    - Exit:     ATR stop (1.2) or ATR TP (2.5)
    """
    df = add_indicators(df.copy())
    df = df.dropna()

    # Align regime to entry df
    common_idx = df.index.intersection(regime_df.index)
    if len(common_idx) < 50:
        return None

    balance     = START_BAL
    in_trade    = False
    entry_price = 0.0
    stop_price  = 0.0
    tp_price    = 0.0
    highest     = 0.0
    coin_held   = 0.0
    entry_strat = ""
    wins = losses = 0
    gross_wins = gross_losses = 0.0
    trend_trades = breakout_trades = pullback_trades = 0
    trend_wins = breakout_wins = pullback_wins = 0
    trade_list = []

    min_bars = max(MACD_SLOW + 10, BREAKOUT_LOOKBACK + 5)

    for i in range(min_bars, len(df)):
        prev = df.iloc[i - 1]
        last = df.iloc[i]
        price = safe_float(last["close"])
        atr   = safe_float(last["atr"])

        # ── Check exits ───────────────────────────────
        if in_trade:
            if price > highest:
                highest = price
            if price <= stop_price or price >= tp_price:
                pnl      = coin_held * (price - entry_price)
                balance += coin_held * price
                reason   = "TP" if price >= tp_price else "SL"

                if pnl > 0:
                    wins += 1; gross_wins += pnl
                    if entry_strat == "Trend": trend_wins += 1
                    elif entry_strat == "Breakout": breakout_wins += 1
                    elif entry_strat == "Pullback": pullback_wins += 1
                else:
                    losses += 1; gross_losses += abs(pnl)

                trade_list.append({
                    "pnl": pnl, "balance": balance,
                    "reason": reason, "strategy": entry_strat
                })
                in_trade = False; coin_held = 0.0; highest = 0.0
                stop_price = 0.0; tp_price = 0.0; entry_strat = ""
            continue

        # ── Get regime ────────────────────────────────
        ts = df.index[i]
        if ts in regime_df.index:
            regime = regime_df.loc[ts]
            regime_up = bool(regime["up"])
            regime_not_bearish = bool(regime["not_bearish"])
        else:
            # Find nearest
            mask = regime_df.index <= ts
            if mask.any():
                regime = regime_df[mask].iloc[-1]
                regime_up = bool(regime["up"])
                regime_not_bearish = bool(regime["not_bearish"])
            else:
                regime_up = False
                regime_not_bearish = False

        # ── Signal logic (exact bot.py) ───────────────
        try:
            ema_cross_up = (safe_float(prev["ema_fast"]) < safe_float(prev["ema_slow"]) and
                            safe_float(last["ema_fast"]) > safe_float(last["ema_slow"]))

            macd_bullish = (safe_float(last["macd"]) > safe_float(last["macd_signal"]) and
                            safe_float(last["macd_hist"]) > 0)

            macd_improving = safe_float(last["macd_hist"]) > safe_float(prev["macd_hist"])

            # A. Trend entry
            trend_buy = ema_cross_up and macd_bullish and regime_up

            # B. Breakout entry
            breakout_buy = (
                price > safe_float(last["recent_high"]) and
                bool(last["strong_bullish"]) and
                bool(last["vol_spike"]) and
                regime_not_bearish
            )

            # C. Pullback entry
            prev_near_ema = safe_float(prev["close"]) <= safe_float(prev["ema_fast"]) * (1 + PULLBACK_BUFFER)
            rebound_candle = (bool(last["bullish_candle"]) and
                              price > safe_float(last["ema_fast"]) and
                              price > safe_float(prev["high"]))
            pullback_buy = regime_up and prev_near_ema and rebound_candle and macd_improving

            # ── Execute buy ───────────────────────────
            strategy = None
            if trend_buy:
                strategy = "Trend"
            elif breakout_buy:
                strategy = "Breakout"
            elif pullback_buy:
                strategy = "Pullback"

            if strategy:
                spend = calc_position_size(balance, price, atr)
                if spend >= 1.0 and spend <= balance:
                    coin_held   = spend / price
                    balance    -= spend
                    entry_price = price
                    highest     = price
                    stop_price  = price - (atr * CRYPTO_STOP_ATR) if atr > 0 else price * 0.99
                    tp_price    = price + (atr * CRYPTO_TP_ATR) if atr > 0 else price * 1.02
                    entry_strat = strategy
                    in_trade    = True

                    if strategy == "Trend": trend_trades += 1
                    elif strategy == "Breakout": breakout_trades += 1
                    elif strategy == "Pullback": pullback_trades += 1

        except:
            continue

    # Close open trade
    if in_trade:
        price = safe_float(df.iloc[-1]["close"])
        pnl   = coin_held * (price - entry_price)
        balance += coin_held * price
        if pnl > 0: wins += 1; gross_wins += pnl
        else: losses += 1; gross_losses += abs(pnl)
        trade_list.append({"pnl": pnl, "balance": balance,
                           "reason": "END", "strategy": entry_strat})

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
        "trend_trades": trend_trades, "trend_wins": trend_wins,
        "breakout_trades": breakout_trades, "breakout_wins": breakout_wins,
        "pullback_trades": pullback_trades, "pullback_wins": pullback_wins,
    }


def print_result(label, r):
    if r is None:
        print(f"\n  {label}")
        print(f"  {'─'*60}")
        print(f"  No trades")
        return
    pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
    print(f"\n  {label}")
    print(f"  {'─'*60}")
    print(f"  Balance:       ${r['balance']:,.2f}  (P&L: ${r['pnl']:+.2f})")
    print(f"  Total trades:  {r['total']}  ({r['wins']}W / {r['losses']}L)")
    print(f"  Win rate:      {r['win_rate']:.1f}%")
    print(f"  Profit factor: {pf_str}")
    print(f"  Max drawdown:  {r['max_dd']*100:.1f}%")
    print(f"  Strategy breakdown:")
    if r['trend_trades'] > 0:
        tw = r['trend_wins'] / r['trend_trades'] * 100 if r['trend_trades'] > 0 else 0
        print(f"    Trend:    {r['trend_trades']:>4} trades  |  WR: {tw:.1f}%")
    if r['breakout_trades'] > 0:
        bw = r['breakout_wins'] / r['breakout_trades'] * 100 if r['breakout_trades'] > 0 else 0
        print(f"    Breakout: {r['breakout_trades']:>4} trades  |  WR: {bw:.1f}%")
    if r['pullback_trades'] > 0:
        pw = r['pullback_wins'] / r['pullback_trades'] * 100 if r['pullback_trades'] > 0 else 0
        print(f"    Pullback: {r['pullback_trades']:>4} trades  |  WR: {pw:.1f}%")


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    # HTF mapping: entry TF → regime TF
    htf_map = {
        "5m":  "15m",
        "15m": "1h",
        "30m": "1h",
        "1h":  "4h",
    }

    timeframes = ["5m", "15m", "30m", "1h"]

    print("\n" + "=" * 70)
    print("  CURRENT BOT BACKTEST")
    print(f"  3 strategies: Trend + Breakout + Pullback")
    print(f"  Exits: ATR stop ({CRYPTO_STOP_ATR}) | ATR TP ({CRYPTO_TP_ATR})")
    print(f"  Regime: EMA fast>slow + slow rising + ATR% >= {REGIME_MIN_ATR_PCT}")
    print(f"  Risk: {RISK*100:.0f}% per trade")
    print(f"  Data: {LOOKBACK} days ({LOOKBACK/365:.1f} years) from Binance US")
    print(f"  Coins: {', '.join(s.split('/')[0] for s in SYMBOLS)}")
    print("=" * 70)

    # ── Download all data ─────────────────────────────
    print(f"\nDownloading data...\n")

    all_data = {}
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        print(f"  {coin}:")
        all_data[sym] = {}
        for tf in timeframes:
            df = fetch_all_candles(sym, tf, LOOKBACK)
            all_data[sym][tf] = df

        # Also fetch HTF data for regime calculation
        # 4h is needed for 1h entry regime
        if len(all_data[sym]["1h"]) > 0:
            all_data[sym]["4h"] = resample(all_data[sym]["1h"], "4h")
            print(f"    4h (resampled): {len(all_data[sym]['4h'])} candles")
        print()

    # ── Compute regime for each entry/HTF pair ────────
    print("Computing regime filters for all timeframe combos...")
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

    # ── Run backtests ─────────────────────────────────
    all_results = {}

    for tf in timeframes:
        htf_tf = htf_map[tf]
        print(f"\n{'='*70}")
        print(f"  TIMEFRAME: {tf.upper()} entry  |  {htf_tf.upper()} regime filter")
        print(f"{'='*70}")

        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            d = all_data[sym]

            if tf not in d or len(d[tf]) < 50:
                print(f"\n  {coin} — not enough data")
                continue

            regime_key = f"{tf}_{htf_tf}"
            regime_df = regime_cache.get(sym, {}).get(regime_key)
            if regime_df is None:
                print(f"\n  {coin} — no regime data for {regime_key}")
                continue

            days_actual = (d[tf].index[-1] - d[tf].index[0]).days
            r = run_backtest(d[tf], regime_df)
            key = f"{coin}_{tf}"
            all_results[key] = r
            print_result(f"{coin} on {tf} ({days_actual}d / {days_actual/365:.1f}y)  |  Regime: {htf_tf}", r)

    # ── Summary table ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FULL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Coin':<5} {'TF':<5} {'HTF':<5} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>11} {'DD':>6}")
    print(f"  {'─'*60}")

    for tf in timeframes:
        htf_tf = htf_map[tf]
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            key = f"{coin}_{tf}"
            r = all_results.get(key)
            if r:
                pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
                print(f"  {coin:<5} {tf:<5} {htf_tf:<5} {r['total']:>7} {r['win_rate']:>6.1f}% "
                      f"{pf_str:>8} ${r['pnl']:>+9.2f} {r['max_dd']*100:>5.1f}%")
        print(f"  {'─'*60}")

    # ── Combined per timeframe ────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMBINED PER TIMEFRAME")
    print(f"{'='*70}")
    print(f"  {'TF':<6} {'HTF':<5} {'Trades':>8} {'Avg WR':>8} {'Avg PF':>8} {'Total P&L':>12}")
    print(f"  {'─'*50}")

    tf_combined = {}
    for tf in timeframes:
        htf_tf = htf_map[tf]
        trades = 0; wins = 0; pf_sum = 0; pnl = 0; count = 0
        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            key = f"{coin}_{tf}"
            r = all_results.get(key)
            if r and r["total"] >= 3:
                trades += r["total"]
                wins   += r["wins"]
                pf_sum += r["pf"]
                pnl    += r["pnl"]
                count  += 1
        if count > 0:
            avg_pf = pf_sum / count
            avg_wr = wins / trades * 100 if trades > 0 else 0
            tf_combined[tf] = {"trades": trades, "avg_wr": avg_wr,
                               "avg_pf": avg_pf, "pnl": pnl, "htf": htf_tf}
            pf_str = f"{avg_pf:.2f}" if avg_pf < 999 else "inf"
            print(f"  {tf:<6} {htf_tf:<5} {trades:>8} {avg_wr:>7.1f}% {pf_str:>8} ${pnl:>+10.2f}")

    # ── Strategy breakdown ────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STRATEGY BREAKDOWN (which entry type works best?)")
    print(f"{'='*70}")

    for strat in ["Trend", "Breakout", "Pullback"]:
        total_t = 0; total_w = 0
        for key, r in all_results.items():
            if r:
                t_key = f"{strat.lower()}_trades"
                w_key = f"{strat.lower()}_wins"
                total_t += r.get(t_key, 0)
                total_w += r.get(w_key, 0)
        if total_t > 0:
            wr = total_w / total_t * 100
            print(f"  {strat:<10}  {total_t:>6} trades  |  WR: {wr:.1f}%  ({total_w}W / {total_t - total_w}L)")
        else:
            print(f"  {strat:<10}  no trades")

    # ── Best per coin ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BEST TIMEFRAME PER COIN")
    print(f"{'='*70}")

    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        best_key = None; best_pf = 0
        for tf in timeframes:
            key = f"{coin}_{tf}"
            r = all_results.get(key)
            if r and r["total"] >= 10 and r["pf"] > best_pf:
                best_pf = r["pf"]; best_key = key
        if best_key:
            r = all_results[best_key]
            tf = best_key.split("_")[1]
            pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
            print(f"\n  {coin}: Best on {tf}")
            print(f"    PF: {pf_str}  |  WR: {r['win_rate']:.1f}%  |  "
                  f"Trades: {r['total']}  |  P&L: ${r['pnl']:+.2f}")

    # ── Final recommendation ──────────────────────────
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")

    if tf_combined:
        best_tf = max(tf_combined, key=lambda k: tf_combined[k]["avg_pf"])
        best = tf_combined[best_tf]
        pf_str = f"{best['avg_pf']:.2f}" if best['avg_pf'] < 999 else "inf"

        print(f"\n  BEST TIMEFRAME: {best_tf} entry + {best['htf']} regime")
        print(f"    Avg PF: {pf_str}  |  Trades: {best['trades']}  |  P&L: ${best['pnl']:+.2f}")

        print(f"\n  Compare to EMA+MACD only (previous test):")
        ema_macd_pf = {"5m": 0.82, "15m": 0.90, "30m": 1.07, "1h": 1.14}
        for tf in timeframes:
            if tf in tf_combined:
                curr = tf_combined[tf]
                prev_pf = ema_macd_pf.get(tf, 0)
                improvement = ((curr["avg_pf"] - prev_pf) / prev_pf * 100) if prev_pf > 0 else 0
                print(f"    {tf}: EMA+MACD PF {prev_pf:.2f} → Current bot PF {curr['avg_pf']:.2f} "
                      f"({'+'if improvement>0 else ''}{improvement:.0f}%)")

    print(f"\n{'='*70}")
    print(f"  Done!")
    print(f"{'='*70}\n")
