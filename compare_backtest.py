"""
═══════════════════════════════════════════════════════════
  OLD vs NEW CONFIG COMPARISON BACKTEST

  Tests the exact live-bot changes on 30m candles:

  OLD (current live bot):
    - HTF regime:  1h
    - Breakout:    regime_not_bearish filter
    - RSI filter:  none
    - Shorts:      allowed any time
    - Early exit:  SELL/COVER signal closes trades

  NEW (updated bot):
    - HTF regime:  4h  (stronger trend filter)
    - Breakout:    regime_up filter (tighter)
    - RSI filter:  < 65 for longs, > 35 for shorts
    - Shorts:      blocked when price > 200 EMA on 4h (macro bull)
    - Early exit:  removed — stop/TP only

  Coins: BTC, ETH, SOL
  Timeframe: 30m entry (live bot setting)
  Data: 2 years from Binance US

  Run:  python3 compare_backtest.py
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import ccxt
import time as time_module
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

SYMBOLS           = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ENTRY_TF          = "30m"
LOOKBACK          = 730   # 2 years

FAST_EMA          = 7
SLOW_EMA          = 18
RSI_PERIOD        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIG          = 9
ATR_PERIOD        = 14

STOP_ATR          = 1.2
TP_ATR            = 2.5
BREAKOUT_LOOKBACK = 10
BREAKOUT_VOL_MULT = 1.5
STRONG_BODY_MULT  = 1.2
REGIME_MIN_ATR_PCT = 0.002
PULLBACK_BUFFER   = 0.003

RISK              = 0.02
START_BAL         = 10_000.0

exchange = ccxt.binanceus()


def safe_float(v, d=0.0):
    try:
        return d if pd.isna(v) else float(v)
    except:
        return d


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

    df = pd.DataFrame(all_candles, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    days = (df.index[-1] - df.index[0]).days
    print(f" {len(df)} candles ({days}d / {days/365:.1f}y)")
    return df


def resample(df, tf):
    return df.resample(tf).agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna()


def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIG)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["atr"]     = ta.volatility.average_true_range(df["high"],df["low"],df["close"],window=ATR_PERIOD)
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


def compute_regime_series(entry_df, htf_df, include_macro=False):
    htf = add_indicators(htf_df.copy())

    # Optional 200-bar EMA for macro bull filter
    if include_macro:
        htf["ema_200"] = ta.trend.ema_indicator(htf["close"], window=200)

    htf["fast_above"]     = htf["ema_fast"] > htf["ema_slow"]
    htf["fast_below"]     = htf["ema_fast"] < htf["ema_slow"]
    htf["slow_rising"]    = htf["ema_slow"] > htf["ema_slow"].shift(1)
    htf["slow_falling"]   = htf["ema_slow"] < htf["ema_slow"].shift(1)
    htf["regime_up"]      = htf["fast_above"] & htf["slow_rising"] & (htf["atr_pct"] >= REGIME_MIN_ATR_PCT)
    htf["regime_down"]    = htf["fast_below"] & htf["slow_falling"] & (htf["atr_pct"] >= REGIME_MIN_ATR_PCT)
    htf["not_bearish"]    = htf["fast_above"] | htf["slow_rising"]
    htf["not_bullish"]    = htf["fast_below"] | htf["slow_falling"]
    if include_macro:
        htf["macro_bull"] = htf["close"] > htf["ema_200"]
    else:
        htf["macro_bull"] = False

    results = []
    htf_times = htf.index
    for ts in entry_df.index:
        mask = htf_times <= ts
        if mask.any():
            row = htf.iloc[mask.sum() - 1]
            results.append({
                "up":         bool(row["regime_up"]),
                "down":       bool(row["regime_down"]),
                "not_bearish": bool(row["not_bearish"]),
                "not_bullish": bool(row["not_bullish"]),
                "macro_bull": bool(row["macro_bull"]),
            })
        else:
            results.append({"up":False,"down":False,"not_bearish":False,"not_bullish":False,"macro_bull":False})
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
#  BACKTEST ENGINE — supports OLD and NEW config
# ═══════════════════════════════════════════════════════

def run_backtest(df, regime_df, config="new"):
    """
    config = "old": old bot logic (not_bearish breakout, no RSI filter, shorts always, SELL/COVER exits)
    config = "new": updated logic (regime_up breakout, RSI filter, macro_bull blocks shorts, stop/TP only)
    """
    df = add_indicators(df.copy())
    df = df.dropna()

    balance     = START_BAL
    in_trade    = False
    trade_dir   = None
    entry_price = 0.0
    stop_price  = 0.0
    tp_price    = 0.0
    coin_held   = 0.0
    entry_strat = ""
    highest     = 0.0
    lowest      = 999999.0

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
        rsi   = safe_float(last["rsi"], 50.0)

        # Regime lookup
        ts = df.index[i]
        if ts in regime_df.index:
            reg = regime_df.loc[ts]
        else:
            mask = regime_df.index <= ts
            if mask.any():
                reg = regime_df[mask].iloc[-1]
            else:
                prev_regime_up = False
                prev_regime_down = False
                continue

        regime_up        = bool(reg["up"])
        regime_down      = bool(reg["down"])
        regime_not_bearish = bool(reg["not_bearish"])
        regime_not_bullish = bool(reg["not_bullish"])
        macro_bull       = bool(reg["macro_bull"])

        regime_just_flipped_up   = regime_up   and not prev_regime_up
        regime_just_flipped_down = regime_down and not prev_regime_down

        # ── Exit ───────────────────────────────────────
        if in_trade:
            if trade_dir == "long":
                if price > highest: highest = price

                # OLD config: also exits on SELL signal (ema_cross_down + macd_bearish)
                if config == "old":
                    ema_cross_down_exit = (safe_float(prev["ema_fast"]) > safe_float(prev["ema_slow"]) and
                                           safe_float(last["ema_fast"]) < safe_float(last["ema_slow"]))
                    macd_bearish_exit   = (safe_float(last["macd"]) < safe_float(last["macd_signal"]) and
                                           safe_float(last["macd_hist"]) < 0)
                    if ema_cross_down_exit and macd_bearish_exit:
                        pnl = coin_held * (price - entry_price)
                        balance += coin_held * price
                        if pnl > 0: wins += 1; gross_wins += pnl
                        else: losses += 1; gross_losses += abs(pnl)
                        strat_counts.setdefault(entry_strat, [0,0])
                        strat_counts[entry_strat][0] += 1
                        if pnl > 0: strat_counts[entry_strat][1] += 1
                        trade_list.append({"pnl":pnl,"balance":balance,"strat":entry_strat,"exit":"signal"})
                        in_trade = False; coin_held = 0.0; highest = 0.0
                        prev_regime_up = regime_up; prev_regime_down = regime_down
                        continue

                if in_trade:
                    if price <= stop_price or price >= tp_price:
                        pnl = coin_held * (price - entry_price)
                        balance += coin_held * price
                        if pnl > 0: wins += 1; gross_wins += pnl
                        else: losses += 1; gross_losses += abs(pnl)
                        strat_counts.setdefault(entry_strat, [0,0])
                        strat_counts[entry_strat][0] += 1
                        if pnl > 0: strat_counts[entry_strat][1] += 1
                        trade_list.append({"pnl":pnl,"balance":balance,"strat":entry_strat,"exit":"sl/tp"})
                        in_trade = False; coin_held = 0.0; highest = 0.0

            elif trade_dir == "short":
                if price < lowest: lowest = price

                if config == "old":
                    ema_cross_up_exit = (safe_float(prev["ema_fast"]) < safe_float(prev["ema_slow"]) and
                                         safe_float(last["ema_fast"]) > safe_float(last["ema_slow"]))
                    macd_bullish_exit = (safe_float(last["macd"]) > safe_float(last["macd_signal"]) and
                                         safe_float(last["macd_hist"]) > 0)
                    if ema_cross_up_exit and macd_bullish_exit:
                        pnl = coin_held * (entry_price - price)
                        balance += coin_held * entry_price + pnl
                        if pnl > 0: wins += 1; gross_wins += pnl
                        else: losses += 1; gross_losses += abs(pnl)
                        strat_counts.setdefault(entry_strat, [0,0])
                        strat_counts[entry_strat][0] += 1
                        if pnl > 0: strat_counts[entry_strat][1] += 1
                        trade_list.append({"pnl":pnl,"balance":balance,"strat":entry_strat,"exit":"signal"})
                        in_trade = False; coin_held = 0.0; lowest = 999999.0
                        prev_regime_up = regime_up; prev_regime_down = regime_down
                        continue

                if in_trade:
                    if price >= stop_price or price <= tp_price:
                        pnl = coin_held * (entry_price - price)
                        balance += coin_held * entry_price + pnl
                        if pnl > 0: wins += 1; gross_wins += pnl
                        else: gross_losses += abs(pnl); losses += 1
                        strat_counts.setdefault(entry_strat, [0,0])
                        strat_counts[entry_strat][0] += 1
                        if pnl > 0: strat_counts[entry_strat][1] += 1
                        trade_list.append({"pnl":pnl,"balance":balance,"strat":entry_strat,"exit":"sl/tp"})
                        in_trade = False; coin_held = 0.0; lowest = 999999.0

            if in_trade:
                prev_regime_up = regime_up; prev_regime_down = regime_down
                continue

        # ── Entry signals ──────────────────────────────
        try:
            ema_cross_up   = (safe_float(prev["ema_fast"]) < safe_float(prev["ema_slow"]) and
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

            strategy  = None
            direction = None

            if config == "old":
                # ── OLD: no RSI filter, breakout uses not_bearish, shorts always allowed ──
                if ema_cross_up and macd_bullish and regime_up:
                    strategy = "Trend"; direction = "long"
                if not strategy:
                    if (price > safe_float(last["recent_high"]) and bool(last["strong_bullish"])
                            and bool(last["vol_spike"]) and regime_not_bearish):
                        strategy = "Breakout"; direction = "long"
                if not strategy:
                    prev_near = safe_float(prev["close"]) <= safe_float(prev["ema_fast"]) * (1 + PULLBACK_BUFFER)
                    rebound   = (bool(last["bullish_candle"]) and price > safe_float(last["ema_fast"])
                                 and price > safe_float(prev["high"]))
                    if regime_up and prev_near and rebound and macd_improving:
                        strategy = "Pullback"; direction = "long"
                if not strategy:
                    if regime_just_flipped_up and ema_already_above and macd_bullish:
                        strategy = "RegimeFlip"; direction = "long"
                if not strategy:
                    if ema_cross_down and macd_bearish and regime_down:
                        strategy = "ShortTrend"; direction = "short"
                if not strategy:
                    if (price < safe_float(last["recent_low"]) and bool(last["strong_bearish"])
                            and bool(last["vol_spike"]) and regime_not_bullish):
                        strategy = "ShortBreakout"; direction = "short"
                if not strategy:
                    prev_above = safe_float(prev["close"]) >= safe_float(prev["ema_fast"]) * (1 - PULLBACK_BUFFER)
                    rejection  = (bool(last["bearish_candle"]) and price < safe_float(last["ema_fast"])
                                  and price < safe_float(prev["low"]))
                    if regime_down and prev_above and rejection and macd_worsening:
                        strategy = "ShortPullback"; direction = "short"
                if not strategy:
                    if regime_just_flipped_down and ema_already_below and macd_bearish:
                        strategy = "ShortRegimeFlip"; direction = "short"

            else:  # config == "new"
                # ── NEW: RSI filter, breakout uses regime_up, shorts blocked in macro bull ──
                if ema_cross_up and macd_bullish and regime_up and rsi < 65:
                    strategy = "Trend"; direction = "long"
                if not strategy:
                    if (price > safe_float(last["recent_high"]) and bool(last["strong_bullish"])
                            and bool(last["vol_spike"]) and regime_up and rsi < 65):
                        strategy = "Breakout"; direction = "long"
                if not strategy:
                    prev_near = safe_float(prev["close"]) <= safe_float(prev["ema_fast"]) * (1 + PULLBACK_BUFFER)
                    rebound   = (bool(last["bullish_candle"]) and price > safe_float(last["ema_fast"])
                                 and price > safe_float(prev["high"]))
                    if regime_up and prev_near and rebound and macd_improving and rsi < 65:
                        strategy = "Pullback"; direction = "long"
                if not strategy:
                    if regime_just_flipped_up and ema_already_above and macd_bullish and rsi < 65:
                        strategy = "RegimeFlip"; direction = "long"
                if not macro_bull:
                    if not strategy:
                        if ema_cross_down and macd_bearish and regime_down and rsi > 35:
                            strategy = "ShortTrend"; direction = "short"
                    if not strategy:
                        if (price < safe_float(last["recent_low"]) and bool(last["strong_bearish"])
                                and bool(last["vol_spike"]) and regime_not_bullish and rsi > 35):
                            strategy = "ShortBreakout"; direction = "short"
                    if not strategy:
                        prev_above = safe_float(prev["close"]) >= safe_float(prev["ema_fast"]) * (1 - PULLBACK_BUFFER)
                        rejection  = (bool(last["bearish_candle"]) and price < safe_float(last["ema_fast"])
                                      and price < safe_float(prev["low"]))
                        if regime_down and prev_above and rejection and macd_worsening and rsi > 35:
                            strategy = "ShortPullback"; direction = "short"
                    if not strategy:
                        if regime_just_flipped_down and ema_already_below and macd_bearish and rsi > 35:
                            strategy = "ShortRegimeFlip"; direction = "short"

            if strategy and direction and atr > 0:
                spend = calc_position_size(balance, price, atr)
                if spend >= 1.0 and spend <= balance:
                    coin_held   = spend / price
                    entry_price = price
                    entry_strat = strategy
                    trade_dir   = direction
                    in_trade    = True
                    strat_counts.setdefault(strategy, [0,0])
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

        prev_regime_up   = regime_up
        prev_regime_down = regime_down

    # Close open trade at end
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
        strat_counts.setdefault(entry_strat, [0,0])
        strat_counts[entry_strat][0] += 1
        if pnl > 0: strat_counts[entry_strat][1] += 1
        trade_list.append({"pnl":pnl,"balance":balance,"strat":entry_strat,"exit":"open"})

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
        print(f"  {label}: No trades")
        return
    pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
    print(f"  {label}")
    print(f"    Trades: {r['total']} ({r['wins']}W / {r['losses']}L)  |  WR: {r['win_rate']:.1f}%  |  PF: {pf_str}  |  P&L: ${r['pnl']:+,.2f}  |  MaxDD: {r['max_dd']*100:.1f}%")
    for strat, (cnt, w) in sorted(r["strat_counts"].items()):
        wr = w/cnt*100 if cnt > 0 else 0
        print(f"      {strat:<18} {cnt:>4} trades  WR: {wr:.1f}%")


if __name__ == "__main__":
    print("\n" + "="*75)
    print("  OLD vs NEW CONFIG — 30m COMPARISON BACKTEST")
    print(f"  OLD: 1h HTF, not_bearish Breakout, no RSI, shorts always, SELL/COVER exits")
    print(f"  NEW: 4h HTF, regime_up Breakout, RSI<65/RSI>35, macro_bull blocks shorts, stop/TP only")
    print(f"  Coins: {', '.join(s.split('/')[0] for s in SYMBOLS)}  |  TF: 30m  |  Data: 2 years")
    print("="*75)

    # Download 30m data + both regime timeframes
    print("\nDownloading 30m data...")
    data_30m = {}
    for sym in SYMBOLS:
        df = fetch_all_candles(sym, "30m", LOOKBACK)
        data_30m[sym] = df

    print("\nDownloading 1h data (for OLD regime)...")
    data_1h = {}
    for sym in SYMBOLS:
        df = fetch_all_candles(sym, "1h", LOOKBACK)
        data_1h[sym] = df

    print("\nBuilding 4h data (resampled from 1h, for NEW regime)...")
    data_4h = {}
    for sym in SYMBOLS:
        if len(data_1h[sym]) > 0:
            data_4h[sym] = resample(data_1h[sym], "4h")
            coin = sym.split("/")[0]
            print(f"    {coin} 4h: {len(data_4h[sym])} candles")

    print("\nComputing regimes...")
    regimes_old = {}  # 30m entry → 1h HTF, no macro filter
    regimes_new = {}  # 30m entry → 4h HTF, with macro filter
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        if len(data_30m[sym]) > 50 and len(data_1h[sym]) > 20:
            regimes_old[sym] = compute_regime_series(data_30m[sym], data_1h[sym], include_macro=False)
            print(f"    {coin} old regime (1h): {len(regimes_old[sym])} rows")
        if len(data_30m[sym]) > 50 and len(data_4h.get(sym, pd.DataFrame())) > 200:
            regimes_new[sym] = compute_regime_series(data_30m[sym], data_4h[sym], include_macro=True)
            print(f"    {coin} new regime (4h+macro): {len(regimes_new[sym])} rows")

    print("\n" + "="*75)
    print("  RUNNING BACKTESTS...")
    print("="*75)

    results = {"old": {}, "new": {}}
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        df = data_30m[sym]
        if len(df) < 100:
            continue
        print(f"\n  {coin}:")
        if sym in regimes_old:
            r = run_backtest(df, regimes_old[sym], config="old")
            results["old"][sym] = r
            print_result("OLD", r)
        if sym in regimes_new:
            r = run_backtest(df, regimes_new[sym], config="new")
            results["new"][sym] = r
            print_result("NEW", r)

    # ── AGGREGATE COMPARISON ──────────────────────────
    print("\n" + "="*75)
    print("  AGGREGATE RESULTS (all 3 coins combined)")
    print("="*75)

    for cfg in ["old", "new"]:
        tot_trades = tot_wins = tot_losses = 0
        tot_pnl = gw = gl = max_dd = 0.0
        for sym in SYMBOLS:
            r = results[cfg].get(sym)
            if r:
                tot_trades += r["total"]
                tot_wins   += r["wins"]
                tot_losses += r["losses"]
                tot_pnl    += r["pnl"]
                max_dd      = max(max_dd, r["max_dd"])
        if tot_trades > 0:
            wr  = tot_wins / tot_trades * 100
            label = "OLD config" if cfg == "old" else "NEW config"
            print(f"\n  {label}:")
            print(f"    Total trades:  {tot_trades}  ({tot_wins}W / {tot_losses}L)")
            print(f"    Win rate:      {wr:.1f}%")
            print(f"    Total P&L:     ${tot_pnl:+,.2f}")
            print(f"    Max drawdown:  {max_dd*100:.1f}%")

    # ── VERDICT ──────────────────────────────────────
    old_pnl = sum(r["pnl"] for r in results["old"].values() if r)
    new_pnl = sum(r["pnl"] for r in results["new"].values() if r)
    old_wr  = (sum(r["wins"] for r in results["old"].values() if r) /
               max(sum(r["total"] for r in results["old"].values() if r), 1)) * 100
    new_wr  = (sum(r["wins"] for r in results["new"].values() if r) /
               max(sum(r["total"] for r in results["new"].values() if r), 1)) * 100

    print(f"\n{'='*75}")
    print(f"  VERDICT")
    print(f"{'='*75}")
    print(f"  OLD → WR: {old_wr:.1f}%  |  P&L: ${old_pnl:+,.2f}")
    print(f"  NEW → WR: {new_wr:.1f}%  |  P&L: ${new_pnl:+,.2f}")
    pnl_diff = new_pnl - old_pnl
    wr_diff  = new_wr - old_wr
    if new_pnl > old_pnl:
        print(f"\n  ✅ NEW config is BETTER by ${pnl_diff:+,.2f} P&L  (+{wr_diff:.1f}% WR)")
    else:
        print(f"\n  ❌ NEW config is worse by ${abs(pnl_diff):,.2f}")
    print(f"{'='*75}\n")
