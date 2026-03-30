"""
═══════════════════════════════════════════════════════════
  TRADINGVIEW EMA CROSSOVER BOT — EXACT REPLICA BACKTEST
  
  Replicates your TradingView Pine Script exactly:
    BUY:  EMA 7 crosses ABOVE EMA 18 + RSI < 70
    SELL: EMA 7 crosses BELOW EMA 18 + RSI > 30
  
  Tests both LONG and SHORT:
    LONG:  Buy on crossover, close on crossunder
    SHORT: Short on crossunder, close on crossover
  
  NO other filters — no MACD, no regime, no volume, no ATR stops.
  Exit = opposite crossover signal (not SL/TP based).
  
  Coins: BTC, ETH, SOL
  Timeframes: 15m, 30m, 1h
  Data: 4 years from Binance US
  
  Run:  python3 tv_ema_backtest.py
═══════════════════════════════════════════════════════════
"""

import pandas as pd
import ta
import ccxt
import time as time_module
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

SYMBOLS    = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
FAST_EMA   = 7
SLOW_EMA   = 18
RSI_PERIOD = 14
RSI_BUY_MAX  = 70   # buy only when RSI < 70
RSI_SELL_MIN = 30   # sell only when RSI > 30
RISK       = 0.10   # 10% of equity per trade (matches TV default_qty_value=10)
START_BAL  = 10_000.0
LOOKBACK   = 1460

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


def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    return df


def run_backtest(df, mode="long_only"):
    """
    Exact TradingView EMA Crossover Bot logic:
      BUY:   EMA 7 crosses above EMA 18 + RSI < 70
      SELL:  EMA 7 crosses below EMA 18 + RSI > 30
    
    mode: "long_only", "short_only", "both"
    
    Exit = opposite signal (no SL/TP, just like TradingView strategy)
    """
    df = add_indicators(df.copy())
    df = df.dropna()

    balance     = START_BAL
    in_long     = False
    in_short    = False
    entry_price = 0.0
    coin_held   = 0.0
    
    long_wins = long_losses = 0
    short_wins = short_losses = 0
    long_gross_wins = long_gross_losses = 0.0
    short_gross_wins = short_gross_losses = 0.0
    trade_list = []

    for i in range(2, len(df)):
        prev = df.iloc[i - 1]
        last = df.iloc[i]
        price = safe_float(last["close"])
        rsi   = safe_float(last["rsi"])

        # Crossover signals (exact match to Pine Script)
        cross_up   = (safe_float(prev["ema_fast"]) < safe_float(prev["ema_slow"]) and
                      safe_float(last["ema_fast"]) > safe_float(last["ema_slow"]))
        cross_down = (safe_float(prev["ema_fast"]) > safe_float(prev["ema_slow"]) and
                      safe_float(last["ema_fast"]) < safe_float(last["ema_slow"]))

        buy_signal  = cross_up and rsi < RSI_BUY_MAX
        sell_signal = cross_down and rsi > RSI_SELL_MIN

        # ── Close existing positions on opposite signal ──
        if in_long and sell_signal:
            proceeds = coin_held * price
            pnl = proceeds - (coin_held * entry_price)
            balance += proceeds
            if pnl >= 0:
                long_wins += 1; long_gross_wins += pnl
            else:
                long_losses += 1; long_gross_losses += abs(pnl)
            trade_list.append({"pnl": pnl, "balance": balance, "dir": "long"})
            in_long = False; coin_held = 0.0

        if in_short and buy_signal:
            pnl = coin_held * (entry_price - price)
            balance += coin_held * entry_price + pnl
            if pnl >= 0:
                short_wins += 1; short_gross_wins += pnl
            else:
                short_losses += 1; short_gross_losses += abs(pnl)
            trade_list.append({"pnl": pnl, "balance": balance, "dir": "short"})
            in_short = False; coin_held = 0.0

        # ── Open new positions ──
        if buy_signal and not in_long and not in_short:
            if mode in ("long_only", "both"):
                spend = balance * RISK
                if spend >= 1.0 and spend <= balance:
                    coin_held   = spend / price
                    balance    -= spend
                    entry_price = price
                    in_long     = True

        if sell_signal and not in_short and not in_long:
            if mode in ("short_only", "both"):
                spend = balance * RISK
                if spend >= 1.0 and spend <= balance:
                    coin_held   = spend / price
                    balance    -= spend  # collateral
                    entry_price = price
                    in_short    = True

    # Close any open position at end
    if in_long:
        price = safe_float(df.iloc[-1]["close"])
        proceeds = coin_held * price
        pnl = proceeds - (coin_held * entry_price)
        balance += proceeds
        if pnl >= 0: long_wins += 1; long_gross_wins += pnl
        else: long_losses += 1; long_gross_losses += abs(pnl)
        trade_list.append({"pnl": pnl, "balance": balance, "dir": "long"})

    if in_short:
        price = safe_float(df.iloc[-1]["close"])
        pnl = coin_held * (entry_price - price)
        balance += coin_held * entry_price + pnl
        if pnl >= 0: short_wins += 1; short_gross_wins += pnl
        else: short_losses += 1; short_gross_losses += abs(pnl)
        trade_list.append({"pnl": pnl, "balance": balance, "dir": "short"})

    total_wins = long_wins + short_wins
    total_losses = long_losses + short_losses
    total = total_wins + total_losses
    if total < 1:
        return None

    total_gross_wins = long_gross_wins + short_gross_wins
    total_gross_losses = long_gross_losses + short_gross_losses

    peak = START_BAL; max_dd = 0.0
    for t in trade_list:
        if t["balance"] > peak: peak = t["balance"]
        dd = (peak - t["balance"]) / peak
        if dd > max_dd: max_dd = dd

    long_total = long_wins + long_losses
    short_total = short_wins + short_losses

    return {
        "wins": total_wins, "losses": total_losses, "total": total,
        "win_rate": total_wins / total * 100,
        "pf": total_gross_wins / total_gross_losses if total_gross_losses > 0 else 999,
        "pnl": balance - START_BAL, "max_dd": max_dd,
        "long_trades": long_total, "long_wins": long_wins,
        "long_pf": long_gross_wins / long_gross_losses if long_gross_losses > 0 else 999,
        "short_trades": short_total, "short_wins": short_wins,
        "short_pf": short_gross_wins / short_gross_losses if short_gross_losses > 0 else 999,
        "avg_win": total_gross_wins / total_wins if total_wins > 0 else 0,
        "avg_loss": total_gross_losses / total_losses if total_losses > 0 else 0,
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
    print(f"  Avg win: ${r['avg_win']:.2f}  |  Avg loss: ${r['avg_loss']:.2f}")
    if r['long_trades'] > 0:
        long_wr = r['long_wins'] / r['long_trades'] * 100
        lpf = f"{r['long_pf']:.2f}" if r['long_pf'] < 999 else "inf"
        print(f"  Longs:  {r['long_trades']} trades  |  WR: {long_wr:.1f}%  |  PF: {lpf}")
    if r['short_trades'] > 0:
        short_wr = r['short_wins'] / r['short_trades'] * 100
        spf = f"{r['short_pf']:.2f}" if r['short_pf'] < 999 else "inf"
        print(f"  Shorts: {r['short_trades']} trades  |  WR: {short_wr:.1f}%  |  PF: {spf}")


if __name__ == "__main__":
    timeframes = ["15m", "30m", "1h"]
    modes = [
        ("long_only",  "LONG ONLY"),
        ("short_only", "SHORT ONLY"),
        ("both",       "LONG + SHORT"),
    ]

    print("\n" + "=" * 75)
    print("  TRADINGVIEW EMA CROSSOVER BOT — EXACT BACKTEST")
    print(f"  BUY:  EMA {FAST_EMA} crosses above EMA {SLOW_EMA} + RSI < {RSI_BUY_MAX}")
    print(f"  SELL: EMA {FAST_EMA} crosses below EMA {SLOW_EMA} + RSI > {RSI_SELL_MIN}")
    print(f"  Exit: opposite crossover signal (no SL/TP)")
    print(f"  Size: {RISK*100:.0f}% of equity per trade")
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
        print()

    # Run all
    all_results = {}
    config_num = 0
    total_configs = len(timeframes) * len(SYMBOLS) * len(modes)

    for tf in timeframes:
        print(f"{'='*75}")
        print(f"  TIMEFRAME: {tf.upper()}")
        print(f"{'='*75}")

        for sym in SYMBOLS:
            coin = sym.split("/")[0]
            df = all_data[sym][tf]
            if len(df) < 50:
                continue

            for mode_key, mode_label in modes:
                config_num += 1
                pct = config_num / total_configs * 100
                filled = int(30 * config_num / total_configs)
                bar = "█" * filled + "░" * (30 - filled)
                print(f"  [{bar}] {pct:>5.1f}%  {coin} {tf} {mode_key}", flush=True)

                r = run_backtest(df, mode_key)
                key = f"{coin}_{tf}_{mode_key}"
                all_results[key] = r

        # Print results per TF
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
    print(f"  {'Config':<35} {'Trades':>7} {'WR':>7} {'PF':>7} {'P&L':>12} {'DD':>6}")
    print(f"  {'─'*75}")

    for tf in timeframes:
        for mode_key, mode_label in modes:
            trades = 0; pnl = 0; pf_sum = 0; count = 0; w = 0; max_dd = 0
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
                print(f"  {tf} {mode_label:<30} {trades:>7} {avg_wr:>6.1f}% "
                      f"{pf_str:>7} ${pnl:>+10.2f} {max_dd*100:>5.1f}%")
        print(f"  {'─'*75}")

    # ── COMPARE TO YOUR BOT ──────────────────────────
    print(f"\n{'='*75}")
    print(f"  TV EMA BOT vs YOUR CURRENT BOT (15m)")
    print(f"{'='*75}")

    tv_15m_pnl = 0
    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        r = all_results.get(f"{coin}_15m_both")
        if r: tv_15m_pnl += r["pnl"]

    print(f"\n  TradingView EMA bot (15m, long+short):  ${tv_15m_pnl:>+12.2f}")
    print(f"  Your bot (15m, all 8 strategies):        $+178,513.98  (from backtest)")
    print(f"  Your bot (15m, long only, 4 strategies): $ +78,334.35  (from backtest)")

    if tv_15m_pnl > 0:
        ratio = 178513.98 / tv_15m_pnl if tv_15m_pnl > 0 else 999
        print(f"\n  Your bot makes {ratio:.1f}x more than the TV EMA bot")
    else:
        print(f"\n  TV EMA bot loses money — your bot is significantly better")

    # ── PER COIN BEST ─────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  BEST TIMEFRAME PER COIN")
    print(f"{'='*75}")

    for sym in SYMBOLS:
        coin = sym.split("/")[0]
        best_key = None; best_pnl = -999999
        for tf in timeframes:
            for mode_key, _ in modes:
                key = f"{coin}_{tf}_{mode_key}"
                r = all_results.get(key)
                if r and r["pnl"] > best_pnl:
                    best_pnl = r["pnl"]; best_key = key
        if best_key:
            r = all_results[best_key]
            parts = best_key.split("_")
            tf_label = parts[1]
            mode_label = parts[2]
            pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "inf"
            print(f"\n  {coin}: Best on {tf_label} {mode_label}")
            print(f"    PF: {pf_str}  |  WR: {r['win_rate']:.1f}%  |  "
                  f"Trades: {r['total']}  |  P&L: ${r['pnl']:+.2f}")

    print(f"\n{'='*75}")
    print(f"  Done!")
    print(f"{'='*75}\n")
