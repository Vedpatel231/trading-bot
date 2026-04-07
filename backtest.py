"""
╔══════════════════════════════════════════════════════════════════════╗
║  BACKTEST ENGINE  —  Trailing TP/SL vs Fixed TP/SL                  ║
║                                                                       ║
║  Fetches 2 years of 15-min OHLCV from Binance US (ccxt),             ║
║  runs every strategy signal, and compares:                            ║
║    • OLD: fixed stop (1×ATR) + fixed TP (2.5×ATR) + breakeven lock   ║
║    • NEW: 3-stage trailing TP/SL + 50% partial close                  ║
║                                                                       ║
║  Usage:  python3 backtest.py                                          ║
║  Output: backtest_results.csv  +  console summary                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys, os, time, logging, warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Try to import ta / ccxt (same deps as bot.py) ──────────────────────────
try:
    import ta
except ImportError:
    print("❌  Missing: pip install ta")
    sys.exit(1)

try:
    import ccxt
except ImportError:
    print("❌  Missing: pip install ccxt")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG  (mirrors bot.py settings)
# ══════════════════════════════════════════════════════════════════════
CRYPTO_SYMBOLS   = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CRYPTO_TF        = "15m"
CRYPTO_HTF       = "4h"

FAST_EMA         = 9
SLOW_EMA         = 21
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
ATR_PERIOD       = 14
BREAKOUT_LOOKBACK= 20
BREAKOUT_VOL_MULT= 1.5
STRONG_BODY_MULT = 1.2
REGIME_MIN_ATR_PCT=0.003

CRYPTO_STOP_ATR  = 1.0       # stop = entry ± 1×ATR
CRYPTO_TP_ATR    = 2.5       # old fixed TP = entry ± 2.5×ATR
PARTIAL_ATR      = 1.5       # new: sell 50% at 1.5×ATR profit
TRAIL_ATR        = 1.0       # trailing distance = 1×ATR
ADX_MIN          = 23
CHOP_RANGE_MIN   = 0.004
TRADING_FEE_PCT  = 0.001     # 0.10% Binance US
RISK_PCT         = 0.02      # 2% risk per trade
MAX_POSITION_PCT = 0.25      # max 25% of balance per trade

STARTING_BALANCE = 10_000.0  # per symbol
YEARS_BACK       = 2

CACHE_DIR        = os.path.join(os.path.dirname(__file__), ".backtest_cache")

# ══════════════════════════════════════════════════════════════════════
#  EXCHANGE INIT
# ══════════════════════════════════════════════════════════════════════
exchange = ccxt.binanceus({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})


# ══════════════════════════════════════════════════════════════════════
#  DATA FETCH  (with disk cache to avoid re-fetching)
# ══════════════════════════════════════════════════════════════════════
def fetch_ohlcv_full(symbol: str, timeframe: str, years: int) -> pd.DataFrame:
    """Fetch full history in 1000-candle batches, cache to CSV."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_sym = symbol.replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, f"{safe_sym}_{timeframe}_{years}y.csv")

    # Use cache if it exists and is < 4 hours old
    if os.path.exists(cache_path):
        age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_h < 4:
            print(f"  📂  {symbol} {timeframe}: loading from cache …")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

    print(f"  🌐  {symbol} {timeframe}: fetching {years}y from Binance US …", end="", flush=True)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=365 * years)).timestamp() * 1000)
    all_bars  = []
    limit     = 1000

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except Exception as e:
            print(f"\n  ⚠️  fetch error: {e} — retrying in 5s")
            time.sleep(5)
            continue

        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000) - 2 * 60_000:
            break
        since_ms = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)
        print(".", end="", flush=True)

    df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(cache_path)
    print(f"  ✓  {len(df):,} candles")
    return df


# ══════════════════════════════════════════════════════════════════════
#  INDICATORS  (mirrors bot.py add_indicators)
# ══════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]   = ta.trend.ema_indicator(df["close"], window=FAST_EMA)
    df["ema_slow"]   = ta.trend.ema_indicator(df["close"], window=SLOW_EMA)
    df["rsi"]        = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["rsi_prev"]   = df["rsi"].shift(1)

    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df["macd"]       = macd.macd()
    df["macd_signal"]= macd.macd_signal()
    df["macd_hist"]  = macd.macd_diff()

    df["atr"]        = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=ATR_PERIOD)
    df["atr_pct"]    = df["atr"] / df["close"]

    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"]        = adx_ind.adx()

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"]       = (
        (df["typical_price"] * df["volume"]).rolling(20).sum()
        / df["volume"].rolling(20).sum()
    )
    df["chop_range"] = (df["high"].rolling(3).max() - df["low"].rolling(3).min()) / df["close"]

    df["vol_avg"]    = df["volume"].rolling(20).mean()
    df["vol_spike"]  = df["volume"] > df["vol_avg"] * BREAKOUT_VOL_MULT

    df["body"]          = (df["close"] - df["open"]).abs()
    df["body_avg"]      = df["body"].rolling(20).mean()
    df["bullish_candle"]= df["close"] > df["open"]
    df["bearish_candle"]= df["close"] < df["open"]
    df["strong_bullish"]= df["bullish_candle"] & (df["body"] > df["body_avg"] * STRONG_BODY_MULT)
    df["strong_bearish"]= df["bearish_candle"] & (df["body"] > df["body_avg"] * STRONG_BODY_MULT)
    df["recent_high"]   = df["high"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["recent_low"]    = df["low"].shift(1).rolling(BREAKOUT_LOOKBACK).min()
    return df


def get_htf_regime(df4h: pd.DataFrame) -> pd.DataFrame:
    """Return per-row regime flags from 4h data."""
    d = add_indicators(df4h).copy()
    d["fast_above"]  = d["ema_fast"] > d["ema_slow"]
    d["slow_rising"] = d["ema_slow"] > d["ema_slow"].shift(1)
    d["regime_up"]   = d["fast_above"] & d["slow_rising"] & (d["atr_pct"] >= REGIME_MIN_ATR_PCT)
    d["regime_down"] = (~d["fast_above"]) & (~d["slow_rising"]) & (d["atr_pct"] >= REGIME_MIN_ATR_PCT)
    ema200           = ta.trend.ema_indicator(d["close"], window=200)
    d["macro_bull"]  = d["close"] > ema200
    return d[["regime_up", "regime_down", "macro_bull"]]


def safe_float(v, default=0.0):
    try:
        return default if pd.isna(v) else float(v)
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION  (mirrors bot.py get_crypto_signal)
# ══════════════════════════════════════════════════════════════════════
def get_signal(row, regime_up: bool, regime_down: bool, macro_bull: bool):
    """Return (direction, strategy) or (None, None)."""
    price  = safe_float(row["close"])
    rsi    = safe_float(row["rsi"], 50.0)
    rsi_p  = safe_float(row["rsi_prev"], 50.0)
    adx    = safe_float(row["adx"], 0.0)
    vwap   = safe_float(row["vwap"])
    chop   = safe_float(row["chop_range"], 1.0)
    macd_h = safe_float(row["macd_hist"])
    macd_p = safe_float(row.get("macd_hist_prev", macd_h))
    vol_sp = bool(row["vol_spike"])

    adx_ok   = adx >= ADX_MIN
    not_chop = chop >= CHOP_RANGE_MIN
    above_vw = price > vwap and vwap > 0
    below_vw = price < vwap and vwap > 0

    fast_a = safe_float(row["ema_fast"]) > safe_float(row["ema_slow"])
    fast_b = safe_float(row["ema_fast"]) < safe_float(row["ema_slow"])

    # ── MomentumBurst long ──
    if (rsi_p < 42 and rsi >= 52 and above_vw and vol_sp
            and adx_ok and not_chop and regime_up and rsi < 68):
        return "long", "MomentumBurst"

    # ── MomentumBurst short ──
    if (rsi_p > 58 and rsi <= 48 and below_vw and vol_sp
            and adx_ok and not_chop and regime_down and rsi > 32 and not macro_bull):
        return "short", "MomentumBurst"

    # ── Trend long ──
    if (fast_a and safe_float(row["ema_slow"]) > safe_float(row.get("ema_slow_prev", row["ema_slow"]))
            and rsi >= 50 and rsi < 70 and macd_h > 0 and adx_ok and not_chop and regime_up and above_vw):
        return "long", "Trend"

    # ── Trend short ──
    if (fast_b and safe_float(row["ema_slow"]) < safe_float(row.get("ema_slow_prev", row["ema_slow"]))
            and rsi <= 50 and rsi > 30 and macd_h < 0 and adx_ok and not_chop and regime_down and below_vw
            and not macro_bull):
        return "short", "Trend"

    # ── Breakout long ──
    high_ok = safe_float(row["recent_high"]) > 0 and price > safe_float(row["recent_high"])
    if (high_ok and vol_sp and safe_float(row["strong_bullish"])
            and rsi >= 50 and rsi < 75 and adx_ok and not_chop and regime_up):
        return "long", "Breakout"

    # ── Breakout short ──
    low_ok = safe_float(row["recent_low"]) > 0 and price < safe_float(row["recent_low"])
    if (low_ok and vol_sp and safe_float(row["strong_bearish"])
            and rsi <= 50 and rsi > 25 and adx_ok and not_chop and regime_down and not macro_bull):
        return "short", "Breakout"

    # ── Pullback long ──
    ema_mid = (safe_float(row["ema_fast"]) + safe_float(row["ema_slow"])) / 2
    if (fast_a and price <= ema_mid * 1.003 and price >= ema_mid * 0.995
            and rsi >= 40 and rsi < 60 and adx_ok and not_chop and regime_up and above_vw):
        return "long", "Pullback"

    # ── Pullback short ──
    if (fast_b and price >= ema_mid * 0.997 and price <= ema_mid * 1.005
            and rsi <= 60 and rsi > 40 and adx_ok and not_chop and regime_down and not macro_bull):
        return "short", "Pullback"

    # ── RegimeFlip long ──
    if (not regime_up and fast_a and macd_h > 0 and macd_p <= 0
            and rsi >= 50 and adx_ok and not_chop):
        return "long", "RegimeFlip"

    # ── RegimeFlip short ──
    if (not regime_down and fast_b and macd_h < 0 and macd_p >= 0
            and rsi <= 50 and adx_ok and not_chop and not macro_bull):
        return "short", "RegimeFlip"

    return None, None


# ══════════════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════════════
def calc_qty(balance: float, entry: float, atr: float) -> float:
    risk_dollars = balance * RISK_PCT
    if atr <= 0:
        return 0.0
    raw_qty = risk_dollars / (CRYPTO_STOP_ATR * atr)
    max_qty = (balance * MAX_POSITION_PCT) / entry
    qty     = min(raw_qty, max_qty)
    cost    = qty * entry * (1 + TRADING_FEE_PCT)
    if cost > balance:
        qty = balance * 0.98 / (entry * (1 + TRADING_FEE_PCT))
    return max(qty, 0.0)


# ══════════════════════════════════════════════════════════════════════
#  TRADE SIMULATOR — OLD (fixed TP/SL + breakeven lock)
# ══════════════════════════════════════════════════════════════════════
class OldExitSim:
    """
    Stage 1 only:
      stop   = entry ∓ 1×ATR
      tp     = entry ± 2.5×ATR
      breakeven lock at ± 1×ATR
    """
    def __init__(self, direction, entry, atr, qty):
        self.direction = direction
        self.entry     = entry
        self.atr       = atr
        self.qty       = qty
        self.be_locked = False
        if direction == "long":
            self.stop = entry - CRYPTO_STOP_ATR * atr
            self.tp   = entry + CRYPTO_TP_ATR  * atr
        else:
            self.stop = entry + CRYPTO_STOP_ATR * atr
            self.tp   = entry - CRYPTO_TP_ATR  * atr

    def process_candle(self, o, h, l, c):
        """
        Returns (exit_price, reason) or (None, None) if still open.
        Uses conservative intra-candle ordering: stop checked before TP.
        """
        if self.direction == "long":
            # Breakeven lock
            if not self.be_locked and h >= self.entry + self.atr:
                if self.stop < self.entry:
                    self.stop = self.entry
                    self.be_locked = True
            # Check stop first (worst-case)
            if l <= self.stop:
                return max(self.stop, l), "stop"
            if h >= self.tp:
                return self.tp, "tp"
        else:
            if not self.be_locked and l <= self.entry - self.atr:
                if self.stop > self.entry:
                    self.stop = self.entry
                    self.be_locked = True
            if h >= self.stop:
                return min(self.stop, h), "stop"
            if l <= self.tp:
                return self.tp, "tp"
        return None, None

    def calc_pnl(self, exit_price, qty=None):
        q = qty or self.qty
        if self.direction == "long":
            fee = q * exit_price * TRADING_FEE_PCT
            return (q * exit_price - fee) - (q * self.entry)
        else:
            fee = q * exit_price * TRADING_FEE_PCT
            raw = q * (self.entry - exit_price)
            return raw - fee


# ══════════════════════════════════════════════════════════════════════
#  TRADE SIMULATOR — NEW (3-stage trailing + partial close)
# ══════════════════════════════════════════════════════════════════════
class NewExitSim:
    """
    Stage 1: fixed stop (1×ATR) / TP (2.5×ATR) + breakeven lock at 1×ATR
    Stage 2: partial close (50%) at 1.5×ATR profit → stop=entry, tp=high+1×ATR
    Stage 3: trail stop = highest-1×ATR, trail tp = highest+1×ATR
    """
    def __init__(self, direction, entry, atr, qty):
        self.direction  = direction
        self.entry      = entry
        self.atr        = atr
        self.qty        = qty
        self.be_locked  = False
        self.half_sold  = False
        self.highest    = entry   # for long
        self.lowest     = entry   # for short
        self.partial_pnl= 0.0
        if direction == "long":
            self.stop = entry - CRYPTO_STOP_ATR * atr
            self.tp   = entry + CRYPTO_TP_ATR  * atr
        else:
            self.stop = entry + CRYPTO_STOP_ATR * atr
            self.tp   = entry - CRYPTO_TP_ATR  * atr

    def _partial_pnl_calc(self, price):
        half = self.qty / 2
        if self.direction == "long":
            fee = half * price * TRADING_FEE_PCT
            return (half * price - fee) - (half * self.entry)
        else:
            fee = half * price * TRADING_FEE_PCT
            return (half * (self.entry - price)) - fee

    def process_candle(self, o, h, l, c):
        """
        Returns list of (exit_price, reason, qty_fraction) events this candle.
        qty_fraction: 0.5 = partial, 1.0 = full (or remaining 0.5 after partial).
        """
        events = []

        if self.direction == "long":
            # Update highest
            if h > self.highest:
                self.highest = h

            if not self.half_sold:
                # Breakeven lock
                if not self.be_locked and self.highest >= self.entry + self.atr:
                    if self.stop < self.entry:
                        self.stop = self.entry
                        self.be_locked = True

                # Stage 2: partial close at 1.5×ATR
                if self.highest >= self.entry + PARTIAL_ATR * self.atr:
                    partial_price = min(h, self.entry + PARTIAL_ATR * self.atr)
                    partial_price = max(partial_price, l)  # clamp to candle
                    self.partial_pnl = self._partial_pnl_calc(partial_price)
                    events.append((partial_price, "partial", 0.5))
                    self.half_sold = True
                    self.stop = self.entry
                    self.tp   = self.highest + TRAIL_ATR * self.atr
                    self.qty  = self.qty / 2
                    # Check stop/TP for remainder in same candle
                    if l <= self.stop:
                        events.append((max(self.stop, l), "trail_stop", 1.0))
                        return events
                    if h >= self.tp:
                        events.append((self.tp, "trail_tp", 1.0))
                        return events
                    return events

                # Stage 1 exits
                if l <= self.stop:
                    events.append((max(self.stop, l), "stop", 1.0))
                    return events
                if h >= self.tp:
                    events.append((self.tp, "tp", 1.0))
                    return events

            else:
                # Stage 3: update trailing
                new_stop = self.highest - TRAIL_ATR * self.atr
                new_tp   = self.highest + TRAIL_ATR * self.atr
                if new_stop > self.stop: self.stop = new_stop
                if new_tp   > self.tp:   self.tp   = new_tp
                if l <= self.stop:
                    events.append((max(self.stop, l), "trail_stop", 1.0))
                    return events
                if h >= self.tp:
                    events.append((self.tp, "trail_tp", 1.0))
                    return events

        else:  # short
            if l < self.lowest:
                self.lowest = l

            if not self.half_sold:
                if not self.be_locked and self.lowest <= self.entry - self.atr:
                    if self.stop > self.entry:
                        self.stop = self.entry
                        self.be_locked = True

                if self.lowest <= self.entry - PARTIAL_ATR * self.atr:
                    partial_price = max(l, self.entry - PARTIAL_ATR * self.atr)
                    partial_price = min(partial_price, h)
                    self.partial_pnl = self._partial_pnl_calc(partial_price)
                    events.append((partial_price, "partial", 0.5))
                    self.half_sold = True
                    self.stop = self.entry
                    self.tp   = self.lowest - TRAIL_ATR * self.atr
                    self.qty  = self.qty / 2
                    if h >= self.stop:
                        events.append((min(self.stop, h), "trail_stop", 1.0))
                        return events
                    if l <= self.tp:
                        events.append((self.tp, "trail_tp", 1.0))
                        return events
                    return events

                if h >= self.stop:
                    events.append((min(self.stop, h), "stop", 1.0))
                    return events
                if l <= self.tp:
                    events.append((self.tp, "tp", 1.0))
                    return events

            else:
                new_stop = self.lowest + TRAIL_ATR * self.atr
                new_tp   = self.lowest - TRAIL_ATR * self.atr
                if new_stop < self.stop: self.stop = new_stop
                if new_tp   < self.tp:   self.tp   = new_tp
                if h >= self.stop:
                    events.append((min(self.stop, h), "trail_stop", 1.0))
                    return events
                if l <= self.tp:
                    events.append((self.tp, "trail_tp", 1.0))
                    return events

        return events  # empty = still open

    def total_pnl(self, exit_price):
        """Full PnL of the remaining position at exit_price."""
        if self.direction == "long":
            fee = self.qty * exit_price * TRADING_FEE_PCT
            return (self.qty * exit_price - fee) - (self.qty * self.entry)
        else:
            fee = self.qty * exit_price * TRADING_FEE_PCT
            return self.qty * (self.entry - exit_price) - fee


# ══════════════════════════════════════════════════════════════════════
#  SINGLE SYMBOL BACKTEST
# ══════════════════════════════════════════════════════════════════════
def backtest_symbol(symbol: str, df15: pd.DataFrame, df4h: pd.DataFrame):
    """Run both OLD and NEW strategies on `symbol`. Return list of trade dicts."""
    print(f"\n  ▶  {symbol}: adding indicators …")
    df = add_indicators(df15.copy())
    df["macd_hist_prev"]= df["macd_hist"].shift(1)
    df["ema_slow_prev"] = df["ema_slow"].shift(1)
    df = df.dropna(subset=["ema_fast","ema_slow","rsi","atr","adx","vwap","chop_range"])

    # Build 4h regime indexed to 15m
    print(f"  ▶  {symbol}: computing HTF regime …")
    regime4h = get_htf_regime(df4h)
    # Forward-fill regime onto 15m index
    regime_reindexed = regime4h.reindex(df.index, method="ffill")

    trades_old = []
    trades_new = []

    bal_old = STARTING_BALANCE
    bal_new = STARTING_BALANCE

    in_old  = False
    in_new  = False
    sim_old = None
    sim_new = None
    entry_bar_old = None
    entry_bar_new = None
    cooldown_old = None
    cooldown_new = None
    COOLDOWN_MIN = 45

    rows = list(df.itertuples())
    n    = len(rows)
    print(f"  ▶  {symbol}: simulating {n:,} candles …")

    for i, row in enumerate(rows):
        ts  = row.Index
        o   = float(row.open)
        h   = float(row.high)
        l   = float(row.low)
        c   = float(row.close)
        atr = float(row.atr) if row.atr and not pd.isna(row.atr) else 0.0

        # Regime at this timestamp
        reg_row = regime_reindexed.loc[ts] if ts in regime_reindexed.index else pd.Series({"regime_up":False,"regime_down":False,"macro_bull":True})
        r_up    = bool(reg_row.get("regime_up",  False))
        r_down  = bool(reg_row.get("regime_down",False))
        r_macro = bool(reg_row.get("macro_bull", True))

        # ── OLD strategy ─────────────────────────────────────────────
        if in_old:
            events = sim_old.process_candle(o, h, l, c)
            # Only care about the last exit event (first = partial for old, not applicable)
            final_exit = next((e for e in reversed(events) if e[1] != "partial"), None)
            if final_exit:
                xprice, xreason, xfrac = final_exit
                pnl = sim_old.calc_pnl(xprice)
                bal_old += pnl
                trades_old.append({
                    "symbol": symbol, "entry_ts": entry_bar_old,
                    "exit_ts": ts, "direction": sim_old.direction,
                    "strategy": sim_old.strategy if hasattr(sim_old,"strategy") else "",
                    "entry": sim_old.entry, "exit": xprice, "atr": atr,
                    "pnl": round(pnl, 4), "reason": xreason,
                    "half_sold": False, "partial_pnl": 0.0,
                    "balance": round(bal_old, 2),
                })
                in_old = False
                if pnl < 0:
                    cooldown_old = ts

        if in_new:
            events = sim_new.process_candle(o, h, l, c)
            partial_pnl_this = 0.0
            full_exit = None
            for ev in events:
                xp, xr, xf = ev
                if xr == "partial":
                    partial_pnl_this = sim_new.partial_pnl
                else:
                    full_exit = ev

            if full_exit:
                xprice, xreason, _ = full_exit
                remainder_pnl = sim_new.total_pnl(xprice)
                total_pnl = partial_pnl_this + sim_new.partial_pnl + remainder_pnl
                bal_new += total_pnl
                trades_new.append({
                    "symbol": symbol, "entry_ts": entry_bar_new,
                    "exit_ts": ts, "direction": sim_new.direction,
                    "strategy": sim_new.strategy if hasattr(sim_new,"strategy") else "",
                    "entry": sim_new.entry, "exit": xprice, "atr": atr,
                    "pnl": round(total_pnl, 4), "reason": xreason,
                    "half_sold": sim_new.half_sold,
                    "partial_pnl": round(sim_new.partial_pnl, 4),
                    "balance": round(bal_new, 2),
                })
                in_new = False
                if total_pnl < 0:
                    cooldown_new = ts
            elif events:  # only partial happened this candle, no full exit
                bal_new += partial_pnl_this  # bank partial leg
                sim_new.partial_pnl = partial_pnl_this  # remember for final close

        # ── Cooldown check ──
        def _can_trade(cooldown_ts, ts):
            if cooldown_ts is None: return True
            return (ts - cooldown_ts).total_seconds() / 60 >= COOLDOWN_MIN

        # ── Entry signal ─────────────────────────────────────────────
        if i + 1 < n and atr > 0:
            try:
                row_dict = row._asdict()
                direction, strategy = get_signal(row_dict, r_up, r_down, r_macro)
            except Exception:
                direction, strategy = None, None

            if direction:
                # Next candle open = entry price
                next_row = rows[i + 1]
                entry_price = float(next_row.open)
                entry_atr   = atr

                # OLD entry
                if not in_old and _can_trade(cooldown_old, ts) and bal_old > 100:
                    qty = calc_qty(bal_old, entry_price, entry_atr)
                    if qty > 0:
                        cost = qty * entry_price * (1 + TRADING_FEE_PCT)
                        if cost <= bal_old:
                            bal_old -= cost
                            sim_old = OldExitSim(direction, entry_price, entry_atr, qty)
                            sim_old.strategy = strategy
                            in_old = True
                            entry_bar_old = next_row.Index

                # NEW entry
                if not in_new and _can_trade(cooldown_new, ts) and bal_new > 100:
                    qty = calc_qty(bal_new, entry_price, entry_atr)
                    if qty > 0:
                        cost = qty * entry_price * (1 + TRADING_FEE_PCT)
                        if cost <= bal_new:
                            bal_new -= cost
                            sim_new = NewExitSim(direction, entry_price, entry_atr, qty)
                            sim_new.strategy = strategy
                            in_new = True
                            entry_bar_new = next_row.Index

    return trades_old, trades_new, bal_old, bal_new


# ══════════════════════════════════════════════════════════════════════
#  STATISTICS
# ══════════════════════════════════════════════════════════════════════
def calc_stats(trades: list, final_balance: float, label: str) -> dict:
    if not trades:
        return {"label": label, "trades": 0}

    df   = pd.DataFrame(trades)
    pnls = df["pnl"].tolist()
    wins = [p for p in pnls if p >= 0]
    loss = [p for p in pnls if p < 0]

    # Running balance for drawdown
    bal_curve = df["balance"].tolist()
    peak = bal_curve[0]
    max_dd = 0.0
    for b in bal_curve:
        if b > peak: peak = b
        dd = (peak - b) / peak
        if dd > max_dd: max_dd = dd

    # Sharpe-like (daily PnL std)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"])
    daily = df.groupby(df["exit_ts"].dt.date)["pnl"].sum()
    sharpe = (daily.mean() / daily.std() * (252 ** 0.5)) if daily.std() > 0 else 0.0

    # Win streak / loss streak
    streak_w = streak_l = cur = 0
    for p in pnls:
        if p >= 0:
            cur = cur + 1 if cur >= 0 else 1
            streak_w = max(streak_w, cur)
        else:
            cur = cur - 1 if cur <= 0 else -1
            streak_l = max(streak_l, -cur)

    # Strategy breakdown
    strat_grp = df.groupby("strategy")["pnl"].agg(["count","sum","mean"])

    return {
        "label":         label,
        "trades":        len(pnls),
        "win_rate":      round(len(wins) / len(pnls) * 100, 1),
        "total_pnl":     round(sum(pnls), 2),
        "return_pct":    round(sum(pnls) / STARTING_BALANCE * 100, 2),
        "avg_win":       round(np.mean(wins), 2) if wins else 0.0,
        "avg_loss":      round(np.mean(loss), 2) if loss else 0.0,
        "profit_factor": round(sum(wins) / abs(sum(loss)), 2) if loss else float("inf"),
        "max_drawdown":  round(max_dd * 100, 2),
        "sharpe":        round(sharpe, 2),
        "best_trade":    round(max(pnls), 2),
        "worst_trade":   round(min(pnls), 2),
        "streak_win":    streak_w,
        "streak_loss":   streak_l,
        "final_balance": round(final_balance, 2),
        "strat_breakdown": strat_grp.to_dict(),
    }


def print_comparison(old_stats: dict, new_stats: dict, symbol: str):
    def fmt(v, is_pct=False, is_dollar=False):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "  N/A"
        if is_dollar: return f"${v:>10,.2f}"
        if is_pct:    return f"{v:>9.1f}%"
        return f"{v:>10}"

    SEP = "─" * 68
    print(f"\n{'═'*68}")
    print(f"  {symbol}  —  OLD fixed TP/SL  vs  NEW trailing TP/SL")
    print(f"{'═'*68}")
    print(f"  {'Metric':<26}  {'OLD (fixed)':>14}  {'NEW (trailing)':>14}")
    print(SEP)

    rows = [
        ("Trades",              old_stats.get("trades"),     new_stats.get("trades"),     False, False),
        ("Win rate",            old_stats.get("win_rate"),   new_stats.get("win_rate"),   True,  False),
        ("Total P&L",           old_stats.get("total_pnl"),  new_stats.get("total_pnl"),  False, True),
        ("Return %",            old_stats.get("return_pct"), new_stats.get("return_pct"), True,  False),
        ("Profit factor",       old_stats.get("profit_factor"),new_stats.get("profit_factor"),False,False),
        ("Avg win",             old_stats.get("avg_win"),    new_stats.get("avg_win"),    False, True),
        ("Avg loss",            old_stats.get("avg_loss"),   new_stats.get("avg_loss"),   False, True),
        ("Best trade",          old_stats.get("best_trade"), new_stats.get("best_trade"), False, True),
        ("Worst trade",         old_stats.get("worst_trade"),new_stats.get("worst_trade"),False, True),
        ("Max drawdown",        old_stats.get("max_drawdown"),new_stats.get("max_drawdown"),True, False),
        ("Sharpe ratio",        old_stats.get("sharpe"),     new_stats.get("sharpe"),     False, False),
        ("Longest win streak",  old_stats.get("streak_win"), new_stats.get("streak_win"), False, False),
        ("Longest loss streak", old_stats.get("streak_loss"),new_stats.get("streak_loss"),False, False),
        ("Final balance",       old_stats.get("final_balance"),new_stats.get("final_balance"),False,True),
    ]

    for label, ov, nv, pct, dol in rows:
        # Highlight improvements
        better = ""
        if ov is not None and nv is not None and isinstance(ov, (int,float)) and isinstance(nv,(int,float)):
            better_new = (
                (label in ("Win rate","Total P&L","Return %","Profit factor","Avg win","Best trade","Final balance","Sharpe ratio","Longest win streak") and nv > ov)
                or (label in ("Avg loss","Worst trade","Max drawdown","Longest loss streak") and nv > ov and label == "Avg loss")
                or (label in ("Max drawdown","Longest loss streak") and nv < ov)
            )
            better = "  ✅" if better_new else ""

        print(f"  {label:<26}  {fmt(ov,pct,dol):>14}  {fmt(nv,pct,dol):>14}{better}")

    print(SEP)

    # Strategy breakdown
    if "strat_breakdown" in new_stats and new_stats["strat_breakdown"].get("count"):
        print(f"\n  Strategy breakdown (NEW):")
        counts = new_stats["strat_breakdown"]["count"]
        pnls_s = new_stats["strat_breakdown"]["sum"]
        means  = new_stats["strat_breakdown"]["mean"]
        for s in sorted(counts):
            print(f"    {s:<18}  {int(counts.get(s,0)):>4} trades  "
                  f"P&L ${pnls_s.get(s,0):>8,.2f}  avg ${means.get(s,0):>7.2f}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  BACKTEST — Trailing TP/SL vs Fixed TP/SL           ║")
    print(f"║  Period: {YEARS_BACK} years  |  TF: {CRYPTO_TF}  |  HTF: {CRYPTO_HTF}           ║")
    print(f"║  Symbols: {', '.join(CRYPTO_SYMBOLS):<42}║")
    print("╚══════════════════════════════════════════════════════╝\n")

    all_old_trades = []
    all_new_trades = []
    final_bals = {}

    for symbol in CRYPTO_SYMBOLS:
        print(f"\n{'─'*50}")
        print(f"  📊  {symbol}")
        print(f"{'─'*50}")

        # Fetch data
        df15 = fetch_ohlcv_full(symbol, CRYPTO_TF,   YEARS_BACK)
        df4h = fetch_ohlcv_full(symbol, CRYPTO_HTF,  YEARS_BACK)

        if len(df15) < 500:
            print(f"  ⚠️  Not enough data for {symbol} — skipping")
            continue

        t_old, t_new, bal_old, bal_new = backtest_symbol(symbol, df15, df4h)
        all_old_trades.extend(t_old)
        all_new_trades.extend(t_new)
        final_bals[symbol] = (bal_old, bal_new)

        s_old = calc_stats(t_old, bal_old, "OLD")
        s_new = calc_stats(t_new, bal_new, "NEW")
        print_comparison(s_old, s_new, symbol)

    # ── Combined summary ──────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  COMBINED SUMMARY  —  All symbols")
    print(f"{'═'*68}")
    total_bal_old = sum(v[0] for v in final_bals.values())
    total_bal_new = sum(v[1] for v in final_bals.values())
    total_start   = STARTING_BALANCE * len(final_bals)
    print(f"  Starting capital:   ${total_start:>12,.2f}")
    print(f"  OLD final balance:  ${total_bal_old:>12,.2f}  ({(total_bal_old-total_start)/total_start*100:+.1f}%)")
    print(f"  NEW final balance:  ${total_bal_new:>12,.2f}  ({(total_bal_new-total_start)/total_start*100:+.1f}%)")
    extra = total_bal_new - total_bal_old
    print(f"  Extra profit (NEW): ${extra:>12,.2f}  ({extra/total_start*100:+.1f}% vs OLD)")
    print(f"  OLD trades:  {len(all_old_trades):>6}    NEW trades:  {len(all_new_trades):>6}")

    # ── Save to CSV ───────────────────────────────────────────────
    out_dir  = os.path.dirname(__file__)
    csv_old  = os.path.join(out_dir, "backtest_old_trades.csv")
    csv_new  = os.path.join(out_dir, "backtest_new_trades.csv")
    if all_old_trades:
        pd.DataFrame(all_old_trades).to_csv(csv_old, index=False)
        print(f"\n  💾  OLD trades saved → {csv_old}")
    if all_new_trades:
        pd.DataFrame(all_new_trades).to_csv(csv_new, index=False)
        print(f"  💾  NEW trades saved → {csv_new}")

    print("\n  ✅  Backtest complete.\n")


if __name__ == "__main__":
    main()
