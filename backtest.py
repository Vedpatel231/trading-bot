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
        Returns list of (exit_price, reason, qty_fraction) events, or [] if still open.
        Consistent with NewExitSim interface so the backtest loop works identically.
        Uses conservative intra-candle ordering: stop checked before TP.
        """
        if self.direction == "long":
            # Breakeven lock
            if not self.be_locked and h >= self.entry + self.atr:
                if self.stop < self.entry:
                    self.stop = self.entry
                    self.be_locked = True
            if l <= self.stop:
                return [(max(self.stop, l), "stop", 1.0)]
            if h >= self.tp:
                return [(self.tp, "tp", 1.0)]
        else:
            if not self.be_locked and l <= self.entry - self.atr:
                if self.stop > self.entry:
                    self.stop = self.entry
                    self.be_locked = True
            if h >= self.stop:
                return [(min(self.stop, h), "stop", 1.0)]
            if l <= self.tp:
                return [(self.tp, "tp", 1.0)]
        return []

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

    def _can_trade(cooldown_ts, current_ts):
        if cooldown_ts is None: return True
        return (current_ts - cooldown_ts).total_seconds() / 60 >= COOLDOWN_MIN

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
        # Bug-fix: after reindex+ffill the first few rows may be NaN;
        # bool(NaN) raises ValueError, so we cast via fillna first.
        try:
            reg_row = regime_reindexed.iloc[i] if i < len(regime_reindexed) else None
            if reg_row is None or reg_row.isna().all():
                r_up = False; r_down = False; r_macro = True
            else:
                r_up    = bool(reg_row.get("regime_up",  False) == True)
                r_down  = bool(reg_row.get("regime_down", False) == True)
                r_macro = bool(reg_row.get("macro_bull",  True)  == True)
        except Exception:
            r_up = False; r_down = False; r_macro = True

        # ── OLD strategy ─────────────────────────────────────────────
        if in_old:
            events = sim_old.process_candle(o, h, l, c)
            # OldExitSim never returns a "partial" event — just take the first exit
            final_exit = next((e for e in events if e[1] != "partial"), None)
            if final_exit:
                xprice, xreason, _ = final_exit
                pnl = sim_old.calc_pnl(xprice)
                bal_old += pnl
                trades_old.append({
                    "symbol": symbol, "entry_ts": entry_bar_old,
                    "exit_ts": ts, "direction": sim_old.direction,
                    "strategy": getattr(sim_old, "strategy", ""),
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
            # ── Bug-fix: never double-count partial P&L ─────────────
            # sim_new.partial_pnl is set inside NewExitSim.process_candle()
            # the moment the partial fires (regardless of which candle).
            # We only touch bal_new once, at full close.
            full_exit = next((e for e in events if e[1] != "partial"), None)

            if full_exit:
                xprice, xreason, _ = full_exit
                remainder_pnl = sim_new.total_pnl(xprice)
                # partial_pnl is 0 if no partial fired, or the locked-in partial amount
                total_pnl = sim_new.partial_pnl + remainder_pnl
                bal_new += total_pnl
                trades_new.append({
                    "symbol": symbol, "entry_ts": entry_bar_new,
                    "exit_ts": ts, "direction": sim_new.direction,
                    "strategy": getattr(sim_new, "strategy", ""),
                    "entry": sim_new.entry, "exit": xprice, "atr": atr,
                    "pnl": round(total_pnl, 4), "reason": xreason,
                    "half_sold": sim_new.half_sold,
                    "partial_pnl": round(sim_new.partial_pnl, 4),
                    "balance": round(bal_new, 2),
                })
                in_new = False
                if total_pnl < 0:
                    cooldown_new = ts
            # (no elif branch needed — partial P&L is held in sim_new.partial_pnl
            #  until the full close, keeping the balance curve clean)

        # ── Entry signal (skipped if already in a trade for that sim) ────────
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
def calc_stats(trades: list, label: str, direction_filter: str = "all") -> dict:
    """
    Compute stats for a list of trade dicts.
    direction_filter: "all" | "long" | "short"
    """
    if not trades:
        return {"label": label, "direction": direction_filter, "trades": 0}

    df = pd.DataFrame(trades)
    if direction_filter != "all":
        df = df[df["direction"] == direction_filter]

    if df.empty:
        return {"label": label, "direction": direction_filter, "trades": 0}

    pnls = df["pnl"].tolist()
    wins = [p for p in pnls if p >= 0]
    loss = [p for p in pnls if p < 0]

    # Running drawdown (uses sorted exit_ts to respect time ordering)
    df_sorted = df.sort_values("exit_ts")
    pnl_cum   = df_sorted["pnl"].cumsum()
    peak      = pnl_cum.cummax()
    dd_series = (peak - pnl_cum) / (STARTING_BALANCE + peak)
    max_dd    = float(dd_series.max()) if not dd_series.empty else 0.0

    # Sharpe (annualised from daily P&L)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"])
    daily  = df.groupby(df["exit_ts"].dt.date)["pnl"].sum()
    sharpe = (daily.mean() / daily.std() * (252 ** 0.5)) if len(daily) > 1 and daily.std() > 0 else 0.0

    # Win / loss streaks
    streak_w = streak_l = cur = 0
    for p in pnls:
        if p >= 0:
            cur = cur + 1 if cur >= 0 else 1
            streak_w = max(streak_w, cur)
        else:
            cur = cur - 1 if cur <= 0 else -1
            streak_l = max(streak_l, -cur)

    # Strategy breakdown (count, total_pnl, avg_pnl per strategy)
    strat_grp = df.groupby("strategy")["pnl"].agg(["count", "sum", "mean"])

    # Partial-close rate (NEW only field)
    half_sold_rate = float(df["half_sold"].mean() * 100) if "half_sold" in df.columns else 0.0

    return {
        "label":           label,
        "direction":       direction_filter,
        "trades":          len(pnls),
        "win_rate":        round(len(wins) / len(pnls) * 100, 1),
        "total_pnl":       round(sum(pnls), 2),
        "return_pct":      round(sum(pnls) / STARTING_BALANCE * 100, 2),
        "avg_win":         round(np.mean(wins), 2) if wins else 0.0,
        "avg_loss":        round(np.mean(loss), 2) if loss else 0.0,
        "profit_factor":   round(sum(wins) / abs(sum(loss)), 2) if loss else float("inf"),
        "max_drawdown":    round(max_dd * 100, 2),
        "sharpe":          round(sharpe, 2),
        "best_trade":      round(max(pnls), 2),
        "worst_trade":     round(min(pnls), 2),
        "streak_win":      streak_w,
        "streak_loss":     streak_l,
        "half_sold_rate":  round(half_sold_rate, 1),
        "strat_breakdown": strat_grp.to_dict(),
    }


# ── Formatting helpers ─────────────────────────────────────────────────────
def _fv(v, pct=False, dol=False):
    """Format a stat value for table display."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "     N/A"
    if dol: return f"${v:>9,.0f}"
    if pct: return f"{v:>8.1f}%"
    if isinstance(v, float): return f"{v:>9.2f}"
    return f"{v:>9}"

def _better(label, ov, nv) -> str:
    """Return ✅ if new is better, ❌ if worse, blank if equal/N/A."""
    if ov is None or nv is None: return ""
    if not (isinstance(ov, (int, float)) and isinstance(nv, (int, float))): return ""
    higher_is_better = label in {
        "Win rate", "Total P&L", "Return %", "Profit factor",
        "Avg win", "Best trade", "Sharpe ratio",
        "Win streak", "Partial close %",
    }
    lower_is_better = label in {"Max drawdown", "Loss streak"}
    # Avg loss and Worst trade: less negative = better → higher is better
    if label in ("Avg loss", "Worst trade"):
        higher_is_better = True
    if higher_is_better:
        return " ✅" if nv > ov else (" ❌" if nv < ov else "")
    if lower_is_better:
        return " ✅" if nv < ov else (" ❌" if nv > ov else "")
    return ""


def print_direction_block(title: str, s_old: dict, s_new: dict):
    """Print a single OLD vs NEW comparison block for one direction."""
    W = 72
    print(f"\n  ┌{'─'*(W-2)}┐")
    print(f"  │  {title:<{W-5}}│")
    print(f"  ├{'─'*22}┬{'─'*22}┬{'─'*22}┤")
    print(f"  │  {'Metric':<20}│  {'OLD (fixed)':^19}│  {'NEW (trailing)':^19}│")
    print(f"  ├{'─'*22}┼{'─'*22}┼{'─'*22}┤")

    rows = [
        ("Trades",         s_old.get("trades"),          s_new.get("trades"),          False, False),
        ("Win rate",       s_old.get("win_rate"),         s_new.get("win_rate"),         True,  False),
        ("Total P&L",      s_old.get("total_pnl"),        s_new.get("total_pnl"),        False, True),
        ("Return %",       s_old.get("return_pct"),       s_new.get("return_pct"),       True,  False),
        ("Profit factor",  s_old.get("profit_factor"),    s_new.get("profit_factor"),    False, False),
        ("Avg win",        s_old.get("avg_win"),          s_new.get("avg_win"),          False, True),
        ("Avg loss",       s_old.get("avg_loss"),         s_new.get("avg_loss"),         False, True),
        ("Best trade",     s_old.get("best_trade"),       s_new.get("best_trade"),       False, True),
        ("Worst trade",    s_old.get("worst_trade"),      s_new.get("worst_trade"),      False, True),
        ("Max drawdown",   s_old.get("max_drawdown"),     s_new.get("max_drawdown"),     True,  False),
        ("Sharpe ratio",   s_old.get("sharpe"),           s_new.get("sharpe"),           False, False),
        ("Win streak",     s_old.get("streak_win"),       s_new.get("streak_win"),       False, False),
        ("Loss streak",    s_old.get("streak_loss"),      s_new.get("streak_loss"),      False, False),
        ("Partial close %","—",                           s_new.get("half_sold_rate"),   True,  False),
    ]

    for label, ov, nv, pct, dol in rows:
        badge = _better(label, ov if ov != "—" else None, nv)
        ov_s  = "        —" if ov == "—" else _fv(ov, pct, dol)
        nv_s  = _fv(nv, pct, dol) + badge
        print(f"  │  {label:<20}│  {ov_s:^19}│  {nv_s:<19}│")

    print(f"  └{'─'*22}┴{'─'*22}┴{'─'*22}┘")


def print_strategy_breakdown(trades_old: list, trades_new: list, direction_filter: str):
    """Per-strategy breakdown for OLD and NEW, given direction."""
    df_o = pd.DataFrame(trades_old) if trades_old else pd.DataFrame()
    df_n = pd.DataFrame(trades_new) if trades_new else pd.DataFrame()

    if not df_o.empty and direction_filter != "all":
        df_o = df_o[df_o["direction"] == direction_filter]
    if not df_n.empty and direction_filter != "all":
        df_n = df_n[df_n["direction"] == direction_filter]

    strategies = set()
    if not df_o.empty and "strategy" in df_o.columns:
        strategies |= set(df_o["strategy"].unique())
    if not df_n.empty and "strategy" in df_n.columns:
        strategies |= set(df_n["strategy"].unique())

    if not strategies:
        return

    print(f"\n  Strategy breakdown  ({direction_filter.upper()}):")
    print(f"  {'Strategy':<18}  {'OLD trades':>10}  {'OLD P&L':>10}  {'NEW trades':>10}  {'NEW P&L':>10}  {'Δ P&L':>10}")
    print(f"  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for strat in sorted(strategies):
        grp_o = df_o[df_o["strategy"] == strat]["pnl"] if not df_o.empty else pd.Series(dtype=float)
        grp_n = df_n[df_n["strategy"] == strat]["pnl"] if not df_n.empty else pd.Series(dtype=float)
        ct_o  = len(grp_o);  ct_n = len(grp_n)
        pl_o  = grp_o.sum(); pl_n = grp_n.sum()
        delta = pl_n - pl_o
        badge = " ✅" if delta > 0 else (" ❌" if delta < 0 else "")
        print(f"  {strat:<18}  {ct_o:>10}  ${pl_o:>9,.0f}  {ct_n:>10}  ${pl_n:>9,.0f}  ${delta:>+9,.0f}{badge}")


def print_symbol_report(symbol: str, trades_old: list, trades_new: list):
    """Full OLD vs NEW report for one symbol — Longs, Shorts, Combined."""
    W = 72
    print(f"\n{'═'*W}")
    print(f"  📊  {symbol}  —  OLD fixed TP/SL  vs  NEW trailing TP/SL")
    print(f"{'═'*W}")

    for direction, label in [("long", "🟢 LONGS"), ("short", "🔴 SHORTS"), ("all", "⚪ COMBINED")]:
        s_old = calc_stats(trades_old, "OLD", direction)
        s_new = calc_stats(trades_new, "NEW", direction)

        if s_old.get("trades", 0) == 0 and s_new.get("trades", 0) == 0:
            print(f"\n  {label}: no trades found")
            continue

        print_direction_block(label, s_old, s_new)
        print_strategy_breakdown(trades_old, trades_new, direction)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    W = 72
    print(f"╔{'═'*(W-2)}╗")
    print(f"║  BACKTEST — Trailing TP/SL vs Fixed TP/SL  (Longs + Shorts){' '*(W-62)}║")
    print(f"║  Period: {YEARS_BACK}y  |  TF: {CRYPTO_TF}  |  HTF: {CRYPTO_HTF}  |  Starting: ${STARTING_BALANCE:,.0f}/symbol{' '*(W-66)}║")
    print(f"║  Symbols: {', '.join(CRYPTO_SYMBOLS):<{W-12}}║")
    print(f"╚{'═'*(W-2)}╝\n")

    all_old_trades: list = []
    all_new_trades: list = []
    symbol_results: dict = {}   # symbol → (t_old, t_new, bal_old, bal_new)

    for symbol in CRYPTO_SYMBOLS:
        print(f"\n{'─'*W}")
        print(f"  Fetching data for {symbol} …")
        print(f"{'─'*W}")

        df15 = fetch_ohlcv_full(symbol, CRYPTO_TF,  YEARS_BACK)
        df4h = fetch_ohlcv_full(symbol, CRYPTO_HTF, YEARS_BACK)

        if len(df15) < 500:
            print(f"  ⚠️  Not enough data for {symbol} — skipping")
            continue

        t_old, t_new, bal_old, bal_new = backtest_symbol(symbol, df15, df4h)
        all_old_trades.extend(t_old)
        all_new_trades.extend(t_new)
        symbol_results[symbol] = (t_old, t_new, bal_old, bal_new)

        # Per-symbol report (Longs / Shorts / Combined)
        print_symbol_report(symbol, t_old, t_new)

    # ── All-symbols combined report ───────────────────────────────
    if symbol_results:
        print(f"\n{'═'*W}")
        print(f"  🌐  ALL SYMBOLS COMBINED  —  Longs + Shorts + Overall")
        print(f"{'═'*W}")
        print_symbol_report("BTC + ETH + SOL", all_old_trades, all_new_trades)

    # ── High-level P&L summary ────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  💰  FINAL BALANCE SUMMARY")
    print(f"{'═'*W}")
    print(f"  {'Symbol':<12}  {'Start':>10}  {'OLD final':>12}  {'OLD ret':>8}  {'NEW final':>12}  {'NEW ret':>8}  {'Extra $':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*10}")

    total_start = total_old = total_new = 0.0
    for sym, (_, _, b_old, b_new) in symbol_results.items():
        start = STARTING_BALANCE
        ret_o = (b_old - start) / start * 100
        ret_n = (b_new - start) / start * 100
        extra = b_new - b_old
        badge = " ✅" if extra > 0 else " ❌"
        print(f"  {sym:<12}  ${start:>9,.0f}  ${b_old:>11,.0f}  {ret_o:>+7.1f}%  ${b_new:>11,.0f}  {ret_n:>+7.1f}%  ${extra:>+9,.0f}{badge}")
        total_start += start; total_old += b_old; total_new += b_new

    if symbol_results:
        total_extra = total_new - total_old
        ret_o_t = (total_old - total_start) / total_start * 100
        ret_n_t = (total_new - total_start) / total_start * 100
        print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*10}")
        print(f"  {'TOTAL':<12}  ${total_start:>9,.0f}  ${total_old:>11,.0f}  {ret_o_t:>+7.1f}%  ${total_new:>11,.0f}  {ret_n_t:>+7.1f}%  ${total_extra:>+9,.0f}")

    print(f"\n  OLD trades total: {len(all_old_trades):>5}   "
          f"({sum(1 for t in all_old_trades if t['direction']=='long')} long / "
          f"{sum(1 for t in all_old_trades if t['direction']=='short')} short)")
    print(f"  NEW trades total: {len(all_new_trades):>5}   "
          f"({sum(1 for t in all_new_trades if t['direction']=='long')} long / "
          f"{sum(1 for t in all_new_trades if t['direction']=='short')} short)")

    # ── Save CSVs ─────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for label, trades in [("old", all_old_trades), ("new", all_new_trades)]:
        if not trades:
            continue
        df_out = pd.DataFrame(trades)
        # Add a human-readable hold_time column
        df_out["entry_ts"] = pd.to_datetime(df_out["entry_ts"])
        df_out["exit_ts"]  = pd.to_datetime(df_out["exit_ts"])
        df_out["hold_hours"] = (df_out["exit_ts"] - df_out["entry_ts"]).dt.total_seconds() / 3600
        path = os.path.join(out_dir, f"backtest_{label}_trades.csv")
        df_out.to_csv(path, index=False)
        print(f"\n  💾  {label.upper()} trades ({len(trades)}) → {path}")

    print(f"\n{'═'*W}")
    print("  ✅  Backtest complete.")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
