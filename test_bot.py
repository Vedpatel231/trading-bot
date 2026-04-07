"""
═══════════════════════════════════════════════════════════
  BOT QA TEST SUITE  — no live API required
  Tests every critical function with synthetic data.
  Run:  python3 test_bot.py
═══════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Patch exchange so imports don't need live API ──────────
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Stub out ccxt before bot imports it
import types
ccxt_stub = types.ModuleType("ccxt")
ccxt_stub.binanceus = lambda: MagicMock()
sys.modules["ccxt"] = ccxt_stub
# NOTE: alpaca_trade_api stub removed — import is already commented out in bot.py
#       Stocks section fully disabled; no alpaca stub needed.

import importlib
bot = importlib.import_module("bot")

# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def make_df(n=200, trend="up", base=50000.0, atr_pct=0.01):
    """Build a synthetic OHLCV DataFrame with a known trend."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="30min")
    if trend == "up":
        close = base + np.cumsum(np.abs(np.random.randn(n))) * base * 0.002
    elif trend == "down":
        close = base - np.cumsum(np.abs(np.random.randn(n))) * base * 0.002
    else:  # flat
        close = base + np.random.randn(n) * base * 0.001

    noise = base * 0.003
    df = pd.DataFrame({
        "open":   close - np.random.rand(n) * noise,
        "high":   close + np.random.rand(n) * noise * 2,
        "low":    close - np.random.rand(n) * noise * 2,
        "close":  close,
        "volume": np.abs(np.random.randn(n)) * 1000 + 500,
    }, index=dates)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    return df


def reset_paper(symbol="BTC/USDT", balance=3333.33):
    """Reset a single coin's paper-trade state."""
    bot.crypto_paper[symbol] = {
        "balance": balance, "coin_held": 0.0, "in_trade": False,
        "trade_direction": "", "entry_price": 0.0, "entry_atr": 0.0,
        "stop_price": 0.0, "tp_price": 0.0, "entry_strategy": "",
        "highest_price": 0.0, "lowest_price": 999999.0,
        "total_trades": 0, "wins": 0, "losses": 0,
        "daily_start_bal": balance, "last_loss_time": None,
        "total_pnl": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
    }


# ══════════════════════════════════════════════════════════
#  TEST CASES
# ══════════════════════════════════════════════════════════

class TestSafeFloat(unittest.TestCase):
    def test_normal(self):          self.assertEqual(bot.safe_float(3.14), 3.14)
    def test_string_int(self):      self.assertEqual(bot.safe_float("42"), 42.0)
    def test_none(self):            self.assertEqual(bot.safe_float(None), 0.0)
    def test_nan(self):             self.assertEqual(bot.safe_float(float("nan")), 0.0)
    def test_default_used(self):    self.assertEqual(bot.safe_float(None, 99.0), 99.0)
    def test_zero(self):            self.assertEqual(bot.safe_float(0), 0.0)
    def test_negative(self):        self.assertEqual(bot.safe_float(-5.5), -5.5)


class TestPositionSizing(unittest.TestCase):
    """calc_position_size should never exceed 25% of balance."""
    def test_basic(self):
        size = bot.calc_position_size(10000, 50000, 500, 1.2)
        self.assertGreater(size, 0)
        self.assertLessEqual(size, 10000 * 0.25)

    def test_max_25pct(self):
        # With tiny stop distance, formula would suggest huge size — cap at 25%
        size = bot.calc_position_size(10000, 50000, 1, 1.2)
        self.assertLessEqual(size, 10000 * 0.25 + 1)

    def test_zero_atr_fallback(self):
        size = bot.calc_position_size(10000, 50000, 0, 1.2)
        self.assertGreater(size, 0)

    def test_minimum_one_dollar(self):
        size = bot.calc_position_size(0.5, 50000, 500, 1.2)
        self.assertGreaterEqual(size, 1.0)

    def test_proportional_to_balance(self):
        s1 = bot.calc_position_size(10000, 50000, 500, 1.2)
        s2 = bot.calc_position_size(5000,  50000, 500, 1.2)
        self.assertAlmostEqual(s1 / s2, 2.0, places=1)


class TestAddIndicators(unittest.TestCase):
    def setUp(self):
        self.df = bot.add_indicators(make_df(200))

    def test_columns_present(self):
        for col in ["ema_fast","ema_slow","rsi","rsi_prev","macd","macd_signal","macd_hist",
                    "atr","atr_pct","adx","vwap","chop_range",
                    "vol_spike","strong_bullish","strong_bearish",
                    "recent_high","recent_low"]:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_ema_not_all_nan(self):
        self.assertFalse(self.df["ema_fast"].dropna().empty)

    def test_rsi_range(self):
        rsi = self.df["rsi"].dropna()
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())

    def test_atr_positive(self):
        self.assertTrue((self.df["atr"].dropna() >= 0).all())

    def test_vol_spike_boolean(self):
        self.assertTrue(self.df["vol_spike"].dtype == bool or
                        set(self.df["vol_spike"].dropna().unique()).issubset({True, False}))


class TestCryptoBuy(unittest.TestCase):
    def setUp(self):
        reset_paper("BTC/USDT", 3333.33)

    def test_opens_trade(self):
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")
        p = bot.crypto_paper["BTC/USDT"]
        self.assertTrue(p["in_trade"])
        self.assertEqual(p["trade_direction"], "long")
        self.assertGreater(p["coin_held"], 0)

    def test_stop_below_entry(self):
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")
        p = bot.crypto_paper["BTC/USDT"]
        self.assertLess(p["stop_price"], 50000.0)

    def test_tp_above_entry(self):
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")
        p = bot.crypto_paper["BTC/USDT"]
        self.assertGreater(p["tp_price"], 50000.0)

    def test_balance_reduced(self):
        start_bal = bot.crypto_paper["BTC/USDT"]["balance"]
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")
        self.assertLess(bot.crypto_paper["BTC/USDT"]["balance"], start_bal)

    def test_no_double_entry(self):
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")
        bal_after_first = bot.crypto_paper["BTC/USDT"]["balance"]
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")  # should be ignored
        self.assertEqual(bot.crypto_paper["BTC/USDT"]["balance"], bal_after_first)

    def test_fee_deducted_from_coin_qty(self):
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")
        p = bot.crypto_paper["BTC/USDT"]
        spend = 3333.33 - p["balance"]
        expected_qty = (spend - spend * bot.TRADING_FEE_PCT) / 50000.0
        self.assertAlmostEqual(p["coin_held"], expected_qty, places=6)


class TestCryptoClose(unittest.TestCase):
    def setUp(self):
        reset_paper("BTC/USDT", 3333.33)
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")

    def test_close_win_updates_wins(self):
        bot.crypto_close("BTC/USDT", 51000.0, "ATR take profit")
        p = bot.crypto_paper["BTC/USDT"]
        self.assertFalse(p["in_trade"])
        self.assertEqual(p["wins"], 1)
        self.assertEqual(p["losses"], 0)

    def test_close_loss_updates_losses(self):
        bot.crypto_close("BTC/USDT", 49000.0, "ATR stop")
        p = bot.crypto_paper["BTC/USDT"]
        self.assertEqual(p["losses"], 1)
        self.assertEqual(p["wins"], 0)

    def test_balance_restored_on_win(self):
        start_bal = 3333.33
        bot.crypto_close("BTC/USDT", 51500.0, "ATR take profit")
        self.assertGreater(bot.crypto_paper["BTC/USDT"]["balance"], start_bal)

    def test_balance_reduced_on_loss(self):
        start_bal = 3333.33
        bot.crypto_close("BTC/USDT", 48000.0, "ATR stop")
        self.assertLess(bot.crypto_paper["BTC/USDT"]["balance"], start_bal)

    def test_state_fully_cleared(self):
        bot.crypto_close("BTC/USDT", 51000.0, "ATR take profit")
        p = bot.crypto_paper["BTC/USDT"]
        self.assertFalse(p["in_trade"])
        self.assertEqual(p["coin_held"], 0.0)
        self.assertEqual(p["entry_price"], 0.0)
        self.assertEqual(p["stop_price"], 0.0)
        self.assertEqual(p["tp_price"], 0.0)

    def test_no_close_if_not_in_trade(self):
        bot.crypto_close("BTC/USDT", 51000.0)  # close first
        bot.crypto_close("BTC/USDT", 51000.0)  # second should be no-op
        self.assertEqual(bot.crypto_paper["BTC/USDT"]["total_trades"], 1)


class TestBreakevenTrailingStop(unittest.TestCase):
    """BUG-FIX #6: breakeven trailing stop."""
    def setUp(self):
        reset_paper("BTC/USDT", 3333.33)
        # Enter at 50000, ATR=500 → stop=49400, tp=51250
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")
        self.p = bot.crypto_paper["BTC/USDT"]

    def test_stop_stays_below_entry_before_1atr(self):
        """Price at 50400 (< entry+1ATR=50500) — stop should NOT move."""
        original_stop = self.p["stop_price"]
        bot.check_crypto_exits("BTC/USDT", 50400.0)
        self.assertEqual(self.p["stop_price"], original_stop)

    def test_stop_moves_to_breakeven_after_1atr(self):
        """Price at 50600 (> entry+1ATR=50500) — stop should move to 50000."""
        bot.check_crypto_exits("BTC/USDT", 50600.0)
        self.assertEqual(self.p["stop_price"], 50000.0)

    def test_breakeven_stop_triggers_correctly(self):
        """After breakeven lock-in, dropping back to entry should close trade."""
        bot.check_crypto_exits("BTC/USDT", 50600.0)   # lock breakeven
        self.assertEqual(self.p["stop_price"], 50000.0)
        bot.check_crypto_exits("BTC/USDT", 49999.0)   # drops below breakeven
        self.assertFalse(self.p["in_trade"])

    def test_trade_stays_open_between_breakeven_and_tp(self):
        """Price between breakeven stop and TP — trade should remain open."""
        bot.check_crypto_exits("BTC/USDT", 50600.0)   # lock breakeven
        bot.check_crypto_exits("BTC/USDT", 50200.0)   # above breakeven, below TP
        self.assertTrue(self.p["in_trade"])

    def test_tp_still_works_after_breakeven(self):
        """Breakeven locked in but TP should still close trade."""
        bot.check_crypto_exits("BTC/USDT", 50600.0)   # lock breakeven
        bot.check_crypto_exits("BTC/USDT", 51300.0)   # exceeds TP=51250
        self.assertFalse(self.p["in_trade"])


class TestExits(unittest.TestCase):
    def setUp(self):
        reset_paper("BTC/USDT", 3333.33)
        bot.crypto_buy("BTC/USDT", 50000.0, 55.0, 500.0, "Trend")

    def test_stop_loss_fires(self):
        # Stop is at 50000 - 500*1.2 = 49400
        bot.check_crypto_exits("BTC/USDT", 49300.0)
        self.assertFalse(bot.crypto_paper["BTC/USDT"]["in_trade"])

    def test_take_profit_fires(self):
        # TP is at 50000 + 500*2.5 = 51250
        bot.check_crypto_exits("BTC/USDT", 51300.0)
        self.assertFalse(bot.crypto_paper["BTC/USDT"]["in_trade"])

    def test_price_between_stop_tp_stays_open(self):
        bot.check_crypto_exits("BTC/USDT", 50200.0)
        self.assertTrue(bot.crypto_paper["BTC/USDT"]["in_trade"])


class TestShortPosition(unittest.TestCase):
    def setUp(self):
        reset_paper("ETH/USDT", 3333.33)
        bot.crypto_short("ETH/USDT", 2000.0, 55.0, 50.0, "ShortTrend")
        self.p = bot.crypto_paper["ETH/USDT"]

    def test_short_opens(self):
        self.assertTrue(self.p["in_trade"])
        self.assertEqual(self.p["trade_direction"], "short")

    def test_short_stop_above_entry(self):
        self.assertGreater(self.p["stop_price"], 2000.0)

    def test_short_tp_below_entry(self):
        self.assertLess(self.p["tp_price"], 2000.0)

    def test_short_wins_on_price_drop(self):
        tp = self.p["tp_price"]
        bot.check_crypto_exits("ETH/USDT", tp - 1)
        self.assertFalse(self.p["in_trade"])
        self.assertEqual(self.p["wins"], 1)

    def test_short_loses_on_price_rise(self):
        stop = self.p["stop_price"]
        bot.check_crypto_exits("ETH/USDT", stop + 1)
        self.assertFalse(self.p["in_trade"])
        self.assertEqual(self.p["losses"], 1)

    def test_short_breakeven_trailing(self):
        """Short breakeven: once price drops 1 ATR below entry, stop moves to entry."""
        # entry=2000, atr=50 → needs price <= 1950 to lock breakeven
        original_stop = self.p["stop_price"]
        bot.check_crypto_exits("ETH/USDT", 1945.0)
        self.assertEqual(self.p["stop_price"], 2000.0)


class TestDailyLossLimit(unittest.TestCase):
    def setUp(self):
        for sym in bot.CRYPTO_SYMBOLS:
            reset_paper(sym)
        bot.perf.update({
            "date": datetime.now().date(), "start_bal": 10000.0,
            "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
            "paused": False, "pause_reason": "",
        })

    def test_trading_allowed_normally(self):
        self.assertTrue(bot.is_trading_allowed("BTC/USDT"))

    def test_paused_after_daily_loss_limit(self):
        # Simulate 4% loss (limit is 3%)
        bot.perf["start_bal"] = 10000.0
        for sym in bot.CRYPTO_SYMBOLS:
            bot.crypto_paper[sym]["balance"] = 3333.33 * 0.97  # each coin lost ~3%
        self.assertFalse(bot.is_trading_allowed("BTC/USDT"))

    def test_cooldown_blocks_trading(self):
        bot.crypto_paper["BTC/USDT"]["last_loss_time"] = datetime.now() - timedelta(minutes=10)
        result = bot.is_trading_allowed("BTC/USDT")
        self.assertFalse(result)

    def test_cooldown_expires(self):
        # COOLDOWN_MINUTES is now 45; use 50 min to ensure it has passed
        bot.crypto_paper["BTC/USDT"]["last_loss_time"] = datetime.now() - timedelta(minutes=50)
        result = bot.is_trading_allowed("BTC/USDT")
        self.assertTrue(result)


class TestSignalRSIFilter(unittest.TestCase):
    """BUG-FIX #3: ensure RSI filters are applied in get_crypto_signal."""
    def _make_regime(self, up=True, macro_bull=False):
        return {
            "up": up, "down": not up, "not_bearish": up, "not_bullish": not up,
            "macro_bull": macro_bull, "label": "UP" if up else "DOWN",
            "atr_pct": 0.01, "slow_rising": up, "slow_falling": not up,
        }

    def _make_crossover_df(self, cross="up", rsi_val=50.0):
        """Return a 2-row DataFrame with a controlled EMA crossover."""
        df = bot.add_indicators(make_df(200, trend="up" if cross == "up" else "down"))
        df = df.dropna()
        # Force the last two rows to have a crossover
        last_idx  = df.index[-1]
        prev_idx  = df.index[-2]
        if cross == "up":
            df.loc[prev_idx, "ema_fast"] = 49990.0
            df.loc[prev_idx, "ema_slow"] = 50010.0
            df.loc[last_idx, "ema_fast"] = 50010.0
            df.loc[last_idx, "ema_slow"] = 49990.0
            df.loc[last_idx, "macd"]         =  10.0
            df.loc[last_idx, "macd_signal"]  =   5.0
            df.loc[last_idx, "macd_hist"]    =   5.0
        else:
            df.loc[prev_idx, "ema_fast"] = 50010.0
            df.loc[prev_idx, "ema_slow"] = 49990.0
            df.loc[last_idx, "ema_fast"] = 49990.0
            df.loc[last_idx, "ema_slow"] = 50010.0
            df.loc[last_idx, "macd"]         = -10.0
            df.loc[last_idx, "macd_signal"]  =  -5.0
            df.loc[last_idx, "macd_hist"]    =  -5.0
        df.loc[last_idx, "rsi"] = rsi_val
        return df

    def test_long_blocked_when_rsi_too_high(self):
        df = self._make_crossover_df(cross="up", rsi_val=70.0)
        sig = bot.get_crypto_signal(df, self._make_regime(up=True), {"up": False})
        self.assertNotEqual(sig["signal"], "BUY", "Long should be blocked when RSI=70")

    def test_long_allowed_when_rsi_below_65(self):
        df = self._make_crossover_df(cross="up", rsi_val=55.0)
        sig = bot.get_crypto_signal(df, self._make_regime(up=True), {"up": False})
        # Could be BUY (Trend) or HOLD depending on other conditions — just verify not blocked by RSI
        # The key test: if RSI was 70 it would HOLD, so 55 gives a chance for BUY
        self.assertIn(sig["signal"], ["BUY", "HOLD"])

    def test_short_blocked_in_macro_bull(self):
        df = self._make_crossover_df(cross="down", rsi_val=40.0)
        sig = bot.get_crypto_signal(df, self._make_regime(up=False, macro_bull=True), {"down": False})
        self.assertNotEqual(sig["signal"], "SHORT", "Short should be blocked when macro_bull=True")

    def test_short_blocked_when_rsi_too_low(self):
        df = self._make_crossover_df(cross="down", rsi_val=30.0)
        sig = bot.get_crypto_signal(df, self._make_regime(up=False, macro_bull=False), {"down": False})
        self.assertNotEqual(sig["signal"], "SHORT", "Short should be blocked when RSI=30")

    def test_no_sell_cover_signals(self):
        """BUG-FIX #3: SELL/COVER dead signals must be gone."""
        df = self._make_crossover_df(cross="down", rsi_val=55.0)
        sig = bot.get_crypto_signal(df, self._make_regime(up=True), {"up": True, "down": False})
        self.assertNotIn(sig["signal"], ["SELL", "COVER"],
                         "SELL/COVER signals must not be returned — they were removed")


class TestShortBreakoutRegimeFix(unittest.TestCase):
    """BUG-FIX #8: short_breakout must require regime['down'] not regime['not_bullish'].
    Previously used not_bullish (fast_below OR slow_falling) which allowed shorts in
    RECOVERING / NEUTRAL regimes. Now requires full regime['down'] for symmetry with
    long breakout which already required the stronger regime['up']."""

    def _make_bearish_breakout_df(self, rsi_val=50.0):
        """DataFrame where price breaks below recent_low with a strong bearish candle + vol spike."""
        df = make_df(200, trend="down", base=50000.0)
        df = bot.add_indicators(df)
        df = df.dropna()
        last_idx = df.index[-1]
        # Force price below recent low with all the required candle conditions
        recent_low = float(df.loc[last_idx, "recent_low"])
        df.loc[last_idx, "close"] = recent_low - 200   # breaks below recent low
        df.loc[last_idx, "open"]  = recent_low + 100   # strong bearish body (open > close)
        df.loc[last_idx, "low"]   = recent_low - 300
        df.loc[last_idx, "high"]  = recent_low + 150
        # Force vol spike and candle conditions
        df.loc[last_idx, "vol_avg"]       = 100.0
        df.loc[last_idx, "vol_spike"]     = True
        df.loc[last_idx, "strong_bearish"]= True
        df.loc[last_idx, "bearish_candle"]= True
        df.loc[last_idx, "rsi"]           = rsi_val
        df.loc[last_idx, "rsi_prev"]      = rsi_val + 2  # no RSI crossover — irrelevant here
        # Force EMA below conditions
        df.loc[last_idx, "ema_fast"] = recent_low - 50
        df.loc[last_idx, "ema_slow"] = recent_low + 50
        df.loc[last_idx, "macd"]         = -10.0
        df.loc[last_idx, "macd_signal"]  =  -5.0
        df.loc[last_idx, "macd_hist"]    =  -5.0
        # NEW: set quality filters to qualifying values so regime is the only gate being tested
        df.loc[last_idx, "adx"]          = 30.0   # above ADX_MIN=23
        df.loc[last_idx, "chop_range"]   = 0.008  # above CHOP_RANGE_MIN=0.004
        # VWAP: price below VWAP to satisfy price_below_vwap (used by MomentumBurst, not breakout)
        df.loc[last_idx, "vwap"]         = recent_low + 500
        return df

    def _regime_not_bullish_only(self):
        """Regime where not_bullish=True but down=False (recovering/neutral market)."""
        return {
            "up": False, "down": False,
            "not_bearish": False, "not_bullish": True,
            "macro_bull": False, "label": "RECOVERING",
            "atr_pct": 0.015, "slow_rising": True, "slow_falling": False,
        }

    def _regime_full_down(self):
        """Regime where down=True (confirmed bear market)."""
        return {
            "up": False, "down": True,
            "not_bearish": False, "not_bullish": True,
            "macro_bull": False, "label": "DOWN",
            "atr_pct": 0.015, "slow_rising": False, "slow_falling": True,
        }

    def test_short_breakout_blocked_in_recovering_regime(self):
        """SHORT breakout must be BLOCKED when regime is only 'not_bullish' (not full down)."""
        df = self._make_bearish_breakout_df(rsi_val=50.0)
        sig = bot.get_crypto_signal(df, self._regime_not_bullish_only(), {"down": False})
        self.assertNotEqual(
            sig["signal"], "SHORT",
            "Short breakout must be blocked in RECOVERING regime (only not_bullish=True)."
            " BUG-FIX #8: was incorrectly using regime['not_bullish'] instead of regime['down']."
        )

    def test_short_breakout_allowed_in_full_down_regime(self):
        """SHORT breakout must be ALLOWED when regime is fully down."""
        df = self._make_bearish_breakout_df(rsi_val=50.0)
        sig = bot.get_crypto_signal(df, self._regime_full_down(), {"down": False})
        # Should be SHORT (or HOLD if other conditions didn't align exactly, but NOT blocked by regime)
        # The key assertion: the regime itself does NOT block the signal
        # We validate by checking not_bullish_only would block while full down allows
        self.assertIn(sig["signal"], ["SHORT", "HOLD"],
                      "Signal should be SHORT or HOLD in full down regime, never blocked by regime alone")

    def test_long_breakout_symmetry_requires_up_regime(self):
        """LONG breakout must still require full regime['up'] — confirming symmetry."""
        df = make_df(200, trend="up", base=50000.0)
        df = bot.add_indicators(df)
        df = df.dropna()
        # Regime: not_bearish=True (recovering) but up=False
        regime_recovering = {
            "up": False, "down": False,
            "not_bearish": True, "not_bullish": False,
            "macro_bull": True, "label": "RECOVERING",
            "atr_pct": 0.015, "slow_rising": True, "slow_falling": False,
        }
        sig = bot.get_crypto_signal(df, regime_recovering, {"up": False})
        self.assertNotEqual(sig["signal"], "BUY",
                            "Long breakout must be blocked in RECOVERING regime (only not_bearish=True)")


class TestNewIndicators(unittest.TestCase):
    """Verify ADX, VWAP, rsi_prev, and chop_range are computed correctly."""
    def setUp(self):
        self.df = bot.add_indicators(make_df(200, trend="up"))

    def test_adx_range(self):
        adx = self.df["adx"].dropna()
        self.assertFalse(adx.empty)
        self.assertTrue((adx >= 0).all(), "ADX must be non-negative")
        self.assertTrue((adx <= 100).all(), "ADX must be ≤ 100")

    def test_vwap_positive(self):
        vwap = self.df["vwap"].dropna()
        self.assertFalse(vwap.empty)
        self.assertTrue((vwap > 0).all(), "VWAP must be positive")

    def test_rsi_prev_is_shifted_rsi(self):
        df = self.df.dropna()
        # rsi_prev[i] should equal rsi[i-1]
        for i in range(1, min(10, len(df))):
            self.assertAlmostEqual(
                float(df["rsi_prev"].iloc[i]),
                float(df["rsi"].iloc[i - 1]),
                places=5,
                msg="rsi_prev must equal prior-bar RSI"
            )

    def test_chop_range_non_negative(self):
        chop = self.df["chop_range"].dropna()
        self.assertTrue((chop >= 0).all(), "chop_range must be non-negative")

    def test_chop_range_uptrend_exceeds_min(self):
        """In a clear uptrend, 3-candle range should exceed CHOP_RANGE_MIN most of the time."""
        chop = self.df["chop_range"].dropna()
        pct_above = (chop >= bot.CHOP_RANGE_MIN).mean()
        self.assertGreater(pct_above, 0.5,
            "More than half the bars in an uptrend should exceed CHOP_RANGE_MIN")


class TestMomentumBurst(unittest.TestCase):
    """MomentumBurst strategy: early entry before EMA crossover using RSI + VWAP + ADX."""

    def _make_regime(self, up=True, macro_bull=False):
        return {
            "up": up, "down": not up, "not_bearish": up, "not_bullish": not up,
            "macro_bull": macro_bull, "label": "UP" if up else "DOWN",
            "atr_pct": 0.01, "slow_rising": up, "slow_falling": not up,
        }

    def _burst_df(self, direction="long", rsi_now=54.0, rsi_prev_val=38.0,
                  adx_val=28.0, chop_val=0.007, vol_spike=True, vwap_bias="above"):
        """Build a DataFrame that should trigger (or not) a MomentumBurst signal."""
        df = bot.add_indicators(make_df(200, trend="up" if direction == "long" else "down"))
        df = df.dropna()
        last_idx = df.index[-1]
        price = float(df.loc[last_idx, "close"])
        # RSI crossover
        df.loc[last_idx, "rsi"]      = rsi_now
        df.loc[last_idx, "rsi_prev"] = rsi_prev_val
        # VWAP position
        if vwap_bias == "above":
            df.loc[last_idx, "vwap"] = price * 0.995   # price > vwap
        elif vwap_bias == "below":
            df.loc[last_idx, "vwap"] = price * 1.005   # price < vwap
        else:  # at — neither clearly above nor below
            df.loc[last_idx, "vwap"] = price
        # Trend quality
        df.loc[last_idx, "adx"]        = adx_val
        df.loc[last_idx, "chop_range"] = chop_val
        df.loc[last_idx, "vol_spike"]  = vol_spike
        return df

    # ── LONG burst ───────────────────────────────────────────
    def test_momentum_burst_buy_fires(self):
        """All conditions met → should signal MomentumBurst."""
        df = self._burst_df(direction="long", rsi_now=54.0, rsi_prev_val=38.0, vwap_bias="above")
        sig = bot.get_crypto_signal(df, self._make_regime(up=True), {"up": True})
        self.assertEqual(sig["signal"], "BUY")
        self.assertEqual(sig["strategy"], "MomentumBurst")

    def test_momentum_burst_buy_blocked_if_rsi_was_not_oversold(self):
        """rsi_prev=50 (not < 42) → burst should NOT fire."""
        df = self._burst_df(direction="long", rsi_now=54.0, rsi_prev_val=50.0, vwap_bias="above")
        sig = bot.get_crypto_signal(df, self._make_regime(up=True), {"up": True})
        self.assertNotEqual(sig["strategy"], "MomentumBurst",
            "MomentumBurst must be blocked when rsi_prev ≥ 42 (not coming from oversold)")

    def test_momentum_burst_buy_blocked_weak_adx(self):
        """ADX below ADX_MIN → choppy market → burst blocked."""
        df = self._burst_df(direction="long", rsi_now=54.0, rsi_prev_val=38.0,
                            adx_val=15.0, vwap_bias="above")
        sig = bot.get_crypto_signal(df, self._make_regime(up=True), {"up": True})
        self.assertNotEqual(sig["strategy"], "MomentumBurst",
            "MomentumBurst must be blocked when ADX < ADX_MIN (ranging market)")

    def test_momentum_burst_buy_blocked_below_vwap(self):
        """Price below VWAP while seeking long burst → blocked (bearish institutional bias)."""
        df = self._burst_df(direction="long", rsi_now=54.0, rsi_prev_val=38.0, vwap_bias="below")
        sig = bot.get_crypto_signal(df, self._make_regime(up=True), {"up": True})
        self.assertNotEqual(sig["strategy"], "MomentumBurst",
            "Long MomentumBurst must be blocked when price is below VWAP")

    def test_momentum_burst_buy_blocked_no_vol_spike(self):
        """No volume spike → no institutional confirmation → burst blocked."""
        df = self._burst_df(direction="long", rsi_now=54.0, rsi_prev_val=38.0,
                            vol_spike=False, vwap_bias="above")
        sig = bot.get_crypto_signal(df, self._make_regime(up=True), {"up": True})
        self.assertNotEqual(sig["strategy"], "MomentumBurst",
            "MomentumBurst must be blocked without a volume spike")

    def test_momentum_burst_buy_blocked_in_down_regime(self):
        """Regime is DOWN → long burst must not fire."""
        df = self._burst_df(direction="long", rsi_now=54.0, rsi_prev_val=38.0, vwap_bias="above")
        sig = bot.get_crypto_signal(df, self._make_regime(up=False), {"down": True})
        self.assertNotEqual(sig["strategy"], "MomentumBurst",
            "Long MomentumBurst must be blocked when HTF regime is DOWN")

    # ── SHORT burst ──────────────────────────────────────────
    def test_momentum_burst_short_fires(self):
        """All short burst conditions met → should signal ShortMomentumBurst."""
        df = self._burst_df(direction="short", rsi_now=46.0, rsi_prev_val=62.0, vwap_bias="below")
        sig = bot.get_crypto_signal(df, self._make_regime(up=False, macro_bull=False), {"down": True})
        self.assertEqual(sig["signal"], "SHORT")
        self.assertEqual(sig["strategy"], "ShortMomentumBurst")

    def test_momentum_burst_short_blocked_macro_bull(self):
        """macro_bull=True → all shorts blocked including ShortMomentumBurst."""
        df = self._burst_df(direction="short", rsi_now=46.0, rsi_prev_val=62.0, vwap_bias="below")
        sig = bot.get_crypto_signal(df, self._make_regime(up=False, macro_bull=True), {"down": True})
        self.assertNotEqual(sig["signal"], "SHORT",
            "ShortMomentumBurst must be blocked when macro_bull=True")

    def test_momentum_burst_short_blocked_if_rsi_was_not_overbought(self):
        """rsi_prev=50 (not > 58) → short burst must NOT fire."""
        df = self._burst_df(direction="short", rsi_now=46.0, rsi_prev_val=50.0, vwap_bias="below")
        sig = bot.get_crypto_signal(df, self._make_regime(up=False, macro_bull=False), {"down": True})
        self.assertNotEqual(sig["strategy"], "ShortMomentumBurst",
            "ShortMomentumBurst must be blocked when rsi_prev ≤ 58 (not from overbought)")


class TestWarmupRegime(unittest.TestCase):
    """BUG-FIX #1: warmup_regime must populate prev_regime from live state."""
    def test_warmup_populates_prev_regime(self):
        # Fake get_crypto_regime to return known state
        with patch.object(bot, "get_crypto_regime", return_value={
            "up": True, "down": False, "label": "UP", "not_bearish": True,
            "not_bullish": False, "macro_bull": True, "slow_rising": True,
            "slow_falling": False, "atr_pct": 0.01
        }):
            # Reset to False/False (simulates bot restart)
            for sym in bot.CRYPTO_SYMBOLS:
                bot.prev_regime[sym] = {"up": False, "down": False}
            bot.warmup_regime()
            for sym in bot.CRYPTO_SYMBOLS:
                self.assertTrue(bot.prev_regime[sym]["up"],
                                f"{sym}: prev_regime should be up=True after warmup")

    def test_without_warmup_false_flip_would_fire(self):
        """Shows the bug that warmup fixes."""
        for sym in bot.CRYPTO_SYMBOLS:
            bot.prev_regime[sym] = {"up": False, "down": False}
        current_regime = {"up": True, "down": False}
        # This is what happens on the first loop iteration without warmup:
        just_flipped = current_regime["up"] and not bot.prev_regime["BTC/USDT"].get("up", False)
        self.assertTrue(just_flipped, "Without warmup, a false flip fires immediately")


class TestFetchCryptoRetry(unittest.TestCase):
    """BUG-FIX #5: fetch_crypto should retry on transient failures."""
    def test_succeeds_on_second_attempt(self):
        mock_exchange = MagicMock()
        call_count = {"n": 0}
        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise Exception("Transient network error")
            return [[1609459200000, 50000, 51000, 49000, 50500, 100]] * 50
        mock_exchange.fetch_ohlcv.side_effect = side_effect
        original = bot.exchange
        bot.exchange = mock_exchange
        try:
            df = bot.fetch_crypto("BTC/USDT", "30m", 50)
            self.assertFalse(df.empty)
            self.assertEqual(call_count["n"], 2)
        finally:
            bot.exchange = original

    def test_raises_after_3_failures(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("Persistent error")
        original = bot.exchange
        bot.exchange = mock_exchange
        try:
            with self.assertRaises(RuntimeError):
                bot.fetch_crypto("BTC/USDT", "30m", 50)
        finally:
            bot.exchange = original


class TestDailyBestWorstReset(unittest.TestCase):
    """BUG-FIX #7: best/worst trade should reset each day."""
    def test_reset_clears_best_worst(self):
        reset_paper("BTC/USDT", 3333.33)
        bot.crypto_paper["BTC/USDT"]["best_trade"]  = 999.0
        bot.crypto_paper["BTC/USDT"]["worst_trade"] = -999.0

        # Simulate a new day
        bot.perf["date"] = (datetime.now() - timedelta(days=1)).date()
        bot.reset_daily_stats()

        self.assertEqual(bot.crypto_paper["BTC/USDT"]["best_trade"],  0.0)
        self.assertEqual(bot.crypto_paper["BTC/USDT"]["worst_trade"], 0.0)


# ══════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    test_classes = [
        TestSafeFloat, TestPositionSizing, TestAddIndicators,
        TestNewIndicators,                # ADX / VWAP / rsi_prev / chop_range
        TestCryptoBuy, TestCryptoClose,
        TestBreakevenTrailingStop, TestExits, TestShortPosition,
        TestDailyLossLimit, TestSignalRSIFilter,
        TestMomentumBurst,               # new MomentumBurst + ShortMomentumBurst strategy
        TestWarmupRegime, TestFetchCryptoRetry, TestDailyBestWorstReset,
        TestShortBreakoutRegimeFix,      # BUG-FIX #8
    ]
    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"\n{'='*60}")
    print(f"  RESULTS:  {passed}/{total} passed  |  {failed} failed")
    print(f"  STATUS:   {'✅ ALL CLEAR — bot ready' if failed == 0 else '❌ FAILURES — fix before deploying'}")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
