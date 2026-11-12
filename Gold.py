"""
XAUUSD Quant Edge PRO - MetaTrader 5
Converted from the supplied Pine Script strategy.

IMPORTANT:
1. Change the exposed MT5 password before using this bot.
2. Put credentials in environment variables:
      set MT5_LOGIN= ""
      set MT5_PASSWORD=""
      set MT5_SERVER=""
3. Install:
      pip install MetaTrader5 pandas numpy python-dotenv
4. Make sure the XM MT5 desktop terminal is installed, logged in, and
   allows algorithmic trading.
5. Start on a DEMO account first.

The bot:
- discovers gold/XAU symbols available in MT5
- uses one position per symbol (1:1 / no pyramiding)
- evaluates closed candles to avoid signal flicker/repainting
- uses EMA 9/21, RSI 14, volume, ATR 14, Bollinger Bands
- uses M15 EMA 50 as HTF confirmation
- reproduces the supplied quant-score logic
- sends a market order immediately after a confirmed signal
- sets ATR SL and TP3
- moves SL to breakeven after TP1
- trails with ATR
- checks execution results and reconnects if needed

The supplied Pine strategy's TP1/TP2 are management milestones while the
actual exit is TP3; this implementation follows that behavior.
"""

from __future__ import annotations

import os
import sys
import time
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# CONFIG
# ============================================================

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "169236160"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "Kanishka@0613")
MT5_SERVER = os.getenv("MT5_SERVER", "XMGlobal-MT5 2")
MT5_PATH = os.getenv("MT5_PATH", "").strip() or None

# Signal timeframe. The supplied Pine strategy can be used on the chart
# timeframe. M1 is selected here for fast gold scalping.
TIMEFRAME = mt5.TIMEFRAME_M1
TIMEFRAME_NAME = "M1"

# Higher timeframe confirmation from the supplied Pine strategy.
HTF_TIMEFRAME = mt5.TIMEFRAME_M15
HTF_TIMEFRAME_NAME = "M15"

# Core indicator settings from Pine.
FAST_EMA = 9
SLOW_EMA = 21
RSI_LEN = 14
RSI_BULL = 55.0
RSI_BEAR = 45.0

VOLUME_LEN = 20
VOLUME_MULTIPLIER = 1.10
USE_VOLUME = True

ATR_LEN = 14
ATR_SL_MULTIPLIER = 1.5
TP1_RR = 1.0
TP2_RR = 2.0
TP3_RR = 3.0

BB_LEN = 20
BB_MULT = 2.0

USE_HTF = True
HTF_EMA_LEN = 50

MIN_SCORE = 60.0

USE_BREAKEVEN = True
USE_TRAILING = True
TRAIL_ATR_MULTIPLIER = 1.0

# "Both", "Long Only", "Short Only"
TRADE_DIRECTION = "Both"

# One active trade per symbol.
ONE_POSITION_PER_SYMBOL = True

# Only evaluate the latest CLOSED candle.
CONFIRM_CLOSED_CANDLE = True

# Poll interval. MT5 terminal/network latency dominates execution;
# extremely tiny sleeps do not guarantee lower broker latency.
POLL_SECONDS = 0.10

# Number of candles downloaded for calculations.
BARS = 250
HTF_BARS = 150

# Risk-based lot sizing.
# Set RISK_PERCENT=0 for fixed lot mode.
# Use a realistic broker-safe default to avoid invalid MT5 volume rejections.
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.00"))
FIXED_LOT = float(os.getenv("FIXED_LOT", "0.01"))

# Maximum spread in points. 0 disables this filter.
MAX_SPREAD_POINTS = float(os.getenv("MAX_SPREAD_POINTS", "0"))

# Magic number.
MAGIC = 26092401

# Deviation allowed for market execution.
DEVIATION_POINTS = int(os.getenv("DEVIATION_POINTS", "50"))

# Retry failed market entries a few times to handle temporary MT5 rejections.
ORDER_RETRY_ATTEMPTS = int(os.getenv("ORDER_RETRY_ATTEMPTS", "3"))
ORDER_RETRY_DELAY = float(os.getenv("ORDER_RETRY_DELAY", "0.75"))

# Order comment.
ORDER_COMMENT = "Yadavji_QuantEdge"

# Gold symbol discovery.
# Symbols containing any of these strings are considered candidates.
GOLD_KEYWORDS = (
    "XAU",
    "GOLD",
)

# Optional exact symbols separated by commas.
# Example:
# GOLD_SYMBOLS=XAUUSD,XAUUSDm,GOLD
EXACT_SYMBOLS = [
    x.strip()
    for x in os.getenv("GOLD_SYMBOLS", "").split(",")
    if x.strip()
]

# Optional maximum number of symbols.
MAX_GOLD_SYMBOLS = int(os.getenv("MAX_GOLD_SYMBOLS", "0"))

# Logging.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("XAUUSD_QuantEdge")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Signal:
    symbol: str
    direction: str
    score: float
    opposite_score: float
    close: float
    atr: float
    bar_time: int


@dataclass
class PositionState:
    symbol: str
    ticket: int
    direction: str
    entry: float
    initial_sl: float
    current_sl: float
    tp1: float
    tp2: float
    tp3: float
    tp1_reached: bool = False
    tp2_reached: bool = False


# ============================================================
# MT5 CONNECTION
# ============================================================

def connect_mt5() -> bool:
    """Connect/login to the already-installed MT5 terminal."""
    if not MT5_PASSWORD:
        log.error(
            "MT5_PASSWORD is empty. Set MT5_PASSWORD as an environment variable."
        )
        return False

    shutdown_mt5()

    if MT5_PATH:
        ok = mt5.initialize(
            path=MT5_PATH,
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
        )
    else:
        ok = mt5.initialize(
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
        )

    if not ok:
        log.error("MT5 initialize/login failed: %s", mt5.last_error())
        return False

    account = mt5.account_info()
    terminal = mt5.terminal_info()

    if account is None:
        log.error("MT5 account_info() failed: %s", mt5.last_error())
        return False

    log.info(
        "CONNECTED | login=%s | server=%s | balance=%.2f | equity=%.2f",
        account.login,
        account.server,
        account.balance,
        account.equity,
    )

    if terminal is not None:
        log.info(
            "TERMINAL | trade_allowed=%s | connected=%s",
            getattr(terminal, "trade_allowed", None),
            getattr(terminal, "connected", None),
        )

    return True


def shutdown_mt5() -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass


def ensure_connection() -> bool:
    info = mt5.account_info()
    terminal = mt5.terminal_info()

    if info is not None and terminal is not None:
        connected = getattr(terminal, "connected", True)
        if connected:
            return True

    log.warning("MT5 connection lost. Reconnecting...")
    return connect_mt5()


# ============================================================
# SYMBOL DISCOVERY
# ============================================================

def discover_gold_symbols() -> List[str]:
    """
    Discover all visible/available gold symbols in MT5.

    If GOLD_SYMBOLS is set, those symbols are preferred.
    Otherwise all symbols whose name contains XAU or GOLD are returned.
    """
    symbols = mt5.symbols_get()
    if symbols is None:
        log.error("symbols_get failed: %s", mt5.last_error())
        return []

    available = {s.name for s in symbols}

    if EXACT_SYMBOLS:
        selected = [s for s in EXACT_SYMBOLS if s in available]
        missing = [s for s in EXACT_SYMBOLS if s not in available]

        for name in missing:
            log.warning("Configured gold symbol not found: %s", name)

        for name in selected:
            mt5.symbol_select(name, True)

        return selected

    found = []
    for s in symbols:
        name_upper = s.name.upper()

        if any(k in name_upper for k in GOLD_KEYWORDS):
            # Select the symbol in Market Watch.
            if mt5.symbol_select(s.name, True):
                found.append(s.name)

    found = sorted(set(found))

    if MAX_GOLD_SYMBOLS > 0:
        found = found[:MAX_GOLD_SYMBOLS]

    return found


# ============================================================
# MARKET DATA
# ============================================================

def get_rates(
    symbol: str,
    timeframe: int,
    count: int,
) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < 60:
        return None

    df = pd.DataFrame(rates)

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Keep only useful columns.
    return df[
        [
            "time",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]
    ].copy()


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100.0 - (100.0 / (1.0 + rs))

    # Handle the all-gain/all-loss edge cases.
    value = value.where(avg_loss != 0, 100.0)
    value = value.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)

    return value


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["close"].shift(1)

    a = df["high"] - df["low"]
    b = (df["high"] - previous_close).abs()
    c = (df["low"] - previous_close).abs()

    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = true_range(df)

    # Wilder-style ATR, matching TradingView's common ATR construction.
    return tr.ewm(
        alpha=1.0 / length,
        adjust=False,
        min_periods=length,
    ).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def bollinger_bands(
    close: pd.Series,
    length: int,
    multiplier: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    basis = sma(close, length)
    dev = close.rolling(
        length,
        min_periods=length,
    ).std(ddof=1) * multiplier

    upper = basis + dev
    lower = basis - dev

    return basis, upper, lower


def roc(series: pd.Series, length: int) -> pd.Series:
    return (series / series.shift(length) - 1.0) * 100.0


# ============================================================
# SIGNAL ENGINE
# ============================================================

def calculate_signal(symbol: str) -> Optional[Signal]:
    """
    Calculate the signal from the latest CLOSED candle.

    This intentionally ignores the currently forming candle to prevent
    an intrabar signal from appearing and disappearing.
    """
    df = get_rates(symbol, TIMEFRAME, BARS)
    htf = get_rates(symbol, HTF_TIMEFRAME, HTF_BARS)

    if df is None or htf is None:
        return None

    if len(df) < 80 or len(htf) < HTF_EMA_LEN + 5:
        return None

    # The final row is the current/forming candle.
    # -2 is the latest fully closed candle.
    idx = -2

    close = df["close"]

    ema_fast = ema(close, FAST_EMA)
    ema_slow = ema(close, SLOW_EMA)
    rsi_v = rsi(close, RSI_LEN)
    atr_v = atr(df, ATR_LEN)

    volume_ma = sma(df["tick_volume"], VOLUME_LEN)
    volume_confirmed = (
        df["tick_volume"] > volume_ma * VOLUME_MULTIPLIER
    )

    bb_basis, bb_upper, bb_lower = bollinger_bands(
        close,
        BB_LEN,
        BB_MULT,
    )

    # ATR average / volatility regime.
    atr_average = sma(atr_v, 50)

    # ROC.
    roc5 = roc(close, 5)
    roc10 = roc(close, 10)

    # HTF.
    htf_close = htf["close"]
    htf_ema = ema(htf_close, HTF_EMA_LEN)

    htf_bull = htf_close.iloc[-2] > htf_ema.iloc[-2]
    htf_bear = htf_close.iloc[-2] < htf_ema.iloc[-2]

    # Current closed candle values.
    c = float(close.iloc[idx])
    o = float(df["open"].iloc[idx])

    ef = float(ema_fast.iloc[idx])
    es = float(ema_slow.iloc[idx])
    rv = float(rsi_v.iloc[idx])
    av = float(atr_v.iloc[idx])

    if not all(np.isfinite(x) for x in [c, ef, es, rv, av]):
        return None

    vm = float(volume_ma.iloc[idx])
    volume_ok = bool(volume_confirmed.iloc[idx])

    atr_avg = float(atr_average.iloc[idx])

    if not np.isfinite(atr_avg) or atr_avg <= 0:
        return None

    bb_lo = float(bb_lower.iloc[idx])
    bb_hi = float(bb_upper.iloc[idx])

    r5 = float(roc5.iloc[idx])
    r10 = float(roc10.iloc[idx])

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    bull_momentum = (
        ef > es
        and c > ef
        and rv > RSI_BULL
    )

    bear_momentum = (
        ef < es
        and c < ef
        and rv < RSI_BEAR
    )

    # --------------------------------------------------------
    # Mean reversion
    # --------------------------------------------------------

    bull_mean_reversion = (
        c < bb_lo
        and rv < 40.0
    )

    bear_mean_reversion = (
        c > bb_hi
        and rv > 60.0
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    normal_volatility = (
        av > atr_avg * 0.70
        and av < atr_avg * 2.0
    )

    high_volatility = av > atr_avg * 2.0

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    positive_roc = r5 > 0 and r10 > 0
    negative_roc = r5 < 0 and r10 < 0

    # --------------------------------------------------------
    # Quant Score
    #
    # This follows the supplied Pine scoring:
    # Momentum = 25
    # ROC = 15
    # Volume = 15
    # Volatility = 10
    # HTF = 20
    # Mean Reversion = 15
    # --------------------------------------------------------

    bull_score = 0.0
    bear_score = 0.0

    # Momentum.
    if ef > es:
        bull_score += 15.0

    if c > ef:
        bull_score += 5.0

    if rv > RSI_BULL:
        bull_score += 5.0

    if ef < es:
        bear_score += 15.0

    if c < ef:
        bear_score += 5.0

    if rv < RSI_BEAR:
        bear_score += 5.0

    # ROC.
    if positive_roc:
        bull_score += 15.0

    if negative_roc:
        bear_score += 15.0

    # Volume.
    if USE_VOLUME:
        if volume_ok and c > o:
            bull_score += 15.0

        if volume_ok and c < o:
            bear_score += 15.0
    else:
        bull_score += 7.5
        bear_score += 7.5

    # Volatility.
    if normal_volatility:
        bull_score += 5.0
        bear_score += 5.0

    # HTF.
    if USE_HTF:
        if htf_bull:
            bull_score += 20.0

        if htf_bear:
            bear_score += 20.0
    else:
        bull_score += 10.0
        bear_score += 10.0

    # Mean reversion.
    if bull_mean_reversion:
        bull_score += 15.0

    if bear_mean_reversion:
        bear_score += 15.0

    bull_probability = min(bull_score, 100.0)
    bear_probability = min(bear_score, 100.0)

    bull_signal = (
        bull_probability >= MIN_SCORE
        and bull_probability > bear_probability
    )

    bear_signal = (
        bear_probability >= MIN_SCORE
        and bear_probability > bull_probability
    )

    bar_time = int(df["time"].iloc[idx].timestamp())

    if bull_signal and TRADE_DIRECTION in ("Both", "Long Only"):
        return Signal(
            symbol=symbol,
            direction="BUY",
            score=bull_probability,
            opposite_score=bear_probability,
            close=c,
            atr=av,
            bar_time=bar_time,
        )

    if bear_signal and TRADE_DIRECTION in ("Both", "Short Only"):
        return Signal(
            symbol=symbol,
            direction="SELL",
            score=bear_probability,
            opposite_score=bull_probability,
            close=c,
            atr=av,
            bar_time=bar_time,
        )

    return None


# ============================================================
# SYMBOL / PRICE UTILITIES
# ============================================================

def symbol_info(symbol: str):
    info = mt5.symbol_info(symbol)

    if info is None:
        log.error("%s | symbol_info failed: %s", symbol, mt5.last_error())
        return None

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            log.error("%s | could not select symbol", symbol)
            return None

    return info


def normalize_price(symbol: str, price: float) -> float:
    info = symbol_info(symbol)
    if info is None:
        return price

    return round(float(price), int(info.digits))


def normalize_volume(symbol: str, volume: float) -> float:
    info = symbol_info(symbol)
    if info is None:
        return volume

    min_vol = float(info.volume_min)
    max_vol = float(info.volume_max)
    step = float(info.volume_step)

    if max_vol <= 0:
        return max(min_vol, 0.0)

    if min_vol > max_vol:
        min_vol, max_vol = max_vol, min_vol

    clamped = max(min_vol, min(max_vol, float(volume)))

    if step <= 0:
        normalized = clamped
    else:
        steps = math.floor(clamped / step + 1e-10)
        normalized = steps * step
        if normalized < min_vol:
            normalized = min_vol
        if normalized > max_vol:
            normalized = max_vol

    rounded = round(float(normalized), 8)
    if abs(rounded - float(volume)) > 1e-9:
        log.info(
            "%s | volume adjusted from %.8f to %.8f (min=%.8f max=%.8f step=%.8f)",
            symbol,
            float(volume),
            rounded,
            min_vol,
            max_vol,
            step,
        )

    return rounded


def get_tick(symbol: str):
    return mt5.symbol_info_tick(symbol)


def spread_points(symbol: str) -> Optional[float]:
    info = symbol_info(symbol)
    tick = get_tick(symbol)

    if info is None or tick is None:
        return None

    if info.point <= 0:
        return None

    return (float(tick.ask) - float(tick.bid)) / float(info.point)


def get_filling_mode(symbol: str) -> int:
    """
    Select a filling mode supported by the symbol.
    """
    info = symbol_info(symbol)

    if info is None:
        return mt5.ORDER_FILLING_IOC

    filling = getattr(info, "filling_mode", None)

    if filling is not None:
        # MT5 exposes a bitmask on many brokers.
        if filling & mt5.ORDER_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK

        if filling & mt5.ORDER_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC

    return mt5.ORDER_FILLING_IOC


def send_order_with_retry(request: dict, symbol: str):
    """Retry market entries a few times when MT5 rejects a valid signal."""
    last_result = None

    for attempt in range(1, max(1, ORDER_RETRY_ATTEMPTS) + 1):
        if not ensure_connection():
            time.sleep(1.0)
            continue

        # Validate before execution to catch obvious broker-side issues early.
        check = mt5.order_check(request)
        if check is not None and check.retcode != mt5.TRADE_RETCODE_DONE:
            log.warning(
                "%s | order_check failed on attempt %s/%s | retcode=%s | comment=%s",
                symbol,
                attempt,
                ORDER_RETRY_ATTEMPTS,
                getattr(check, "retcode", "unknown"),
                getattr(check, "comment", ""),
            )

            if attempt < ORDER_RETRY_ATTEMPTS:
                time.sleep(ORDER_RETRY_DELAY)
                continue

            return None

        result = mt5.order_send(request)
        last_result = result

        if result is None:
            log.warning(
                "%s | order_send returned None on attempt %s/%s | %s",
                symbol,
                attempt,
                ORDER_RETRY_ATTEMPTS,
                mt5.last_error(),
            )
            if attempt < ORDER_RETRY_ATTEMPTS:
                time.sleep(ORDER_RETRY_DELAY)
                continue
            return None

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return result

        log.warning(
            "%s | order attempt %s/%s rejected | retcode=%s | comment=%s",
            symbol,
            attempt,
            ORDER_RETRY_ATTEMPTS,
            getattr(result, "retcode", "unknown"),
            getattr(result, "comment", ""),
        )

        if attempt < ORDER_RETRY_ATTEMPTS:
            time.sleep(ORDER_RETRY_DELAY)
            continue

    return last_result


# ============================================================
# RISK / LOT SIZE
# ============================================================

def calculate_risk_lot(
    symbol: str,
    entry: float,
    stop: float,
) -> float:
    """
    Calculate volume so approximately RISK_PERCENT of equity is at risk.

    Uses MT5 order_calc_profit, which is preferable to assuming a fixed
    gold contract specification.
    """
    if RISK_PERCENT <= 0:
        return normalize_volume(symbol, FIXED_LOT)

    account = mt5.account_info()
    if account is None:
        return normalize_volume(symbol, FIXED_LOT)

    risk_money = float(account.equity) * RISK_PERCENT / 100.0

    if risk_money <= 0:
        return normalize_volume(symbol, FIXED_LOT)

    order_type = (
        mt5.ORDER_TYPE_BUY
        if entry > stop
        else mt5.ORDER_TYPE_SELL
    )

    # Profit/loss for one lot between entry and SL.
    loss_one_lot = mt5.order_calc_profit(
        order_type,
        symbol,
        1.0,
        entry,
        stop,
    )

    if loss_one_lot is None:
        log.warning(
            "%s | order_calc_profit failed; using fixed lot %.4f",
            symbol,
            FIXED_LOT,
        )
        return normalize_volume(symbol, FIXED_LOT)

    loss_one_lot = abs(float(loss_one_lot))

    if loss_one_lot <= 0:
        return normalize_volume(symbol, FIXED_LOT)

    volume = risk_money / loss_one_lot

    return normalize_volume(symbol, volume)


# ============================================================
# STOP-LEVEL SAFETY
# ============================================================

def enforce_minimum_stop_distance(
    symbol: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
) -> Tuple[float, float]:
    info = symbol_info(symbol)

    if info is None:
        return sl, tp

    point = float(info.point)
    stops_level = int(getattr(info, "trade_stops_level", 0))

    if point <= 0 or stops_level <= 0:
        return (
            normalize_price(symbol, sl),
            normalize_price(symbol, tp),
        )

    min_distance = stops_level * point

    if direction == "BUY":
        if entry - sl < min_distance:
            sl = entry - min_distance

        if tp - entry < min_distance:
            tp = entry + min_distance

    else:
        if sl - entry < min_distance:
            sl = entry + min_distance

        if entry - tp < min_distance:
            tp = entry - min_distance

    return (
        normalize_price(symbol, sl),
        normalize_price(symbol, tp),
    )


# ============================================================
# POSITION DISCOVERY
# ============================================================

def get_bot_positions(symbol: Optional[str] = None):
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()

    if positions is None:
        return []

    return [
        p for p in positions
        if int(getattr(p, "magic", 0)) == MAGIC
    ]


def get_symbol_bot_position(symbol: str):
    positions = get_bot_positions(symbol)

    if not positions:
        return None

    # 1:1 mode means we expect at most one.
    return positions[0]


def has_any_open_position(symbol: str) -> bool:
    """Return True when the symbol already has an open position, regardless of magic."""
    positions = mt5.positions_get(symbol=symbol)
    return bool(positions and len(positions) > 0)


# ============================================================
# ORDER SEND
# ============================================================

def send_market_order(
    signal: Signal,
) -> Optional[PositionState]:
    symbol = signal.symbol

    info = symbol_info(symbol)
    tick = get_tick(symbol)

    if info is None or tick is None:
        return None

    # Spread filter.
    sp = spread_points(symbol)

    if (
        MAX_SPREAD_POINTS > 0
        and sp is not None
        and sp > MAX_SPREAD_POINTS
    ):
        log.warning(
            "%s | signal=%s blocked by spread %.1f > %.1f points",
            symbol,
            signal.direction,
            sp,
            MAX_SPREAD_POINTS,
        )
        return None

    if signal.direction == "BUY":
        entry = float(tick.ask)

        initial_sl = entry - signal.atr * ATR_SL_MULTIPLIER

        risk_distance = entry - initial_sl

        tp1 = entry + risk_distance * TP1_RR
        tp2 = entry + risk_distance * TP2_RR
        tp3 = entry + risk_distance * TP3_RR

    else:
        entry = float(tick.bid)

        initial_sl = entry + signal.atr * ATR_SL_MULTIPLIER

        risk_distance = initial_sl - entry

        tp1 = entry - risk_distance * TP1_RR
        tp2 = entry - risk_distance * TP2_RR
        tp3 = entry - risk_distance * TP3_RR

    initial_sl, tp3 = enforce_minimum_stop_distance(
        symbol,
        signal.direction,
        entry,
        initial_sl,
        tp3,
    )

    # Recalculate management targets from the actual SL after
    # broker minimum-stop adjustment.
    if signal.direction == "BUY":
        risk_distance = entry - initial_sl
        tp1 = entry + risk_distance * TP1_RR
        tp2 = entry + risk_distance * TP2_RR
        tp3 = entry + risk_distance * TP3_RR
    else:
        risk_distance = initial_sl - entry
        tp1 = entry - risk_distance * TP1_RR
        tp2 = entry - risk_distance * TP2_RR
        tp3 = entry - risk_distance * TP3_RR

    initial_sl, tp3 = enforce_minimum_stop_distance(
        symbol,
        signal.direction,
        entry,
        initial_sl,
        tp3,
    )

    volume = calculate_risk_lot(
        symbol,
        entry,
        initial_sl,
    )

    if volume <= 0:
        log.error("%s | invalid calculated volume", symbol)
        return None

    order_type = (
        mt5.ORDER_TYPE_BUY
        if signal.direction == "BUY"
        else mt5.ORDER_TYPE_SELL
    )

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": entry,
        "sl": initial_sl,
        "tp": tp3,
        "deviation": DEVIATION_POINTS,
        "magic": MAGIC,
        "comment": ORDER_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(symbol),
    }

    log.info(
        "%s | %s SIGNAL | score=%.1f | entry=%.*f | SL=%.*f | "
        "TP1=%.*f | TP2=%.*f | TP3=%.*f | volume=%.4f",
        symbol,
        signal.direction,
        signal.score,
        info.digits,
        entry,
        info.digits,
        initial_sl,
        info.digits,
        tp1,
        info.digits,
        tp2,
        info.digits,
        tp3,
        volume,
    )

    result = send_order_with_retry(request, symbol)

    if result is None:
        log.error(
            "%s | order_send failed after %s attempts",
            symbol,
            ORDER_RETRY_ATTEMPTS,
        )
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(
            "%s | ORDER FAILED | retcode=%s | comment=%s | "
            "request_id=%s",
            symbol,
            result.retcode,
            getattr(result, "comment", ""),
            getattr(result, "request_id", ""),
        )

        if result.retcode in (
            getattr(mt5, "TRADE_RETCODE_INVALID_FILL", -1),
            getattr(mt5, "TRADE_RETCODE_INVALID", -1),
        ):
            log.warning(
                "%s | broker rejected order parameters/filling mode. "
                "Check symbol trading settings.",
                symbol,
            )

        return None

    ticket = int(
        getattr(result, "order", 0)
        or getattr(result, "deal", 0)
    )

    log.info(
        "%s | ORDER ACCEPTED | order=%s | deal=%s | price=%s",
        symbol,
        getattr(result, "order", 0),
        getattr(result, "deal", 0),
        getattr(result, "price", entry),
    )

    # Find actual open position. For hedging accounts this is preferable
    # to trusting the requested price.
    time.sleep(0.02)

    position = get_symbol_bot_position(symbol)

    if position is None:
        log.warning(
            "%s | order accepted but active bot position was not found yet",
            symbol,
        )

        # Return a state using requested values.
        actual_entry = entry
        actual_ticket = ticket
    else:
        actual_entry = float(position.price_open)
        actual_ticket = int(position.ticket)

    # Recalculate actual management levels from actual entry.
    if signal.direction == "BUY":
        actual_sl = actual_entry - signal.atr * ATR_SL_MULTIPLIER
        risk = actual_entry - actual_sl
        actual_tp1 = actual_entry + risk * TP1_RR
        actual_tp2 = actual_entry + risk * TP2_RR
        actual_tp3 = actual_entry + risk * TP3_RR
    else:
        actual_sl = actual_entry + signal.atr * ATR_SL_MULTIPLIER
        risk = actual_sl - actual_entry
        actual_tp1 = actual_entry - risk * TP1_RR
        actual_tp2 = actual_entry - risk * TP2_RR
        actual_tp3 = actual_entry - risk * TP3_RR

    return PositionState(
        symbol=symbol,
        ticket=actual_ticket,
        direction=signal.direction,
        entry=actual_entry,
        initial_sl=actual_sl,
        current_sl=actual_sl,
        tp1=actual_tp1,
        tp2=actual_tp2,
        tp3=actual_tp3,
    )


# ============================================================
# SL MODIFICATION
# ============================================================

def as_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def modify_position_sl(
    position,
    new_sl: float,
) -> bool:
    symbol = position.symbol

    info = symbol_info(symbol)
    if info is None:
        return False

    new_sl = normalize_price(symbol, new_sl)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": int(position.ticket),
        "sl": new_sl,
        "tp": as_float(getattr(position, "tp", 0.0), 0.0),
    }

    result = mt5.order_send(request)

    if result is None:
        log.error(
            "%s | SL modification returned None: %s",
            symbol,
            mt5.last_error(),
        )
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.warning(
            "%s | SL modification failed | retcode=%s | comment=%s",
            symbol,
            result.retcode,
            getattr(result, "comment", ""),
        )
        return False

    log.info(
        "%s | SL UPDATED | ticket=%s | new_sl=%.*f",
        symbol,
        position.ticket,
        info.digits,
        new_sl,
    )

    return True


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def manage_position(position) -> None:
    symbol = position.symbol

    tick = get_tick(symbol)
    if tick is None:
        return

    direction = (
        "BUY"
        if int(position.type) == mt5.POSITION_TYPE_BUY
        else "SELL"
    )

    entry = float(position.price_open)
    current_sl_value = as_float(getattr(position, "sl", 0.0), 0.0)
    current_sl = current_sl_value if current_sl_value > 0 else None

    # We need current ATR.
    df = get_rates(symbol, TIMEFRAME, 100)

    if df is None:
        return

    atr_series = atr(df, ATR_LEN)
    av = float(atr_series.iloc[-2])

    if not np.isfinite(av) or av <= 0:
        return

    if direction == "BUY":
        current_price = float(tick.bid)
        initial_risk = av * ATR_SL_MULTIPLIER

        tp1 = entry + initial_risk * TP1_RR
        tp2 = entry + initial_risk * TP2_RR

        # TP1 -> breakeven.
        if USE_BREAKEVEN and current_price >= tp1:
            desired_sl = entry

            if current_sl is None or desired_sl > current_sl:
                modify_position_sl(position, desired_sl)
                return

        # ATR trailing.
        if USE_TRAILING:
            trailing_sl = current_price - av * TRAIL_ATR_MULTIPLIER

            if current_sl is None or trailing_sl > current_sl:
                modify_position_sl(position, trailing_sl)

    else:
        current_price = float(tick.ask)
        initial_risk = av * ATR_SL_MULTIPLIER

        tp1 = entry - initial_risk * TP1_RR
        tp2 = entry - initial_risk * TP2_RR

        # TP1 -> breakeven.
        if USE_BREAKEVEN and current_price <= tp1:
            desired_sl = entry

            if current_sl is None or desired_sl < current_sl:
                modify_position_sl(position, desired_sl)
                return

        # ATR trailing.
        if USE_TRAILING:
            trailing_sl = current_price + av * TRAIL_ATR_MULTIPLIER

            if current_sl is None or trailing_sl < current_sl:
                modify_position_sl(position, trailing_sl)


# ============================================================
# SIGNAL STATE
# ============================================================

last_processed_bar: Dict[str, int] = {}


def is_new_closed_bar(symbol: str, bar_time: int) -> bool:
    previous = last_processed_bar.get(symbol)

    if previous == bar_time:
        return False

    last_processed_bar[symbol] = bar_time
    return True


# ============================================================
# MAIN SYMBOL PROCESSING
# ============================================================

def process_symbol(symbol: str) -> None:
    # Strict 1:1 enforcement: no new entries while any position is open.
    position = get_symbol_bot_position(symbol)
    if position is not None:
        manage_position(position)
        return

    if ONE_POSITION_PER_SYMBOL and has_any_open_position(symbol):
        log.info("%s | 1:1 mode: existing open position exists, skipping new entry", symbol)
        return

    signal = calculate_signal(symbol)

    if signal is None:
        return

    # Critical duplicate protection:
    # only act once per closed candle.
    if not is_new_closed_bar(symbol, signal.bar_time):
        return

    send_market_order(signal)


# ============================================================
# STARTUP DIAGNOSTICS
# ============================================================

def print_configuration(symbols: List[str]) -> None:
    log.info("=" * 72)
    log.info("YADAVJI QUANT EDGE PRO - MT5")
    log.info("=" * 72)
    log.info("Account: %s", MT5_LOGIN)
    log.info("Server: %s", MT5_SERVER)
    log.info("Timeframe: %s", TIMEFRAME_NAME)
    log.info("HTF: %s", HTF_TIMEFRAME_NAME)
    log.info("Symbols: %s", ", ".join(symbols))
    log.info("EMA: %s/%s", FAST_EMA, SLOW_EMA)
    log.info("RSI: %s | bull>%s | bear<%s", RSI_LEN, RSI_BULL, RSI_BEAR)
    log.info("ATR: %s | SL x%.2f", ATR_LEN, ATR_SL_MULTIPLIER)
    log.info("TP RR: %.1f / %.1f / %.1f", TP1_RR, TP2_RR, TP3_RR)
    log.info("HTF EMA: %s", HTF_EMA_LEN)
    log.info("Minimum score: %.1f", MIN_SCORE)
    log.info("Risk: %.2f%% | fixed lot: %.2f", RISK_PERCENT, FIXED_LOT)
    log.info("Polling: %.3fs", POLL_SECONDS)
    log.info("Magic: %s", MAGIC)
    log.info("=" * 72)


# ============================================================
# MAIN LOOP
# ============================================================

def main() -> None:
    log.info("Starting XAUUSD Quant Edge MT5 bot...")

    if not connect_mt5():
        sys.exit(1)

    symbols = discover_gold_symbols()

    if not symbols:
        log.error(
            "No gold symbols found. Check MT5 Market Watch or set "
            "GOLD_SYMBOLS=XAUUSD,XAUUSDm,etc."
        )
        shutdown_mt5()
        sys.exit(1)

    print_configuration(symbols)

    # Warm-up: initialize last processed bars without entering immediately
    # on a stale historical signal.
    for symbol in symbols:
        signal = calculate_signal(symbol)

        if signal is not None:
            last_processed_bar[symbol] = signal.bar_time

    log.info("Warm-up complete. Waiting for NEW confirmed signals...")

    try:
        while True:
            if not ensure_connection():
                time.sleep(2.0)
                continue

            # Process all symbols as quickly as practical.
            for symbol in symbols:
                try:
                    process_symbol(symbol)
                except Exception as exc:
                    log.exception(
                        "%s | processing error: %s",
                        symbol,
                        exc,
                    )

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        log.info("Stopping bot...")

    finally:
        shutdown_mt5()
        log.info("MT5 disconnected.")


if __name__ == "__main__":
    main()
