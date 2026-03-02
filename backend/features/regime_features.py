"""
Market regime detection features.
Classifies trend, volatility regime, and drawdown state.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market regime features from OHLCV + technical data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least: Close, EMA_50, EMA_200, Volatility_20.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with regime columns appended.
    """
    df = df.copy()

    # --- Trend Classification ---
    df["Trend"] = _classify_trend(df)

    # --- Volatility Regime ---
    df["Vol_Regime"] = _classify_volatility_regime(df)

    # --- Drawdown State ---
    df["Drawdown"] = _compute_drawdown(df["Close"])
    df["Drawdown_State"] = _classify_drawdown_state(df["Drawdown"])

    # --- Regime Numeric Encoding (for model input) ---
    trend_map = {"strong_bull": 2, "bull": 1, "sideways": 0, "bear": -1, "strong_bear": -2}
    df["Trend_Encoded"] = df["Trend"].map(trend_map).fillna(0).astype(float)

    vol_map = {"low": 0, "normal": 1, "high": 2, "extreme": 3}
    df["Vol_Regime_Encoded"] = df["Vol_Regime"].map(vol_map).fillna(1).astype(float)

    dd_map = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
    df["Drawdown_State_Encoded"] = df["Drawdown_State"].map(dd_map).fillna(0).astype(float)

    # --- Market Breadth Proxy (price relative to range) ---
    high_252 = df["Close"].rolling(252).max()
    low_252 = df["Close"].rolling(252).min()
    df["Price_Position_52w"] = (df["Close"] - low_252) / (high_252 - low_252).replace(0, np.nan)

    logger.info("Computed regime features: Trend, Vol_Regime, Drawdown_State")
    return df


def _classify_trend(df: pd.DataFrame) -> pd.Series:
    """Classify market trend based on EMA crossovers and slope."""
    trend = pd.Series("sideways", index=df.index)

    if "EMA_50" not in df.columns or "EMA_200" not in df.columns:
        return trend

    ema50 = df["EMA_50"]
    ema200 = df["EMA_200"]
    close = df["Close"]

    # EMA slope (20-day rate of change)
    ema50_slope = ema50.pct_change(20)

    # Golden cross / Death cross
    above_200 = close > ema200
    above_50 = close > ema50
    ema50_above_200 = ema50 > ema200

    # Strong Bull: price > EMA50 > EMA200, positive slope
    strong_bull = above_200 & above_50 & ema50_above_200 & (ema50_slope > 0.02)
    trend[strong_bull] = "strong_bull"

    # Bull: price > EMA50, EMA50 > EMA200
    bull = above_50 & ema50_above_200 & ~strong_bull
    trend[bull] = "bull"

    # Strong Bear: price < EMA50 < EMA200, negative slope
    strong_bear = ~above_200 & ~above_50 & ~ema50_above_200 & (ema50_slope < -0.02)
    trend[strong_bear] = "strong_bear"

    # Bear: price < EMA50, EMA50 < EMA200
    bear = ~above_50 & ~ema50_above_200 & ~strong_bear
    trend[bear] = "bear"

    return trend


def _classify_volatility_regime(df: pd.DataFrame) -> pd.Series:
    """Classify volatility regime based on rolling volatility percentiles."""
    regime = pd.Series("normal", index=df.index)

    if "Volatility_20" not in df.columns:
        return regime

    vol = df["Volatility_20"]
    vol_percentile = vol.rolling(252).rank(pct=True)

    regime[vol_percentile <= 0.25] = "low"
    regime[(vol_percentile > 0.25) & (vol_percentile <= 0.75)] = "normal"
    regime[(vol_percentile > 0.75) & (vol_percentile <= 0.95)] = "high"
    regime[vol_percentile > 0.95] = "extreme"

    return regime


def _compute_drawdown(close: pd.Series) -> pd.Series:
    """Compute drawdown from peak."""
    peak = close.cummax()
    drawdown = (close - peak) / peak
    return drawdown


def _classify_drawdown_state(drawdown: pd.Series) -> pd.Series:
    """Classify drawdown severity."""
    state = pd.Series("none", index=drawdown.index)
    state[drawdown < -0.05] = "mild"
    state[drawdown < -0.10] = "moderate"
    state[drawdown < -0.20] = "severe"
    return state
