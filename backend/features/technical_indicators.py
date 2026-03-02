"""
Technical indicator calculations.
All indicators are computed without future data leakage.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Original data with new technical indicator columns appended.
    """
    df = df.copy()

    # Exponential Moving Averages
    for period in [20, 50, 200]:
        df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

    # RSI (14)
    df["RSI_14"] = _rsi(df["Close"], 14)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # ATR (14)
    df["ATR_14"] = _atr(df, 14)

    # Momentum
    df["Momentum_10"] = df["Close"].pct_change(10)
    df["Momentum_20"] = df["Close"].pct_change(20)

    # Rate of Change
    df["ROC_10"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)

    # Volatility (rolling std of returns)
    returns = df["Close"].pct_change()
    df["Volatility_20"] = returns.rolling(20).std()
    df["Volatility_60"] = returns.rolling(60).std()

    # Volatility clustering (GARCH-like proxy)
    df["Vol_Cluster"] = (returns ** 2).rolling(20).mean()

    # Rolling Sharpe Ratio (annualized, 60-day window)
    rolling_mean = returns.rolling(60).mean()
    rolling_std = returns.rolling(60).std()
    df["Sharpe_60"] = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

    # Volume features
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"].replace(0, np.nan)

    # Price relative to moving averages
    df["Price_vs_EMA20"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"]
    df["Price_vs_EMA50"] = (df["Close"] - df["EMA_50"]) / df["EMA_50"]
    df["Price_vs_EMA200"] = (df["Close"] - df["EMA_200"]) / df["EMA_200"]

    # Daily returns
    df["Returns"] = returns
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

    logger.info(f"Computed {len([c for c in df.columns if c not in ['Open','High','Low','Close','Volume']])} technical indicators")
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()
