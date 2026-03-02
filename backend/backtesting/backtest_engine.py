"""
Backtesting engine with rolling walk-forward validation.
Transaction costs, slippage, and comprehensive performance metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100_000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    risk_free_rate: float = 0.04  # Annual


@dataclass
class BacktestResult:
    """Complete backtesting results."""
    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)
    # Trade log
    trades: list = field(default_factory=list)
    # Performance metrics
    metrics: dict = field(default_factory=dict)
    # Daily returns
    daily_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    """
    Full historical backtesting with:
    - Walk-forward validation
    - Transaction cost & slippage modeling
    - Comprehensive metrics suite
    - Benchmark comparison
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        signals: pd.Series,
        prices: pd.Series,
        benchmark_prices: pd.Series | None = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Parameters
        ----------
        signals : pd.Series
            Trading signals: 1 (buy/hold), -1 (sell/short), 0 (flat).
            Index must align with prices.
        prices : pd.Series
            Close prices to trade on.
        benchmark_prices : pd.Series, optional
            Benchmark prices for comparison (buy-and-hold).

        Returns
        -------
        BacktestResult
        """
        # Align data
        common_idx = signals.index.intersection(prices.index)
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]

        capital = self.config.initial_capital
        position = 0  # Number of shares
        equity = []
        trades = []
        prev_signal = 0

        for i, (date, signal) in enumerate(signals.items()):
            price = prices.loc[date]
            signal = int(signal)

            # Execute trades
            if signal != prev_signal:
                # Close existing position
                if position != 0:
                    sell_price = price * (1 - self.config.slippage)
                    proceeds = position * sell_price
                    cost = abs(proceeds) * self.config.transaction_cost
                    capital += proceeds - cost
                    trades.append({
                        "date": date,
                        "action": "SELL",
                        "price": sell_price,
                        "shares": position,
                        "cost": cost,
                        "capital": capital,
                    })
                    position = 0

                # Open new position
                if signal == 1:
                    buy_price = price * (1 + self.config.slippage)
                    shares = int(capital / buy_price)
                    if shares > 0:
                        cost = shares * buy_price * self.config.transaction_cost
                        capital -= shares * buy_price + cost
                        position = shares
                        trades.append({
                            "date": date,
                            "action": "BUY",
                            "price": buy_price,
                            "shares": shares,
                            "cost": cost,
                            "capital": capital,
                        })

            prev_signal = signal

            # Track portfolio value
            portfolio_value = capital + position * price
            equity.append({"date": date, "value": portfolio_value})

        # Build equity curve
        equity_df = pd.DataFrame(equity).set_index("date")
        equity_curve = equity_df["value"]

        # Benchmark (buy-and-hold)
        if benchmark_prices is not None:
            bench = benchmark_prices.loc[common_idx]
        else:
            bench = prices
        benchmark_curve = (bench / bench.iloc[0]) * self.config.initial_capital

        # Compute metrics
        daily_returns = equity_curve.pct_change().dropna()
        bench_returns = benchmark_curve.pct_change().dropna()
        metrics = self._compute_metrics(
            equity_curve, daily_returns, bench_returns
        )

        result = BacktestResult(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            trades=trades,
            metrics=metrics,
            daily_returns=daily_returns,
            benchmark_returns=bench_returns,
        )

        logger.info(f"Backtest complete: {len(trades)} trades, Sharpe={metrics.get('sharpe_ratio', 0):.2f}")
        return result

    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        bench_returns: pd.Series,
    ) -> dict:
        """Compute comprehensive performance metrics."""
        total_days = len(daily_returns)
        if total_days < 2:
            return {}

        trading_days = 252
        rf_daily = self.config.risk_free_rate / trading_days

        # Basic returns
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        years = total_days / trading_days

        # CAGR
        cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

        # Sharpe Ratio
        excess_returns = daily_returns - rf_daily
        sharpe = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)
            if excess_returns.std() > 0 else 0
        )

        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-10
        sortino = excess_returns.mean() / downside_std * np.sqrt(trading_days)

        # Maximum Drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (daily_returns > 0).sum()
        total_trading_days = (daily_returns != 0).sum()
        win_rate = winning_days / max(total_trading_days, 1)

        # Profit factor
        gross_profit = daily_returns[daily_returns > 0].sum()
        gross_loss = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = gross_profit / max(gross_loss, 1e-10)

        # Alpha and Beta
        if len(bench_returns) > 0 and bench_returns.std() > 0:
            cov = np.cov(daily_returns.values[-len(bench_returns):], bench_returns.values)
            beta = cov[0, 1] / max(cov[1, 1], 1e-10)
            bench_annual_return = bench_returns.mean() * trading_days
            strategy_annual_return = daily_returns.mean() * trading_days
            alpha = strategy_annual_return - (
                self.config.risk_free_rate + beta * (bench_annual_return - self.config.risk_free_rate)
            )
        else:
            alpha, beta = 0.0, 1.0

        # Information ratio
        tracking_error = (daily_returns.values[-len(bench_returns):] - bench_returns.values)
        info_ratio = (
            tracking_error.mean() / max(tracking_error.std(), 1e-10) * np.sqrt(trading_days)
        )

        # Volatility
        annual_vol = daily_returns.std() * np.sqrt(trading_days)

        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "alpha": float(alpha),
            "beta": float(beta),
            "information_ratio": float(info_ratio),
            "annual_volatility": float(annual_vol),
            "calmar_ratio": float(calmar),
            "total_trades": total_trading_days,
            "total_days": total_days,
        }


def walk_forward_backtest(
    signals_fn,
    prices: pd.Series,
    train_window: int = 756,  # ~3 years
    test_window: int = 252,   # ~1 year
    step: int = 63,           # ~quarterly rebalance
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Walk-forward backtesting.

    Parameters
    ----------
    signals_fn : callable
        Function(train_prices) -> signals for test period.
    prices : pd.Series
        Full price series.
    train_window : int
        Training window size (days).
    test_window : int
        Test window size (days).
    step : int
        Step size for rolling.
    config : BacktestConfig

    Returns
    -------
    BacktestResult
    """
    engine = BacktestEngine(config)
    all_signals = pd.Series(dtype=float)

    n = len(prices)
    for start in range(0, n - train_window - test_window + 1, step):
        train_end = start + train_window
        test_end = min(train_end + test_window, n)

        train_prices = prices.iloc[start:train_end]
        test_prices = prices.iloc[train_end:test_end]

        try:
            test_signals = signals_fn(train_prices)
            # Align signals with test period
            common = test_signals.index.intersection(test_prices.index)
            all_signals = pd.concat([all_signals, test_signals.loc[common]])
        except Exception as e:
            logger.warning(f"Walk-forward step failed at {start}: {e}")
            continue

    if len(all_signals) == 0:
        logger.error("Walk-forward produced no signals")
        return BacktestResult()

    return engine.run(all_signals, prices)
