"""
Portfolio optimization module.
Mean-Variance, Risk Parity, and Black-Litterman allocation methods.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


class PortfolioOptimizer:
    """
    Multi-method portfolio optimizer with:
    - Mean-Variance Optimization
    - Risk Parity
    - Black-Litterman
    - Position sizing based on predicted volatility and confidence
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        trading_days: int = 252,
    ):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: float | None = None,
        allow_short: bool = False,
    ) -> dict:
        """
        Mean-Variance Optimization (Markowitz).

        Parameters
        ----------
        expected_returns : array
            Expected annual return for each asset.
        cov_matrix : array
            Covariance matrix of asset returns.
        target_return : float, optional
            Target portfolio return. If None, maximize Sharpe.
        allow_short : bool
            Whether to allow short selling.

        Returns
        -------
        dict with weights, expected_return, volatility, sharpe_ratio
        """
        n = len(expected_returns)

        def neg_sharpe(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / max(port_vol, 1e-10)

        bounds = [(-1 if allow_short else 0, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.dot(w, expected_returns) - target_return,
            })

        x0 = np.ones(n) / n
        result = minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        weights = result.x
        port_return = float(np.dot(weights, expected_returns))
        port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
        sharpe = (port_return - self.risk_free_rate) / max(port_vol, 1e-10)

        return {
            "weights": weights.tolist(),
            "expected_return": port_return,
            "volatility": port_vol,
            "sharpe_ratio": float(sharpe),
            "method": "mean_variance",
        }

    def risk_parity(
        self,
        cov_matrix: np.ndarray,
    ) -> dict:
        """
        Risk Parity allocation — equalizes risk contribution from each asset.
        """
        n = cov_matrix.shape[0]

        def risk_contribution_error(weights):
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / max(port_vol, 1e-10)
            risk_contrib = weights * marginal_contrib
            target_contrib = port_vol / n
            return np.sum((risk_contrib - target_contrib) ** 2)

        bounds = [(0.01, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        x0 = np.ones(n) / n
        result = minimize(
            risk_contribution_error, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        weights = result.x
        port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))

        return {
            "weights": weights.tolist(),
            "volatility": port_vol,
            "method": "risk_parity",
        }

    def black_litterman(
        self,
        market_caps: np.ndarray,
        cov_matrix: np.ndarray,
        views: np.ndarray,
        view_confidences: np.ndarray,
        tau: float = 0.05,
    ) -> dict:
        """
        Black-Litterman model.

        Parameters
        ----------
        market_caps : array
            Market capitalizations for equilibrium weights.
        cov_matrix : array
            Covariance matrix.
        views : array
            Investor views on expected returns.
        view_confidences : array
            Confidence in each view (0 to 1).
        tau : float
            Uncertainty scaling factor.
        """
        n = len(market_caps)

        # Equilibrium weights (proportional to market cap)
        eq_weights = market_caps / market_caps.sum()

        # Implied equilibrium returns
        delta = (self.risk_free_rate + 0.05)  # Risk aversion coefficient
        pi = delta * np.dot(cov_matrix, eq_weights)

        # View matrix P (identity for absolute views)
        P = np.eye(n)

        # View uncertainty
        omega = np.diag((1 - view_confidences) * np.diag(cov_matrix) + 1e-10)

        # Black-Litterman formula
        tau_sigma = tau * cov_matrix
        inv_tau_sigma = np.linalg.inv(tau_sigma + 1e-10 * np.eye(n))
        inv_omega = np.linalg.inv(omega + 1e-10 * np.eye(n))

        bl_returns = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P) @ (
            inv_tau_sigma @ pi + P.T @ inv_omega @ views
        )

        # Optimize with BL returns
        result = self.mean_variance(bl_returns, cov_matrix)
        result["method"] = "black_litterman"
        result["bl_returns"] = bl_returns.tolist()
        result["equilibrium_returns"] = pi.tolist()

        return result

    def position_sizing(
        self,
        weights: np.ndarray,
        predicted_volatilities: np.ndarray,
        confidences: np.ndarray,
        expected_returns: np.ndarray,
        total_capital: float,
        max_position_pct: float = 0.25,
    ) -> dict:
        """
        Adjust position sizes based on model predictions.

        Parameters
        ----------
        weights : array
            Base allocation weights.
        predicted_volatilities : array
            Model-predicted volatility per asset.
        confidences : array
            Model confidence per asset.
        expected_returns : array
            Model-predicted expected returns.
        total_capital : float
            Total portfolio value.
        max_position_pct : float
            Maximum single position as fraction of portfolio.
        """
        # Inverse volatility scaling
        inv_vol = 1.0 / np.maximum(predicted_volatilities, 1e-6)
        vol_adjusted = weights * inv_vol
        vol_adjusted /= vol_adjusted.sum()

        # Confidence-weighted
        conf_adjusted = vol_adjusted * confidences
        conf_adjusted /= conf_adjusted.sum()

        # Cap individual positions
        capped = np.minimum(conf_adjusted, max_position_pct)
        capped /= capped.sum()

        # Dollar amounts
        dollar_amounts = capped * total_capital

        # Risk exposure
        risk_contrib = capped * predicted_volatilities
        total_risk = risk_contrib.sum()
        risk_pct = risk_contrib / max(total_risk, 1e-10)

        # Diversification score (Herfindahl index inverse)
        hhi = np.sum(capped ** 2)
        diversification = 1 / max(hhi * len(capped), 1e-10)

        return {
            "adjusted_weights": capped.tolist(),
            "dollar_amounts": dollar_amounts.tolist(),
            "risk_contribution": risk_pct.tolist(),
            "diversification_score": float(min(diversification, 1.0)),
            "total_risk_exposure": float(total_risk),
        }
