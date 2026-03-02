"""
Inference pipeline — orchestrates data fetching, feature computation,
model prediction, and explainability for end-to-end predictions.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from loguru import logger

from backend.utils.config import DEVICE, MODEL_DIR, SEQUENCE_LENGTH
from backend.data.market_data import fetch_market_data
from backend.data.news_data import fetch_news
from backend.features.technical_indicators import compute_all_technical_indicators
from backend.features.sentiment_features import SentimentAnalyzer, compute_sentiment_features
from backend.features.regime_features import compute_regime_features
from backend.models.fusion_model import StockPredictionModel
from backend.models.probabilistic import mc_dropout_predict, compute_price_forecast
from backend.inference.explainability import ModelExplainer


# Feature columns used by the model (order matters!)
PRICE_FEATURE_COLS = [
    "EMA_20", "EMA_50", "EMA_200",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Width", "BB_Position",
    "ATR_14",
    "Momentum_10", "Momentum_20", "ROC_10",
    "Volatility_20", "Volatility_60", "Vol_Cluster",
    "Sharpe_60",
    "Volume_Ratio",
    "Price_vs_EMA20", "Price_vs_EMA50", "Price_vs_EMA200",
    "Returns", "Log_Returns",
    "Trend_Encoded", "Vol_Regime_Encoded", "Drawdown_State_Encoded",
    "Price_Position_52w", "Drawdown",
]

SENTIMENT_FEATURE_KEYS = [
    "sentiment_polarity", "sentiment_confidence",
    "sentiment_momentum", "sentiment_volatility",
    "bullish_ratio", "n_headlines", "n_bullish", "n_bearish",
]


class InferencePipeline:
    """End-to-end inference pipeline for stock predictions."""

    def __init__(self):
        self._model = None
        self._sentiment_analyzer = None
        self._explainer = None

    def _load_model(self):
        """Load the trained model."""
        if self._model is not None:
            return

        self._model = StockPredictionModel(
            price_input_dim=len(PRICE_FEATURE_COLS),
            sentiment_input_dim=len(SENTIMENT_FEATURE_KEYS),
        ).to(DEVICE)

        model_path = MODEL_DIR / "best_model.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded trained model")
        else:
            logger.warning("No trained model found — using random initialization")

        self._model.eval()
        self._explainer = ModelExplainer(self._model, PRICE_FEATURE_COLS)

    def _get_sentiment_analyzer(self) -> SentimentAnalyzer:
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = SentimentAnalyzer()
        return self._sentiment_analyzer

    def predict(self, ticker: str, include_explainability: bool = True) -> dict:
        """
        Full prediction for a single ticker.

        Returns
        -------
        dict with prediction, forecast, sentiment, explainability, historical data
        """
        self._load_model()

        # 1. Fetch market data
        logger.info(f"[{ticker}] Starting prediction pipeline")
        df = fetch_market_data(ticker)

        # 2. Compute technical indicators
        df = compute_all_technical_indicators(df)

        # 3. Compute regime features
        df = compute_regime_features(df)

        # 4. Fetch and compute sentiment
        news_df = fetch_news(ticker)
        sentiment = compute_sentiment_features(news_df, self._get_sentiment_analyzer())

        # 5. Prepare model input
        df_clean = df.dropna()
        if len(df_clean) < SEQUENCE_LENGTH + 1:
            return {"error": f"Insufficient data: {len(df_clean)} rows after cleaning"}

        # Normalize features
        price_features = df_clean[PRICE_FEATURE_COLS].values
        means = np.nanmean(price_features, axis=0)
        stds = np.nanstd(price_features, axis=0)
        stds[stds == 0] = 1
        price_features_norm = (price_features - means) / stds

        # Latest sequence
        seq = price_features_norm[-SEQUENCE_LENGTH:]
        price_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        # Sentiment tensor
        sentiment_vals = [sentiment.get(k, 0) for k in SENTIMENT_FEATURE_KEYS]
        sentiment_tensor = torch.FloatTensor(sentiment_vals).unsqueeze(0).to(DEVICE)

        # 6. Probabilistic prediction
        mc_result = mc_dropout_predict(self._model, price_tensor, sentiment_tensor)

        # 7. Price forecast
        current_price = float(df_clean["Close"].iloc[-1])
        forecast = compute_price_forecast(current_price, mc_result)

        # 8. Explainability
        explain = {}
        if include_explainability:
            try:
                explain["feature_importance"] = self._explainer.compute_feature_importance(
                    price_tensor, sentiment_tensor
                )
                explain["attention"] = self._explainer.extract_attention_weights(
                    price_tensor, sentiment_tensor
                )
                explain["sentiment_impact"] = self._explainer.sentiment_contribution(
                    price_tensor, sentiment_tensor
                )
                explain["risk"] = self._explainer.risk_decomposition(
                    mc_result, explain["feature_importance"]
                )
            except Exception as e:
                logger.warning(f"Explainability failed: {e}")

        # 9. Build response
        # Get recent OHLCV for charts
        recent_df = df_clean.tail(120)
        historical = {
            "dates": recent_df.index.strftime("%Y-%m-%d").tolist(),
            "open": recent_df["Open"].tolist(),
            "high": recent_df["High"].tolist(),
            "low": recent_df["Low"].tolist(),
            "close": recent_df["Close"].tolist(),
            "volume": recent_df["Volume"].tolist(),
        }

        return {
            "ticker": ticker,
            "current_price": current_price,
            "prediction": {
                "predicted_price": forecast["predicted_price"],
                "predicted_return_pct": forecast["predicted_return_pct"],
                "direction_prob": forecast["direction_prob"],
                "confidence": forecast["confidence"],
                "uncertainty": forecast["uncertainty"],
            },
            "forecast": forecast,
            "sentiment": sentiment,
            "explainability": explain,
            "historical": historical,
            "news_count": len(news_df) if news_df is not None else 0,
        }

    def get_forecast_distribution(self, ticker: str) -> dict:
        """Get detailed probabilistic forecast distribution."""
        self._load_model()

        df = fetch_market_data(ticker)
        df = compute_all_technical_indicators(df)
        df = compute_regime_features(df)
        df_clean = df.dropna()

        price_features = df_clean[PRICE_FEATURE_COLS].values
        means = np.nanmean(price_features, axis=0)
        stds = np.nanstd(price_features, axis=0)
        stds[stds == 0] = 1
        price_features_norm = (price_features - means) / stds

        seq = price_features_norm[-SEQUENCE_LENGTH:]
        price_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        news_df = fetch_news(ticker)
        sentiment = compute_sentiment_features(news_df, self._get_sentiment_analyzer())
        sentiment_vals = [sentiment.get(k, 0) for k in SENTIMENT_FEATURE_KEYS]
        sentiment_tensor = torch.FloatTensor(sentiment_vals).unsqueeze(0).to(DEVICE)

        mc_result = mc_dropout_predict(self._model, price_tensor, sentiment_tensor, n_samples=100)
        current_price = float(df_clean["Close"].iloc[-1])
        forecast = compute_price_forecast(current_price, mc_result)

        # Distribution data for visualization
        price_samples = current_price * (1 + mc_result["all_predictions"])

        return {
            "ticker": ticker,
            "current_price": current_price,
            "forecast": forecast,
            "distribution": {
                "samples": price_samples.tolist(),
                "mean": float(np.mean(price_samples)),
                "std": float(np.std(price_samples)),
                "percentiles": {
                    "p5": float(np.percentile(price_samples, 5)),
                    "p10": float(np.percentile(price_samples, 10)),
                    "p25": float(np.percentile(price_samples, 25)),
                    "p50": float(np.percentile(price_samples, 50)),
                    "p75": float(np.percentile(price_samples, 75)),
                    "p90": float(np.percentile(price_samples, 90)),
                    "p95": float(np.percentile(price_samples, 95)),
                },
            },
            "uncertainty": {
                "epistemic": mc_result["epistemic_uncertainty"],
                "aleatoric": mc_result["aleatoric_uncertainty"],
                "total": mc_result["std_prediction"],
            },
        }


# Singleton pipeline
_pipeline = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline
