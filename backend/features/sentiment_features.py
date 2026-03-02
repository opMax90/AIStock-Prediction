"""
Sentiment feature extraction using FinBERT.
Produces polarity scores, weighted confidence, momentum, and volatility.
"""

import torch
import numpy as np
import pandas as pd
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.utils.config import FINBERT_MODEL, DEVICE


class SentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial text."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or FINBERT_MODEL
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load FinBERT model."""
        if self._loaded:
            return
        logger.info(f"Loading FinBERT model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(DEVICE)
        self._model.eval()
        self._loaded = True
        logger.info("FinBERT loaded successfully")

    @torch.no_grad()
    def score_headlines(self, headlines: list[str]) -> pd.DataFrame:
        """
        Score a list of headlines for sentiment.

        Returns
        -------
        pd.DataFrame
            Columns: headline, positive, negative, neutral, polarity, confidence
        """
        self._load_model()

        if not headlines:
            return pd.DataFrame(
                columns=["headline", "positive", "negative", "neutral", "polarity", "confidence"]
            )

        results = []
        batch_size = 16

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(DEVICE)

            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for j, headline in enumerate(batch):
                pos, neg, neu = probs[j][0], probs[j][1], probs[j][2]
                # Polarity: +1 (bullish) to -1 (bearish)
                polarity = float(pos - neg)
                # Confidence: how decisive the model is
                confidence = float(max(pos, neg, neu))

                results.append({
                    "headline": headline,
                    "positive": float(pos),
                    "negative": float(neg),
                    "neutral": float(neu),
                    "polarity": polarity,
                    "confidence": confidence,
                })

        return pd.DataFrame(results)


def compute_sentiment_features(
    news_df: pd.DataFrame,
    analyzer: SentimentAnalyzer | None = None,
) -> dict[str, float]:
    """
    Compute aggregated sentiment features from a DataFrame of news.

    Parameters
    ----------
    news_df : pd.DataFrame
        Must have a 'title' column with headlines.
    analyzer : SentimentAnalyzer, optional
        Pre-initialized analyzer. Creates one if not provided.

    Returns
    -------
    dict
        Sentiment features: polarity, confidence, momentum, volatility, etc.
    """
    if news_df.empty or "title" not in news_df.columns:
        return _default_sentiment_features()

    analyzer = analyzer or SentimentAnalyzer()
    headlines = news_df["title"].tolist()
    scores = analyzer.score_headlines(headlines)

    if scores.empty:
        return _default_sentiment_features()

    polarity = scores["polarity"].values
    confidence = scores["confidence"].values

    # Weighted polarity (confidence-weighted average)
    weights = confidence / confidence.sum() if confidence.sum() > 0 else np.ones_like(confidence) / len(confidence)
    weighted_polarity = float(np.sum(polarity * weights))

    # Sentiment momentum: trend in polarity across headlines (most recent first)
    if len(polarity) >= 3:
        recent = polarity[: len(polarity) // 3].mean()
        older = polarity[len(polarity) // 3 :].mean()
        sentiment_momentum = float(recent - older)
    else:
        sentiment_momentum = 0.0

    # Sentiment volatility: how dispersed the opinions are
    sentiment_volatility = float(np.std(polarity)) if len(polarity) > 1 else 0.0

    # Aggregate confidence
    avg_confidence = float(np.mean(confidence))

    # Bullish / bearish ratio
    n_bullish = int(np.sum(polarity > 0.1))
    n_bearish = int(np.sum(polarity < -0.1))
    total = len(polarity)
    bullish_ratio = n_bullish / total if total > 0 else 0.5

    return {
        "sentiment_polarity": weighted_polarity,
        "sentiment_confidence": avg_confidence,
        "sentiment_momentum": sentiment_momentum,
        "sentiment_volatility": sentiment_volatility,
        "bullish_ratio": bullish_ratio,
        "n_headlines": total,
        "n_bullish": n_bullish,
        "n_bearish": n_bearish,
    }


def _default_sentiment_features() -> dict[str, float]:
    """Return neutral sentiment when no data is available."""
    return {
        "sentiment_polarity": 0.0,
        "sentiment_confidence": 0.0,
        "sentiment_momentum": 0.0,
        "sentiment_volatility": 0.0,
        "bullish_ratio": 0.5,
        "n_headlines": 0,
        "n_bullish": 0,
        "n_bearish": 0,
    }
