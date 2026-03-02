"""
Explainability module.
SHAP feature importance, attention visualization, sentiment contribution,
and risk decomposition.
"""

import numpy as np
import torch
from loguru import logger


class ModelExplainer:
    """
    Provides model explainability via:
    - Feature importance (approximated via gradient-based attribution)
    - Attention weight analysis
    - Sentiment contribution scoring
    - Risk decomposition
    """

    def __init__(self, model, feature_names: list[str] | None = None):
        self.model = model
        self.feature_names = feature_names or []

    def compute_feature_importance(
        self,
        price_seq: torch.Tensor,
        sentiment_features: torch.Tensor,
    ) -> dict:
        """
        Compute feature importance using gradient-based attribution.
        Uses integrated gradients approximation.

        Parameters
        ----------
        price_seq : torch.Tensor
            (1, seq_len, n_features)
        sentiment_features : torch.Tensor
            (1, n_sentiment)

        Returns
        -------
        dict with feature_importance array and names
        """
        self.model.eval()
        price_seq = price_seq.detach().requires_grad_(True)
        sentiment_features = sentiment_features.detach().requires_grad_(True)

        # Forward pass
        outputs = self.model(price_seq, sentiment_features)
        target = outputs["mean"]

        # Backward pass
        target.backward()

        # Price feature importance (averaged over sequence)
        price_grad = price_seq.grad.abs().mean(dim=1).squeeze().cpu().numpy()

        # Sentiment feature importance
        sentiment_grad = sentiment_features.grad.abs().squeeze().cpu().numpy()

        # Normalize
        total = price_grad.sum() + sentiment_grad.sum()
        if total > 0:
            price_grad = price_grad / total
            sentiment_grad = sentiment_grad / total

        # Map to names
        price_names = self.feature_names[:len(price_grad)] if self.feature_names else [
            f"feature_{i}" for i in range(len(price_grad))
        ]
        sentiment_names = [
            "sentiment_polarity", "sentiment_confidence",
            "sentiment_momentum", "sentiment_volatility",
            "bullish_ratio", "n_headlines", "n_bullish", "n_bearish",
        ]

        importance = {}
        for name, val in zip(price_names, price_grad):
            importance[name] = float(val)
        for name, val in zip(sentiment_names, sentiment_grad):
            importance[name] = float(val)

        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "feature_importance": sorted_importance,
            "top_features": list(sorted_importance.keys())[:10],
            "price_total_importance": float(price_grad.sum()),
            "sentiment_total_importance": float(sentiment_grad.sum()),
        }

    def extract_attention_weights(
        self,
        price_seq: torch.Tensor,
        sentiment_features: torch.Tensor,
    ) -> dict:
        """
        Extract attention weights from the transformer encoder.

        Returns
        -------
        dict with attention heatmap data
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                price_seq, sentiment_features, return_attention=True
            )

        attention = outputs.get("attention_weights")
        if attention is None:
            return {"attention_available": False}

        # attention shape: (n_layers, batch, n_heads, seq_len, seq_len)
        attn = attention.cpu().numpy()

        # Average over heads and layers
        avg_attention = attn.mean(axis=(0, 1, 2))  # (seq_len, seq_len)

        # Per-layer attention
        layer_attention = attn.mean(axis=(1, 2))  # (n_layers, seq_len, seq_len)

        # Temporal attention (which positions attend most to which)
        temporal_importance = avg_attention.mean(axis=0)  # importance of each source position

        return {
            "attention_available": True,
            "avg_attention_heatmap": avg_attention.tolist(),
            "layer_attention": layer_attention.tolist(),
            "temporal_importance": temporal_importance.tolist(),
            "n_layers": attn.shape[0],
            "n_heads": attn.shape[2] if len(attn.shape) > 2 else 0,
        }

    def sentiment_contribution(
        self,
        price_seq: torch.Tensor,
        sentiment_features: torch.Tensor,
    ) -> dict:
        """
        Measure how much sentiment contributes to the prediction.
        Compares prediction with and without sentiment.
        """
        self.model.eval()
        with torch.no_grad():
            # Full prediction
            full_output = self.model(price_seq, sentiment_features)
            full_pred = full_output["mean"].item()
            full_direction = full_output["direction_prob"].item()

            # Zero-out sentiment (ablation)
            zero_sentiment = torch.zeros_like(sentiment_features)
            ablated_output = self.model(price_seq, zero_sentiment)
            ablated_pred = ablated_output["mean"].item()
            ablated_direction = ablated_output["direction_prob"].item()

        pred_diff = full_pred - ablated_pred
        direction_diff = full_direction - ablated_direction

        return {
            "full_prediction": full_pred,
            "prediction_without_sentiment": ablated_pred,
            "sentiment_impact_on_return": pred_diff,
            "sentiment_impact_on_direction": direction_diff,
            "sentiment_impact_magnitude": abs(pred_diff),
            "sentiment_is_bullish": pred_diff > 0,
            "sentiment_contribution_pct": abs(pred_diff) / max(abs(full_pred), 1e-10) * 100,
        }

    def risk_decomposition(
        self,
        prediction: dict,
        feature_importance: dict,
    ) -> dict:
        """
        Decompose prediction risk into contributing factors.

        Parameters
        ----------
        prediction : dict
            Output from probabilistic forecaster.
        feature_importance : dict
            Output from compute_feature_importance.
        """
        uncertainty = prediction.get("std_prediction", 0)
        epistemic = prediction.get("epistemic_uncertainty", 0)
        aleatoric = prediction.get("aleatoric_uncertainty", 0)
        direction_prob = prediction.get("direction_prob", 0.5)

        total_importance = sum(feature_importance.get("feature_importance", {}).values())
        sentiment_imp = feature_importance.get("sentiment_total_importance", 0)
        price_imp = feature_importance.get("price_total_importance", 0)

        return {
            "total_uncertainty": uncertainty,
            "model_uncertainty": epistemic,
            "data_uncertainty": aleatoric,
            "uncertainty_ratio": epistemic / max(aleatoric, 1e-10),
            "directional_risk": 1.0 - abs(direction_prob - 0.5) * 2,
            "sentiment_risk_contribution": sentiment_imp / max(total_importance, 1e-10),
            "technical_risk_contribution": price_imp / max(total_importance, 1e-10),
            "risk_level": _classify_risk(uncertainty, direction_prob),
            "recommendation": _generate_recommendation(
                direction_prob, uncertainty, prediction.get("confidence", 0)
            ),
        }


def _classify_risk(uncertainty: float, direction_prob: float) -> str:
    """Classify overall risk level."""
    ambiguity = abs(direction_prob - 0.5)
    if uncertainty > 0.03 or ambiguity < 0.1:
        return "HIGH"
    elif uncertainty > 0.015 or ambiguity < 0.2:
        return "MODERATE"
    else:
        return "LOW"


def _generate_recommendation(
    direction_prob: float, uncertainty: float, confidence: float
) -> str:
    """Generate human-readable trading recommendation."""
    if confidence < 0.4 or uncertainty > 0.04:
        return "CAUTION — High uncertainty. Consider reducing position size."
    elif direction_prob > 0.65 and confidence > 0.6:
        return "BULLISH — Strong upward signal with reasonable confidence."
    elif direction_prob < 0.35 and confidence > 0.6:
        return "BEARISH — Strong downward signal with reasonable confidence."
    elif 0.45 <= direction_prob <= 0.55:
        return "NEUTRAL — No clear directional signal. Consider holding."
    else:
        return "MIXED — Moderate signal. Consider partial position."
