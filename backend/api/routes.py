"""
FastAPI routes for the stock prediction API.
"""

import traceback
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from backend.inference.pipeline import get_pipeline, PRICE_FEATURE_COLS, SENTIMENT_FEATURE_KEYS
from backend.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from backend.portfolio.portfolio_optimizer import PortfolioOptimizer
from backend.data.market_data import fetch_market_data
from backend.features.technical_indicators import compute_all_technical_indicators
from backend.features.regime_features import compute_regime_features


router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol", examples=["AAPL"])
    include_explainability: bool = Field(True, description="Include SHAP / attention analysis")


class ForecastRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])


class BacktestRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    initial_capital: float = Field(100_000, ge=1000)
    transaction_cost: float = Field(0.001, ge=0, le=0.05)
    slippage: float = Field(0.0005, ge=0, le=0.01)
    direction_threshold: float = Field(0.5, ge=0.3, le=0.8)


class PortfolioRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, examples=[["AAPL", "MSFT", "GOOGL"]])
    method: str = Field("mean_variance", description="Optimization method")
    total_capital: float = Field(100_000, ge=1000)


class TrainRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    epochs: int = Field(50, ge=1, le=500)
    batch_size: int = Field(32, ge=8, le=256)


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(req: PredictRequest):
    """Generate prediction for a stock ticker."""
    try:
        pipeline = get_pipeline()
        result = pipeline.predict(req.ticker, req.include_explainability)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast_distribution")
async def forecast_distribution(req: ForecastRequest):
    """Get probabilistic forecast distribution."""
    try:
        pipeline = get_pipeline()
        result = pipeline.get_forecast_distribution(req.ticker)
        return result
    except Exception as e:
        logger.error(f"Forecast failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def backtest(req: BacktestRequest):
    """Run historical backtest."""
    try:
        # Fetch data
        df = fetch_market_data(req.ticker)
        df = compute_all_technical_indicators(df)
        df = compute_regime_features(df)
        df = df.dropna()

        # Generate simple signals from RSI + MACD
        signals = pd.Series(0, index=df.index)
        bullish = (df["RSI_14"] < 70) & (df["MACD_Hist"] > 0) & (df["Trend_Encoded"] >= 0)
        bearish = (df["RSI_14"] > 30) & (df["MACD_Hist"] < 0) & (df["Trend_Encoded"] <= 0)
        signals[bullish] = 1
        signals[bearish] = 0

        # If a model is trained, use model predictions instead
        try:
            pipeline = get_pipeline()
            pipeline._load_model()
            # Use model direction probability as signal
            # (simplified for backtest — in production use rolling predictions)
        except Exception:
            pass  # Fallback to technical signals

        config = BacktestConfig(
            initial_capital=req.initial_capital,
            transaction_cost=req.transaction_cost,
            slippage=req.slippage,
        )
        engine = BacktestEngine(config)
        result = engine.run(signals, df["Close"])

        return {
            "ticker": req.ticker,
            "metrics": result.metrics,
            "equity_curve": {
                "dates": result.equity_curve.index.strftime("%Y-%m-%d").tolist(),
                "values": result.equity_curve.tolist(),
            },
            "benchmark_curve": {
                "dates": result.benchmark_curve.index.strftime("%Y-%m-%d").tolist(),
                "values": result.benchmark_curve.tolist(),
            },
            "total_trades": len(result.trades),
            "trades": result.trades[:50],  # Limit response size
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio")
async def portfolio(req: PortfolioRequest):
    """Portfolio optimization."""
    try:
        optimizer = PortfolioOptimizer()
        tickers = req.tickers

        # Fetch returns for all tickers
        returns_data = {}
        for ticker in tickers:
            df = fetch_market_data(ticker, years=5)
            returns_data[ticker] = df["Close"].pct_change().dropna()

        # Align dates
        returns_df = pd.DataFrame(returns_data).dropna()
        if len(returns_df) < 60:
            raise HTTPException(status_code=400, detail="Insufficient shared data across tickers")

        expected_returns = returns_df.mean().values * 252  # Annualize
        cov_matrix = returns_df.cov().values * 252

        # Run optimization
        if req.method == "risk_parity":
            result = optimizer.risk_parity(cov_matrix)
            result["expected_return"] = float(np.dot(result["weights"], expected_returns))
        elif req.method == "mean_variance":
            result = optimizer.mean_variance(expected_returns, cov_matrix)
        else:
            result = optimizer.mean_variance(expected_returns, cov_matrix)

        # Position sizing
        vol_per_asset = np.sqrt(np.diag(cov_matrix))
        confidences = np.ones(len(tickers)) * 0.7  # Default confidence
        sizing = optimizer.position_sizing(
            np.array(result["weights"]),
            vol_per_asset,
            confidences,
            expected_returns,
            req.total_capital,
        )

        return {
            "tickers": tickers,
            "optimization": result,
            "position_sizing": sizing,
            "individual_stats": {
                ticker: {
                    "expected_return": float(expected_returns[i]),
                    "volatility": float(vol_per_asset[i]),
                    "weight": float(result["weights"][i]),
                }
                for i, ticker in enumerate(tickers)
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train(req: TrainRequest):
    """Trigger model training (simplified)."""
    try:
        from backend.training.trainer import Trainer, prepare_data, StockDataset
        from backend.models.fusion_model import StockPredictionModel
        from backend.utils.config import DEVICE
        from torch.utils.data import DataLoader

        # Fetch and prepare data
        df = fetch_market_data(req.ticker, years=15)
        df = compute_all_technical_indicators(df)
        df = compute_regime_features(df)
        df = df.dropna()

        if len(df) < 500:
            raise HTTPException(status_code=400, detail="Insufficient data for training")

        # Prepare features
        price_features = df[PRICE_FEATURE_COLS].values
        means = np.nanmean(price_features, axis=0)
        stds = np.nanstd(price_features, axis=0)
        stds[stds == 0] = 1
        price_features_norm = (price_features - means) / stds

        returns = df["Returns"].values
        sentiment_features = np.zeros((len(df), len(SENTIMENT_FEATURE_KEYS)))

        # Prepare data loaders
        train_loader, val_loader, _ = prepare_data(
            price_features_norm, sentiment_features, returns,
            batch_size=req.batch_size,
        )

        # Initialize model
        model = StockPredictionModel(
            price_input_dim=len(PRICE_FEATURE_COLS),
            sentiment_input_dim=len(SENTIMENT_FEATURE_KEYS),
        )
        trainer = Trainer(model)

        # Train
        history = trainer.train(train_loader, val_loader, epochs=req.epochs)

        return {
            "status": "completed",
            "ticker": req.ticker,
            "epochs_trained": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_direction_acc": max(history["direction_acc"]) if history["direction_acc"] else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def metrics():
    """Get system metrics and model performance summary."""
    from backend.utils.config import MODEL_DIR, DEVICE
    import json

    meta_path = MODEL_DIR / "best_model_meta.json"
    model_info = {}
    if meta_path.exists():
        with open(meta_path) as f:
            model_info = json.load(f)

    return {
        "system": {
            "device": str(DEVICE),
            "model_dir": str(MODEL_DIR),
            "model_loaded": meta_path.exists(),
        },
        "model": model_info,
    }
