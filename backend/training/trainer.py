"""
Training pipeline with walk-forward validation.
Multi-task loss, early stopping, learning rate scheduling, and model checkpointing.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from torch.utils.data import Dataset, DataLoader

from backend.utils.config import (
    DEVICE, MODEL_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS, SEQUENCE_LENGTH,
)
from backend.models.fusion_model import StockPredictionModel, MultiTaskLoss


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    """
    Time-series dataset for the stock prediction model.
    Each sample is a (price_sequence, sentiment_features, target_return, target_direction) tuple.
    """

    def __init__(
        self,
        price_features: np.ndarray,
        sentiment_features: np.ndarray,
        returns: np.ndarray,
        seq_length: int = 60,
    ):
        """
        Parameters
        ----------
        price_features : np.ndarray
            (N, n_features) array of technical features.
        sentiment_features : np.ndarray
            (N, n_sentiment_features) array of sentiment features.
        returns : np.ndarray
            (N,) array of next-day returns.
        seq_length : int
            Number of timesteps per sample.
        """
        self.price_features = torch.FloatTensor(price_features)
        self.sentiment_features = torch.FloatTensor(sentiment_features)
        self.returns = torch.FloatTensor(returns)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.returns) - self.seq_length

    def __getitem__(self, idx):
        end = idx + self.seq_length
        price_seq = self.price_features[idx:end]
        sentiment = self.sentiment_features[end - 1]  # latest sentiment
        target_return = self.returns[end]  # next-day return
        target_direction = (target_return > 0).float()
        return price_seq, sentiment, target_return, target_direction


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Walk-forward model trainer with multi-task loss."""

    def __init__(
        self,
        model: StockPredictionModel,
        learning_rate: float = None,
        weight_decay: float = 1e-5,
        patience: int = 10,
    ):
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate or LEARNING_RATE,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        self.criterion = MultiTaskLoss()
        self.patience = patience
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "direction_acc": []}

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_direction_correct = 0
        total_samples = 0

        for price_seq, sentiment, target_return, target_direction in dataloader:
            price_seq = price_seq.to(DEVICE)
            sentiment = sentiment.to(DEVICE)
            target_return = target_return.to(DEVICE)
            target_direction = target_direction.to(DEVICE)

            self.optimizer.zero_grad()
            predictions = self.model(price_seq, sentiment)
            losses = self.criterion(predictions, target_return, target_direction)
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += losses["total"].item() * len(target_return)
            predicted_dir = (predictions["direction_prob"] > 0.5).float()
            total_direction_correct += (predicted_dir == target_direction).sum().item()
            total_samples += len(target_return)

        avg_loss = total_loss / total_samples
        direction_acc = total_direction_correct / total_samples

        return {"loss": avg_loss, "direction_acc": direction_acc}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate on held-out data."""
        self.model.eval()
        total_loss = 0
        total_direction_correct = 0
        total_samples = 0

        for price_seq, sentiment, target_return, target_direction in dataloader:
            price_seq = price_seq.to(DEVICE)
            sentiment = sentiment.to(DEVICE)
            target_return = target_return.to(DEVICE)
            target_direction = target_direction.to(DEVICE)

            predictions = self.model(price_seq, sentiment)
            losses = self.criterion(predictions, target_return, target_direction)

            total_loss += losses["total"].item() * len(target_return)
            predicted_dir = (predictions["direction_prob"] > 0.5).float()
            total_direction_correct += (predicted_dir == target_direction).sum().item()
            total_samples += len(target_return)

        avg_loss = total_loss / total_samples
        direction_acc = total_direction_correct / total_samples

        return {"loss": avg_loss, "direction_acc": direction_acc}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        save_name: str = "best_model",
    ) -> dict:
        """
        Full training loop with early stopping.
        """
        epochs = epochs or EPOCHS
        logger.info(f"Training for {epochs} epochs (patience={self.patience})")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics["loss"])

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["direction_acc"].append(val_metrics["direction_acc"])

            elapsed = time.time() - t0

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Dir Acc: {val_metrics['direction_acc']:.2%}"
                )

            # Early stopping
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_model(save_name)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best val loss: {self.best_loss:.4f})"
                    )
                    break

        return self.history

    def save_model(self, name: str = "best_model"):
        """Save model checkpoint with metadata."""
        save_path = MODEL_DIR / f"{name}.pt"
        meta_path = MODEL_DIR / f"{name}_meta.json"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
        }, save_path)

        meta = {
            "saved_at": datetime.now().isoformat(),
            "best_val_loss": self.best_loss,
            "epochs_trained": len(self.history["train_loss"]),
            "device": str(DEVICE),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model saved → {save_path}")

    def load_model(self, name: str = "best_model"):
        """Load model checkpoint."""
        load_path = MODEL_DIR / f"{name}.pt"
        if not load_path.exists():
            logger.warning(f"No checkpoint found at {load_path}")
            return False

        checkpoint = torch.load(load_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Model loaded ← {load_path}")
        return True


def prepare_data(
    price_features: np.ndarray,
    sentiment_features: np.ndarray,
    returns: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seq_length: int = None,
    batch_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare train / val / test DataLoaders (time-series split, no shuffling of time order).
    """
    seq_length = seq_length or SEQUENCE_LENGTH
    batch_size = batch_size or BATCH_SIZE
    n = len(returns)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_ds = StockDataset(
        price_features[:train_end],
        sentiment_features[:train_end],
        returns[:train_end],
        seq_length,
    )
    val_ds = StockDataset(
        price_features[train_end:val_end],
        sentiment_features[train_end:val_end],
        returns[train_end:val_end],
        seq_length,
    )
    test_ds = StockDataset(
        price_features[val_end:],
        sentiment_features[val_end:],
        returns[val_end:],
        seq_length,
    )

    # No shuffling — time-series order matters
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        f"Data split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    return train_loader, val_loader, test_loader
