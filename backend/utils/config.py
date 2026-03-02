"""
Central configuration module.
Loads settings from environment variables with sensible defaults.
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load .env file if it exists
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()  # try default locations

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
_device_env = os.getenv("DEVICE", "auto")
if _device_env == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(_device_env)

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH: int = int(os.getenv("SEQUENCE_LENGTH", "60"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.0001"))
EPOCHS: int = int(os.getenv("EPOCHS", "100"))
MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "models" / "saved")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Transformer config
D_MODEL: int = 128
N_HEADS: int = 8
N_ENCODER_LAYERS: int = 4
D_FF: int = 512
DROPOUT: float = 0.1
MC_DROPOUT_SAMPLES: int = 50

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_CACHE_DIR: Path = Path(
    os.getenv("DATA_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "data" / "cache"))
)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKERS: list[str] = os.getenv(
    "DEFAULT_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA"
).split(",")
HISTORY_YEARS: int = int(os.getenv("HISTORY_YEARS", "20"))

# ---------------------------------------------------------------------------
# Sentiment / FinBERT
# ---------------------------------------------------------------------------
FINBERT_MODEL: str = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
MAX_NEWS_HEADLINES: int = int(os.getenv("MAX_NEWS_HEADLINES", "50"))

# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "100000"))
TRANSACTION_COST: float = float(os.getenv("TRANSACTION_COST", "0.001"))
SLIPPAGE: float = float(os.getenv("SLIPPAGE", "0.0005"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
logger.remove()
logger.add(
    "logs/app_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
)
logger.add(
    lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    format="{time:HH:mm:ss} | {level:<8} | {message}",
)

logger.info(f"Device: {DEVICE}")
logger.info(f"Model dir: {MODEL_DIR}")
