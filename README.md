# QuantAI — Institutional Stock Prediction Platform

An institutional-grade quantitative AI system for stock price prediction, probabilistic forecasting, portfolio optimization, and risk analysis.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   React + Vite Frontend                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │  Market   │ │    AI    │ │   Risk   │ │ Portfolio  │ │
│  │  Panel    │ │  Panel   │ │  Panel   │ │   Panel    │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────┴──────────────────────────────────┐
│                    FastAPI Backend                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │   Data    │ │ Features │ │  Models  │ │ Inference  │ │
│  │  Module   │ │ Engine   │ │  Suite   │ │ Pipeline   │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │Training  │ │Backtesting│ │Portfolio │ │Explainabil.│ │
│  │ Engine   │ │  Engine  │ │Optimizer │ │   Module   │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11+, FastAPI, PyTorch |
| Frontend | React 18, Vite, Plotly.js |
| ML Models | Custom Transformer, FinBERT, DQN |
| Data | yfinance, Google News RSS |
| Portfolio | Mean-Variance, Risk Parity, Black-Litterman |

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- (Optional) CUDA GPU for training

### 1. Setup Environment

```bash
# Clone and setup
cp .env.template .env  # Edit as needed

# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 2. Start Backend

```bash
cd backend
python main.py
# API runs on http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 3. Start Frontend

```bash
cd frontend
npm run dev
# Dashboard at http://localhost:5173
```

### 4. (Optional) Train Model

```bash
# Via API
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "epochs": 50}'

# Or via Dashboard — use the train endpoint
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Stock prediction with AI analysis |
| POST | `/api/forecast_distribution` | Probabilistic forecast |
| POST | `/api/backtest` | Historical backtesting |
| POST | `/api/portfolio` | Portfolio optimization |
| POST | `/api/train` | Model training |
| GET | `/api/metrics` | System metrics |

## Model Architecture

- **Time-Series Transformer**: Multi-head self-attention encoder for price/feature sequences
- **FinBERT Sentiment**: Pre-trained financial NLP model for news sentiment
- **Cross-Attention Fusion**: Combines price and sentiment representations
- **Multi-Task Heads**: Simultaneously predicts return mean, variance, direction probability, and confidence
- **MC Dropout**: Monte Carlo dropout for uncertainty estimation
- **DQN Trading Agent**: Reinforcement learning for trading signal generation

## Docker Deployment

```bash
docker build -t quantai .
docker run -p 8000:8000 --gpus all quantai  # With GPU
docker run -p 8000:8000 quantai              # CPU only
```

## Performance Targets

- Directional accuracy: >60%
- Sharpe ratio: >1.0 (backtested)
- Walk-forward validated (no look-ahead bias)
- Drawdown control via RL reward shaping

## Disclaimer

⚠️ **This system is for research and educational purposes only. It does not constitute financial advice. Past performance does not guarantee future results.**
