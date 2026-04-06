# QuantAI System Methodology

This document outlines the theoretical and mathematical foundations of the QuantAI prediction system. This methodology relies on a fusion of Time-Series analysis, Natural Language Processing (NLP), and Portfolio Theory to achieve institutional-grade stock forecasting.

## 1. Data Engineering and Feature Extraction

The model relies on multi-modal data streams encompassing both numerical pricing data and unstructured text data.

### 1.1 Technical Features
For a given asset price series $P_t$, we extract non-stationary attributes into a stationary domain where possible, and compute normalized features:
*   **Returns**: $R_t = \ln(P_t / P_{t-1})$
*   **Moving Averages (SMA / EMA)**: Used for trend identification.
*   **Momentum Indicators**: Relative Strength Index (RSI), MACD.
*   **Volatility Indicators**: Bollinger Bands width, Average True Range (ATR).

### 1.2 NLP Sentiment Features
We utilize a pre-trained **FinBERT** (Financial Bidirectional Encoder Representations from Transformers) model to process financial news headlines. 
For each headline $H_i$ on day $t$:
$$ S(H_i) = \text{Softmax}(\text{FinBERT}(H_i)) \in [0, 1]^3 $$
where the classes are $\{Negative, Neutral, Positive\}$. We aggregate the expected values per day to form a unified Sentiment Polarity score $Sentiment_t$.

---

## 2. Model Architecture: Cross-Attention Fusion Transformer

The core predictive engine is a deep neural network that combines sequential numeric data with sentiment data.

### 2.1 Time-Series Encoder
We use a Multi-Head Self-Attention (MHSA) Transformer architecture to encode the temporal dynamics of the pricing features over a lookback window $W$ (e.g., 60 days).
$$ Z_{price} = \text{MHSA}(\text{LayerNorm}(X_{price})) + X_{price} $$

### 2.2 Cross-Modal Fusion
The outputs from the Time-Series Encoder and the Sentiment Encoder are fused. We use Cross-Attention where the price embeddings act as the Queries ($Q$), and the sentiment embeddings act as the Keys ($K$) and Values ($V$).
$$ Z_{fusion} = \text{Attention}(Q=Z_{price}, K=Sentiment_t, V=Sentiment_t) $$

### 2.3 Multi-Task Output Heads
The network splits into multiple independent Multi-Layer Perceptrons (MLPs) to predict distinct targets:
1.  **Expected Return**: $\hat{R}_{t+1}$ (Regression)
2.  **Directional Probability**: $P(\hat{R}_{t+1} > 0)$ (Classification via Sigmoid)

---

## 3. Uncertainty Estimation (Epistemic & Aleatoric)

Point predictions are insufficient for risk-conscious trading. We must quantify the model's confidence.

### 3.1 Monte Carlo Dropout
To approximate Bayesian inference, we apply **Monte Carlo (MC) Dropout** during inference. 
We perform $N$ stochastic forward passes (e.g., $N=50$) with dropout enabled (dropout rate $p=0.1$). 
Let $\hat{y}_n$ be the outcome of the $n$-th pass:
*   **Expected Prediction**: $\mu = \frac{1}{N} \sum_{n=1}^N \hat{y}_n$
*   **Epistemic Uncertainty (Variance)**: $\sigma^2 = \frac{1}{N} \sum_{n=1}^N (\hat{y}_n - \mu)^2$

We use $\mu \pm z(\alpha/2) \cdot \sigma$ to compute the Confidence Intervals (e.g., 80% and 95% levels) shown in the Market Panel.

---

## 4. Portfolio Optimization

Armed with expected returns ($\vec{\mu}$) and a covariance matrix ($\Sigma$) of asset returns, we construct the optimal allocation vector $\vec{w}$.

### 4.1 Mean-Variance Optimization (Markowitz)
To maximize the Sharpe Ratio, we solve:
$$ \max_{\vec{w}} \frac{\vec{w}^T \vec{\mu} - R_f}{\sqrt{\vec{w}^T \Sigma \vec{w}}} $$
subject to $\sum w_i = 1$ and $w_i \geq 0 \forall i$ (long-only constraint).

### 4.2 Dynamic Sizing via Model Confidence
The base Markowitz weights $\vec{w}$ are iteratively adjusted using the inverse of the model's uncertainty $\sigma^2$ (from MC Dropout). High-uncertainty predictions constrain the maximum allowed weight for an asset, reducing catastrophic drawdowns.

---

## 5. System Evaluation

The model is evaluated out-of-sample using a walk-forward backtesting approach. Primary metrics include:
*   **Directional Accuracy (DA)**: The percentage of correct sign predictions.
*   **Root Mean Squared Error (RMSE)**: Error in absolute price forecasting.
*   **Information Ratio / Sharpe Ratio**: Evaluating the return relative to the risk taken by the trading signals.

*Disclaimer: This methodology is adapted for academic grading and represents a scalable prototype rather than a production-ready trading algorithm.*
