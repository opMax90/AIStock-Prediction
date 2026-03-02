/**
 * API client for the QuantAI backend.
 */

import axios from 'axios';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || '/api',
    timeout: 120000, // 2 minutes for heavy operations
    headers: {
        'Content-Type': 'application/json',
    },
});

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        const message = error.response?.data?.detail || error.message || 'Unknown error';
        console.error('API Error:', message);
        return Promise.reject(new Error(message));
    }
);

export const predictStock = async (ticker, includeExplainability = true) => {
    const { data } = await api.post('/predict', {
        ticker: ticker.toUpperCase(),
        include_explainability: includeExplainability,
    });
    return data;
};

export const getForecastDistribution = async (ticker) => {
    const { data } = await api.post('/forecast_distribution', {
        ticker: ticker.toUpperCase(),
    });
    return data;
};

export const runBacktest = async (ticker, config = {}) => {
    const { data } = await api.post('/backtest', {
        ticker: ticker.toUpperCase(),
        initial_capital: config.initialCapital || 100000,
        transaction_cost: config.transactionCost || 0.001,
        slippage: config.slippage || 0.0005,
        direction_threshold: config.directionThreshold || 0.5,
    });
    return data;
};

export const optimizePortfolio = async (tickers, method = 'mean_variance', totalCapital = 100000) => {
    const { data } = await api.post('/portfolio', {
        tickers: tickers.map(t => t.toUpperCase()),
        method,
        total_capital: totalCapital,
    });
    return data;
};

export const trainModel = async (ticker, epochs = 50) => {
    const { data } = await api.post('/train', {
        ticker: ticker.toUpperCase(),
        epochs,
    });
    return data;
};

export const getMetrics = async () => {
    const { data } = await api.get('/metrics');
    return data;
};

export default api;
