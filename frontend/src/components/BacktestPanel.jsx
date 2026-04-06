import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { runBacktest } from '../services/api';

const CHART_LAYOUT = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#0d1525',
    font: { family: 'Inter', color: '#94a3b8', size: 10 },
    margin: { t: 30, r: 20, b: 30, l: 40 },
    xaxis: { gridcolor: '#1a2332', linecolor: '#1e293b' },
    yaxis: { gridcolor: '#1a2332', linecolor: '#1e293b', tickprefix: '$' },
    legend: { x: 0, y: 1.1, orientation: 'h', font: { size: 10, color: '#64748b' } },
};

export default function BacktestPanel({ defaultTicker = 'AAPL' }) {
    const [ticker, setTicker] = useState(defaultTicker);
    const [capital, setCapital] = useState(100000);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleBacktest = async () => {
        const t = ticker.trim().toUpperCase();
        if (!t) return;

        setLoading(true);
        setError(null);
        try {
            const data = await runBacktest(t, { initialCapital: capital });
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const metrics = result?.metrics;

    return (
        <div className="card">
            <div className="card-header">
                <div className="card-title">
                    <span className="card-title-icon" style={{ background: '#ec4899' }}></span>
                    Strategy Backtest
                </div>
            </div>
            <div className="card-body">
                {/* Input Controls */}
                <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
                    <input
                        className="ticker-input"
                        style={{ width: 100, textTransform: 'uppercase' }}
                        value={ticker}
                        onChange={e => setTicker(e.target.value)}
                        placeholder="Ticker"
                    />
                    <input
                        className="ticker-input"
                        style={{ width: 120, textTransform: 'none' }}
                        type="number"
                        value={capital}
                        onChange={e => setCapital(Number(e.target.value))}
                        placeholder="Capital ($)"
                    />
                    <button
                        className="btn btn-primary"
                        onClick={handleBacktest}
                        disabled={loading}
                    >
                        {loading ? 'Running...' : 'Run Backtest'}
                    </button>
                </div>

                {error && (
                    <div style={{ color: '#ef4444', fontSize: 12, marginBottom: 12 }}>Error: {error}</div>
                )}

                {loading && (
                    <div className="loading-container" style={{ padding: '24px' }}>
                        <div className="spinner"></div>
                        <div className="loading-text">Simulating trading strategy...</div>
                    </div>
                )}

                {!result && !loading && (
                    <div className="empty-state" style={{ padding: '24px' }}>
                        <div className="empty-state-icon">📈</div>
                        <div className="empty-state-title">Historical Backtest</div>
                        <div className="empty-state-text">Run a backtest to evaluate the AI model vs Benchmark</div>
                    </div>
                )}

                {result && !loading && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        {/* Summary Stats */}
                        <div className="stats-grid">
                            <div className="stat-item">
                                <div className="stat-label">Total Return</div>
                                <div className={`stat-value ${metrics?.total_return_pct > 0 ? 'positive' : 'negative'}`}>
                                    {(metrics?.total_return_pct * 100).toFixed(2)}%
                                </div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Benchmark Return</div>
                                <div className="stat-value" style={{ color: '#64748b' }}>
                                    {(metrics?.benchmark_return_pct * 100).toFixed(2)}%
                                </div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Sharpe Ratio</div>
                                <div className={`stat-value ${metrics?.sharpe_ratio > 1 ? 'positive' : ''}`}>
                                    {metrics?.sharpe_ratio?.toFixed(2)}
                                </div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Max Drawdown</div>
                                <div className="stat-value negative">
                                    {(metrics?.max_drawdown * 100).toFixed(2)}%
                                </div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Win Rate</div>
                                <div className="stat-value">
                                    {(metrics?.win_rate * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>

                        {/* Equity Curve Chart */}
                        {result.equity_curve && result.benchmark_curve && (
                            <Plot
                                data={[
                                    {
                                        x: result.benchmark_curve.dates,
                                        y: result.benchmark_curve.values,
                                        type: 'scatter',
                                        mode: 'lines',
                                        name: 'Buy & Hold',
                                        line: { color: '#64748b', width: 2, dash: 'dot' },
                                    },
                                    {
                                        x: result.equity_curve.dates,
                                        y: result.equity_curve.values,
                                        type: 'scatter',
                                        mode: 'lines',
                                        name: 'Model Strategy',
                                        line: { color: '#3b82f6', width: 2 },
                                        fill: 'tozeroy',
                                        fillcolor: 'rgba(59, 130, 246, 0.1)'
                                    }
                                ]}
                                layout={{
                                    ...CHART_LAYOUT,
                                    height: 250,
                                    title: false,
                                }}
                                config={{ displayModeBar: false, responsive: true }}
                                style={{ width: '100%' }}
                            />
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
