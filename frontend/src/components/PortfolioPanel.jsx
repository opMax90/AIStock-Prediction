/**
 * PortfolioPanel — Allocation pie chart, risk contribution,
 * expected return vs risk scatter, and position sizing.
 */

import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { optimizePortfolio } from '../services/api';

const CHART_BG = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#0d1525',
    font: { family: 'Inter', color: '#94a3b8', size: 10 },
};

export default function PortfolioPanel() {
    const [tickers, setTickers] = useState('AAPL,MSFT,GOOGL,AMZN,TSLA');
    const [method, setMethod] = useState('mean_variance');
    const [capital, setCapital] = useState(100000);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleOptimize = async () => {
        const tickerList = tickers.split(',').map(t => t.trim()).filter(Boolean);
        if (tickerList.length < 2) return;

        setLoading(true);
        setError(null);
        try {
            const data = await optimizePortfolio(tickerList, method, capital);
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const optimization = result?.optimization;
    const sizing = result?.position_sizing;
    const tickerList = result?.tickers || [];

    return (
        <div className="card">
            <div className="card-header">
                <div className="card-title">
                    <span className="card-title-icon" style={{ background: '#10b981' }}></span>
                    Portfolio Optimization
                </div>
            </div>
            <div className="card-body">
                {/* Input Controls */}
                <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
                    <input
                        className="ticker-input"
                        style={{ width: 260, textTransform: 'uppercase' }}
                        value={tickers}
                        onChange={e => setTickers(e.target.value)}
                        placeholder="AAPL,MSFT,GOOGL..."
                    />
                    <select
                        className="ticker-input"
                        style={{ width: 160, textTransform: 'none' }}
                        value={method}
                        onChange={e => setMethod(e.target.value)}
                    >
                        <option value="mean_variance">Mean-Variance</option>
                        <option value="risk_parity">Risk Parity</option>
                    </select>
                    <input
                        className="ticker-input"
                        style={{ width: 110, textTransform: 'none' }}
                        type="number"
                        value={capital}
                        onChange={e => setCapital(Number(e.target.value))}
                        placeholder="Capital"
                    />
                    <button
                        className="btn btn-primary"
                        onClick={handleOptimize}
                        disabled={loading}
                    >
                        {loading ? 'Optimizing...' : 'Optimize'}
                    </button>
                </div>

                {error && (
                    <div style={{ color: '#ef4444', fontSize: 12, marginBottom: 12 }}>Error: {error}</div>
                )}

                {loading && (
                    <div className="loading-container">
                        <div className="spinner"></div>
                        <div className="loading-text">Optimizing portfolio...</div>
                    </div>
                )}

                {!result && !loading && (
                    <div className="empty-state">
                        <div className="empty-state-icon">💼</div>
                        <div className="empty-state-title">Portfolio Optimization</div>
                        <div className="empty-state-text">Enter tickers and click Optimize for allocation recommendations</div>
                    </div>
                )}

                {result && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        {/* Summary Stats */}
                        <div className="stats-grid">
                            {optimization?.expected_return != null && (
                                <div className="stat-item">
                                    <div className="stat-label">Expected Return</div>
                                    <div className={`stat-value ${optimization.expected_return > 0 ? 'positive' : 'negative'}`}>
                                        {(optimization.expected_return * 100).toFixed(2)}%
                                    </div>
                                </div>
                            )}
                            {optimization?.volatility != null && (
                                <div className="stat-item">
                                    <div className="stat-label">Portfolio Vol</div>
                                    <div className="stat-value">{(optimization.volatility * 100).toFixed(2)}%</div>
                                </div>
                            )}
                            {optimization?.sharpe_ratio != null && (
                                <div className="stat-item">
                                    <div className="stat-label">Sharpe Ratio</div>
                                    <div className={`stat-value ${optimization.sharpe_ratio > 1 ? 'positive' : ''}`}>
                                        {optimization.sharpe_ratio.toFixed(2)}
                                    </div>
                                </div>
                            )}
                            {sizing?.diversification_score != null && (
                                <div className="stat-item">
                                    <div className="stat-label">Diversification</div>
                                    <div className="stat-value">{(sizing.diversification_score * 100).toFixed(0)}%</div>
                                </div>
                            )}
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                            {/* Allocation Pie */}
                            {optimization?.weights && (
                                <Plot
                                    data={[{
                                        type: 'pie',
                                        labels: tickerList,
                                        values: optimization.weights.map(w => Math.max(w * 100, 0)),
                                        marker: {
                                            colors: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#ef4444'],
                                        },
                                        hole: 0.45,
                                        textinfo: 'label+percent',
                                        textfont: { size: 11, family: 'Inter', color: '#e2e8f0' },
                                    }]}
                                    layout={{
                                        ...CHART_BG,
                                        height: 200,
                                        margin: { t: 10, r: 10, b: 10, l: 10 },
                                        showlegend: false,
                                        annotations: [{
                                            text: method === 'risk_parity' ? 'RP' : 'MVO',
                                            showarrow: false,
                                            font: { size: 14, color: '#94a3b8', family: 'JetBrains Mono' },
                                        }],
                                    }}
                                    config={{ displayModeBar: false, staticPlot: true }}
                                    style={{ width: '100%' }}
                                />
                            )}

                            {/* Risk Contribution */}
                            {sizing?.risk_contribution && (
                                <Plot
                                    data={[{
                                        type: 'bar',
                                        x: tickerList,
                                        y: sizing.risk_contribution.map(r => r * 100),
                                        marker: {
                                            color: sizing.risk_contribution.map(r =>
                                                r > 0.3 ? '#ef4444' : r > 0.2 ? '#f59e0b' : '#10b981'
                                            ),
                                        },
                                        text: sizing.risk_contribution.map(r => `${(r * 100).toFixed(1)}%`),
                                        textposition: 'outside',
                                        textfont: { size: 10, family: 'JetBrains Mono', color: '#94a3b8' },
                                    }]}
                                    layout={{
                                        ...CHART_BG,
                                        height: 200,
                                        margin: { t: 20, r: 10, b: 30, l: 40 },
                                        xaxis: { gridcolor: '#1a2332' },
                                        yaxis: { gridcolor: '#1a2332', title: { text: 'Risk %', font: { size: 10 } } },
                                        title: {
                                            text: 'Risk Contribution',
                                            font: { size: 11, color: '#64748b' },
                                            x: 0.5, y: 0.98,
                                        },
                                    }}
                                    config={{ displayModeBar: false, staticPlot: true }}
                                    style={{ width: '100%' }}
                                />
                            )}
                        </div>

                        {/* Dollar Amounts */}
                        {sizing?.dollar_amounts && (
                            <div>
                                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                    Position Sizing (${capital.toLocaleString()})
                                </div>
                                <div className="stats-grid" style={{ gridTemplateColumns: `repeat(${tickerList.length}, 1fr)` }}>
                                    {tickerList.map((t, i) => (
                                        <div className="stat-item" key={t}>
                                            <div className="stat-label">{t}</div>
                                            <div className="stat-value" style={{ fontSize: 14 }}>
                                                ${sizing.dollar_amounts[i]?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                            </div>
                                            <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
                                                {(optimization.weights[i] * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
