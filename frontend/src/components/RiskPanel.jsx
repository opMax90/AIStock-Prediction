/**
 * RiskPanel — Drawdown chart, Sharpe ratio, volatility meter,
 * risk decomposition, and backtest metrics.
 */

import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { runBacktest } from '../services/api';

function MetricsTable({ metrics }) {
    if (!metrics) return null;

    const rows = [
        ['CAGR', `${(metrics.cagr * 100).toFixed(2)}%`, metrics.cagr > 0],
        ['Sharpe Ratio', metrics.sharpe_ratio?.toFixed(2), metrics.sharpe_ratio > 1],
        ['Sortino Ratio', metrics.sortino_ratio?.toFixed(2), metrics.sortino_ratio > 1],
        ['Max Drawdown', `${(metrics.max_drawdown * 100).toFixed(2)}%`, false],
        ['Win Rate', `${(metrics.win_rate * 100).toFixed(1)}%`, metrics.win_rate > 0.5],
        ['Profit Factor', metrics.profit_factor?.toFixed(2), metrics.profit_factor > 1],
        ['Alpha', `${(metrics.alpha * 100).toFixed(2)}%`, metrics.alpha > 0],
        ['Beta', metrics.beta?.toFixed(2), null],
        ['Info Ratio', metrics.information_ratio?.toFixed(2), metrics.information_ratio > 0],
        ['Volatility', `${(metrics.annual_volatility * 100).toFixed(1)}%`, null],
    ];

    return (
        <table className="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {rows.map(([name, value, isGood]) => (
                    <tr key={name}>
                        <td style={{ color: '#94a3b8' }}>{name}</td>
                        <td style={{
                            color: isGood === true ? '#10b981' : isGood === false ? '#ef4444' : '#e2e8f0'
                        }}>
                            {value}
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

export default function RiskPanel({ data, ticker }) {
    const [backtest, setBacktest] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleBacktest = async () => {
        if (!ticker) return;
        setLoading(true);
        setError(null);
        try {
            const result = await runBacktest(ticker);
            setBacktest(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const riskData = data?.explainability?.risk;

    return (
        <div className="card">
            <div className="card-header">
                <div className="card-title">
                    <span className="card-title-icon" style={{ background: '#ef4444' }}></span>
                    Risk Analysis & Backtest
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={handleBacktest}
                    disabled={loading || !ticker}
                    style={{ fontSize: 11 }}
                >
                    {loading ? 'Running...' : '▶ Backtest'}
                </button>
            </div>
            <div className="card-body">
                {!data && !backtest ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">⚠️</div>
                        <div className="empty-state-title">Risk Analysis</div>
                        <div className="empty-state-text">Analyze a stock to view risk metrics, or run a backtest</div>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        {/* Risk Decomposition */}
                        {riskData && (
                            <div className="stats-grid">
                                <div className="stat-item">
                                    <div className="stat-label">Risk Level</div>
                                    <div className="stat-value" style={{
                                        color: riskData.risk_level === 'LOW' ? '#10b981' :
                                            riskData.risk_level === 'MODERATE' ? '#f59e0b' : '#ef4444'
                                    }}>
                                        {riskData.risk_level}
                                    </div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">Total Uncertainty</div>
                                    <div className="stat-value">{(riskData.total_uncertainty * 100).toFixed(2)}%</div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">Model Uncertainty</div>
                                    <div className="stat-value">{(riskData.model_uncertainty * 100).toFixed(2)}%</div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">Directional Risk</div>
                                    <div className="stat-value">{(riskData.directional_risk * 100).toFixed(1)}%</div>
                                </div>
                            </div>
                        )}

                        {/* Recommendation */}
                        {riskData?.recommendation && (
                            <div className="signal-indicator" style={{ borderColor: 'rgba(245, 158, 11, 0.3)' }}>
                                <div className="signal-dot neutral"></div>
                                <div>
                                    <div className="signal-text" style={{ color: '#f59e0b', fontSize: 12 }}>AI Recommendation</div>
                                    <div className="signal-sub" style={{ fontSize: 12 }}>{riskData.recommendation}</div>
                                </div>
                            </div>
                        )}

                        {/* Error */}
                        {error && (
                            <div style={{ color: '#ef4444', fontSize: 12, padding: 8 }}>Error: {error}</div>
                        )}

                        {/* Loading */}
                        {loading && (
                            <div className="loading-container" style={{ padding: 24 }}>
                                <div className="spinner"></div>
                                <div className="loading-text">Running backtest...</div>
                            </div>
                        )}

                        {/* Backtest Results */}
                        {backtest && (
                            <>
                                {/* Equity Curve */}
                                <Plot
                                    data={[
                                        {
                                            x: backtest.equity_curve?.dates,
                                            y: backtest.equity_curve?.values,
                                            type: 'scatter',
                                            mode: 'lines',
                                            name: 'Strategy',
                                            line: { color: '#3b82f6', width: 2 },
                                            fill: 'tozeroy',
                                            fillcolor: 'rgba(59, 130, 246, 0.05)',
                                        },
                                        {
                                            x: backtest.benchmark_curve?.dates,
                                            y: backtest.benchmark_curve?.values,
                                            type: 'scatter',
                                            mode: 'lines',
                                            name: 'Buy & Hold',
                                            line: { color: '#64748b', width: 1.5, dash: 'dot' },
                                        },
                                    ]}
                                    layout={{
                                        height: 220,
                                        paper_bgcolor: 'transparent',
                                        plot_bgcolor: '#0d1525',
                                        font: { family: 'Inter', color: '#94a3b8', size: 10 },
                                        margin: { t: 20, r: 20, b: 30, l: 60 },
                                        xaxis: { gridcolor: '#1a2332', nticks: 8 },
                                        yaxis: { gridcolor: '#1a2332', tickformat: '$,.0f' },
                                        legend: { x: 0, y: 1.15, orientation: 'h', font: { size: 10 } },
                                        showlegend: true,
                                    }}
                                    config={{ displayModeBar: false, responsive: true }}
                                    style={{ width: '100%' }}
                                />

                                {/* Metrics Table */}
                                <MetricsTable metrics={backtest.metrics} />
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
