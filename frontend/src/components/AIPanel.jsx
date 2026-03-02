/**
 * AIPanel — Sentiment gauge, signal indicator, confidence score,
 * attention visualization, and feature importance.
 */

import React from 'react';
import Plot from 'react-plotly.js';

function SentimentGauge({ polarity, confidence }) {
    const value = polarity || 0;
    const color = value > 0.1 ? '#10b981' : value < -0.1 ? '#ef4444' : '#f59e0b';
    const label = value > 0.1 ? 'Bullish' : value < -0.1 ? 'Bearish' : 'Neutral';

    return (
        <div className="gauge-container">
            <Plot
                data={[{
                    type: 'indicator',
                    mode: 'gauge+number',
                    value: value,
                    gauge: {
                        axis: { range: [-1, 1], tickwidth: 1, tickcolor: '#1e293b' },
                        bar: { color },
                        bgcolor: '#0d1525',
                        borderwidth: 0,
                        steps: [
                            { range: [-1, -0.3], color: 'rgba(239, 68, 68, 0.15)' },
                            { range: [-0.3, 0.3], color: 'rgba(245, 158, 11, 0.1)' },
                            { range: [0.3, 1], color: 'rgba(16, 185, 129, 0.15)' },
                        ],
                    },
                    number: { font: { size: 20, color: color, family: 'JetBrains Mono' }, valueformat: '.2f' },
                }]}
                layout={{
                    width: 200,
                    height: 120,
                    margin: { t: 24, r: 10, b: 0, l: 10 },
                    paper_bgcolor: 'transparent',
                    font: { color: '#94a3b8', family: 'Inter' },
                }}
                config={{ displayModeBar: false, staticPlot: true }}
            />
            <div className="gauge-label">Sentiment — {label}</div>
        </div>
    );
}

function SignalIndicator({ prediction }) {
    if (!prediction) {
        return (
            <div className="signal-indicator">
                <div className="signal-dot neutral"></div>
                <div>
                    <div className="signal-text">No Signal</div>
                    <div className="signal-sub">Run analysis to generate signal</div>
                </div>
            </div>
        );
    }

    const dirProb = prediction.direction_prob || 0.5;
    const isBullish = dirProb > 0.55;
    const isBearish = dirProb < 0.45;
    const signal = isBullish ? 'BUY' : isBearish ? 'SELL' : 'HOLD';
    const signalClass = isBullish ? 'bullish' : isBearish ? 'bearish' : 'neutral';

    return (
        <div className="signal-indicator">
            <div className={`signal-dot ${signalClass}`}></div>
            <div>
                <div className="signal-text" style={{ color: `var(--${signalClass})` }}>{signal}</div>
                <div className="signal-sub">P(Up) = {(dirProb * 100).toFixed(1)}%</div>
            </div>
        </div>
    );
}

function ConfidenceScore({ confidence }) {
    const pct = (confidence || 0) * 100;
    const level = pct > 70 ? 'high' : pct > 40 ? 'medium' : 'low';

    return (
        <div className="stat-item">
            <div className="stat-label">Model Confidence</div>
            <div className={`stat-value ${level === 'high' ? 'positive' : level === 'low' ? 'negative' : ''}`}>
                {pct.toFixed(1)}%
            </div>
            <div className="confidence-bar">
                <div
                    className={`confidence-bar-fill ${level}`}
                    style={{ width: `${pct}%` }}
                ></div>
            </div>
        </div>
    );
}

function FeatureImportance({ features }) {
    if (!features?.feature_importance) return null;

    const topFeatures = Object.entries(features.feature_importance).slice(0, 8);
    const maxVal = Math.max(...topFeatures.map(([, v]) => v), 0.001);

    return (
        <div>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Top Feature Importance
            </div>
            {topFeatures.map(([name, value]) => (
                <div className="feature-bar" key={name}>
                    <div className="feature-name">{name}</div>
                    <div className="feature-bar-track">
                        <div
                            className="feature-bar-fill"
                            style={{ width: `${(value / maxVal) * 100}%` }}
                        ></div>
                    </div>
                    <div className="feature-value">{(value * 100).toFixed(1)}%</div>
                </div>
            ))}
        </div>
    );
}

export default function AIPanel({ data }) {
    const prediction = data?.prediction;
    const sentiment = data?.sentiment;
    const explainability = data?.explainability;

    return (
        <div className="card">
            <div className="card-header">
                <div className="card-title">
                    <span className="card-title-icon" style={{ background: '#8b5cf6' }}></span>
                    AI Analysis
                </div>
                {prediction && (
                    <span className={`badge ${prediction.direction_prob > 0.5 ? 'badge-bullish' : 'badge-bearish'}`}>
                        {prediction.confidence > 0.6 ? 'HIGH CONF' : prediction.confidence > 0.4 ? 'MED CONF' : 'LOW CONF'}
                    </span>
                )}
            </div>
            <div className="card-body">
                {!data ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">🤖</div>
                        <div className="empty-state-title">AI Analysis</div>
                        <div className="empty-state-text">Analyze a stock to see AI predictions, sentiment, and feature importance</div>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                        {/* Signal + Sentiment Row */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                            <SignalIndicator prediction={prediction} />
                            <ConfidenceScore confidence={prediction?.confidence} />
                        </div>

                        {/* Sentiment Gauge */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, alignItems: 'start' }}>
                            <SentimentGauge
                                polarity={sentiment?.sentiment_polarity}
                                confidence={sentiment?.sentiment_confidence}
                            />
                            <div className="stats-grid" style={{ gridTemplateColumns: '1fr' }}>
                                <div className="stat-item">
                                    <div className="stat-label">Sentiment Momentum</div>
                                    <div className={`stat-value ${(sentiment?.sentiment_momentum || 0) > 0 ? 'positive' : 'negative'}`}>
                                        {(sentiment?.sentiment_momentum || 0).toFixed(3)}
                                    </div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">Headlines Analyzed</div>
                                    <div className="stat-value">{sentiment?.n_headlines || 0}</div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">Bullish / Bearish</div>
                                    <div className="stat-value">
                                        <span style={{ color: 'var(--bullish)' }}>{sentiment?.n_bullish || 0}</span>
                                        {' / '}
                                        <span style={{ color: 'var(--bearish)' }}>{sentiment?.n_bearish || 0}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Feature Importance */}
                        <FeatureImportance features={explainability?.feature_importance} />

                        {/* Sentiment Impact */}
                        {explainability?.sentiment_impact && (
                            <div className="stat-item">
                                <div className="stat-label">Sentiment Impact on Prediction</div>
                                <div className={`stat-value ${explainability.sentiment_impact.sentiment_is_bullish ? 'positive' : 'negative'}`}>
                                    {explainability.sentiment_impact.sentiment_is_bullish ? '▲' : '▼'}{' '}
                                    {explainability.sentiment_impact.sentiment_contribution_pct?.toFixed(1)}%
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
