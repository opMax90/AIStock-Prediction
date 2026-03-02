/**
 * MarketPanel — Candlestick chart with prediction overlay and confidence interval shading.
 */

import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

const CHART_LAYOUT = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#0d1525',
    font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 11 },
    margin: { t: 30, r: 20, b: 40, l: 60 },
    xaxis: {
        gridcolor: '#1a2332',
        linecolor: '#1e293b',
        rangeslider: { visible: false },
        type: 'category',
        nticks: 10,
    },
    yaxis: {
        gridcolor: '#1a2332',
        linecolor: '#1e293b',
        side: 'right',
        tickformat: '$.2f',
    },
    legend: {
        x: 0,
        y: 1.1,
        orientation: 'h',
        font: { size: 10, color: '#64748b' },
    },
    dragmode: 'zoom',
};

export default function MarketPanel({ data, prediction }) {
    const traces = useMemo(() => {
        if (!data?.historical) return [];

        const { dates, open, high, low, close } = data.historical;
        const plotTraces = [];

        // Candlestick chart
        plotTraces.push({
            type: 'candlestick',
            x: dates,
            open,
            high,
            low,
            close,
            increasing: { line: { color: '#10b981', width: 1 }, fillcolor: '#10b981' },
            decreasing: { line: { color: '#ef4444', width: 1 }, fillcolor: '#ef4444' },
            name: 'OHLC',
            whiskerwidth: 0.5,
        });

        // Prediction overlay
        if (prediction?.forecast) {
            const f = prediction.forecast;
            const lastIdx = dates.length - 1;
            const predDate = `${dates[lastIdx]}+1`;
            const extDates = [...dates.slice(-5), predDate];

            // Confidence interval (95%)
            if (f.price_ci_upper_95 && f.price_ci_lower_95) {
                const upperBand = [...close.slice(-5), f.price_ci_upper_95];
                const lowerBand = [...close.slice(-5), f.price_ci_lower_95];

                plotTraces.push({
                    x: extDates,
                    y: upperBand,
                    type: 'scatter',
                    mode: 'lines',
                    line: { width: 0 },
                    showlegend: false,
                });

                plotTraces.push({
                    x: extDates,
                    y: lowerBand,
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tonexty',
                    fillcolor: 'rgba(59, 130, 246, 0.08)',
                    line: { width: 0 },
                    name: '95% CI',
                });
            }

            // 80% CI
            if (f.price_ci_upper_80 && f.price_ci_lower_80) {
                const upperBand80 = [...close.slice(-5), f.price_ci_upper_80];
                const lowerBand80 = [...close.slice(-5), f.price_ci_lower_80];

                plotTraces.push({
                    x: extDates,
                    y: upperBand80,
                    type: 'scatter',
                    mode: 'lines',
                    line: { width: 0 },
                    showlegend: false,
                });

                plotTraces.push({
                    x: extDates,
                    y: lowerBand80,
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tonexty',
                    fillcolor: 'rgba(59, 130, 246, 0.15)',
                    line: { width: 0 },
                    name: '80% CI',
                });
            }

            // Predicted price point
            plotTraces.push({
                x: [predDate],
                y: [f.predicted_price],
                type: 'scatter',
                mode: 'markers',
                marker: {
                    size: 12,
                    color: f.predicted_return_pct >= 0 ? '#10b981' : '#ef4444',
                    symbol: 'diamond',
                    line: { color: 'white', width: 1.5 },
                },
                name: `Predicted: $${f.predicted_price?.toFixed(2)}`,
            });
        }

        return plotTraces;
    }, [data, prediction]);

    if (!data?.historical) {
        return (
            <div className="card">
                <div className="card-header">
                    <div className="card-title">
                        <span className="card-title-icon" style={{ background: '#3b82f6' }}></span>
                        Market Data
                    </div>
                </div>
                <div className="card-body">
                    <div className="empty-state">
                        <div className="empty-state-icon">📊</div>
                        <div className="empty-state-title">No Market Data</div>
                        <div className="empty-state-text">Enter a ticker symbol and click Analyze to view market data</div>
                    </div>
                </div>
            </div>
        );
    }

    const currentPrice = data.current_price;
    const pred = prediction?.prediction;

    return (
        <div className="card">
            <div className="card-header">
                <div className="card-title">
                    <span className="card-title-icon" style={{ background: '#3b82f6' }}></span>
                    Market Data — {data.ticker}
                </div>
                {pred && (
                    <span className={`badge ${pred.predicted_return_pct >= 0 ? 'badge-bullish' : 'badge-bearish'}`}>
                        {pred.predicted_return_pct >= 0 ? '▲' : '▼'} {Math.abs(pred.predicted_return_pct)?.toFixed(2)}%
                    </span>
                )}
            </div>
            <div className="card-body" style={{ padding: '8px' }}>
                <div className="stats-grid" style={{ marginBottom: 12, padding: '0 10px' }}>
                    <div className="stat-item">
                        <div className="stat-label">Current Price</div>
                        <div className="stat-value">${currentPrice?.toFixed(2)}</div>
                    </div>
                    {pred && (
                        <>
                            <div className="stat-item">
                                <div className="stat-label">Predicted Closing Price</div>
                                <div className={`stat-value ${pred.predicted_return_pct >= 0 ? 'positive' : 'negative'}`}>
                                    ${prediction.forecast?.predicted_price?.toFixed(2)}
                                </div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Direction Prob</div>
                                <div className={`stat-value ${pred.direction_prob > 0.5 ? 'positive' : 'negative'}`}>
                                    {(pred.direction_prob * 100)?.toFixed(1)}%
                                </div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Confidence</div>
                                <div className="stat-value">{(pred.confidence * 100)?.toFixed(1)}%</div>
                            </div>
                        </>
                    )}
                </div>

                <Plot
                    data={traces}
                    layout={{
                        ...CHART_LAYOUT,
                        height: 380,
                        title: false,
                    }}
                    config={{
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
                    }}
                    style={{ width: '100%' }}
                />
            </div>
        </div>
    );
}
