/**
 * QuantAI Dashboard — Main Application
 * Bloomberg-style institutional stock prediction platform.
 */

import React, { useState, useCallback, useEffect } from 'react';
import MarketPanel from './components/MarketPanel';
import AIPanel from './components/AIPanel';
import RiskPanel from './components/RiskPanel';
import PortfolioPanel from './components/PortfolioPanel';
import { predictStock } from './services/api';

export default function App() {
    const [ticker, setTicker] = useState('');
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentTime, setCurrentTime] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    const handleAnalyze = useCallback(async () => {
        const t = ticker.trim().toUpperCase();
        if (!t) return;

        setLoading(true);
        setError(null);

        try {
            const result = await predictStock(t);
            setData(result);
        } catch (err) {
            setError(err.message);
            setData(null);
        } finally {
            setLoading(false);
        }
    }, [ticker]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') handleAnalyze();
    };

    return (
        <div className="app">
            {/* ── Header ───────────────────────────────────────── */}
            <header className="header">
                <div className="header-brand">
                    <div className="header-logo">Q</div>
                    <span className="header-title">QuantAI</span>
                    <span className="header-subtitle">Institutional Stock Prediction</span>
                </div>

                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 2,
                }}>
                    <span style={{
                        fontFamily: 'JetBrains Mono, monospace',
                        fontSize: 15,
                        fontWeight: 600,
                        color: '#e2e8f0',
                        letterSpacing: '0.5px',
                    }}>
                        {currentTime.toLocaleTimeString('en-US', { hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                    <span style={{
                        fontSize: 10,
                        color: '#64748b',
                        letterSpacing: '0.5px',
                    }}>
                        {currentTime.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}
                    </span>
                </div>

                <div className="ticker-bar">
                    {error && (
                        <span style={{ color: '#ef4444', fontSize: 12, marginRight: 8 }}>
                            {error}
                        </span>
                    )}
                    <input
                        id="ticker-input"
                        className="ticker-input"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        onKeyDown={handleKeyDown}
                        placeholder="Enter ticker..."
                        autoComplete="off"
                    />
                    <button
                        id="analyze-btn"
                        className="btn btn-primary"
                        onClick={handleAnalyze}
                        disabled={loading || !ticker.trim()}
                    >
                        {loading ? (
                            <>
                                <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }}></span>
                                Analyzing...
                            </>
                        ) : (
                            '⚡ Analyze'
                        )}
                    </button>
                </div>
            </header>

            {/* ── Loading Overlay ──────────────────────────────── */}
            {loading && (
                <div style={{
                    padding: '12px 24px',
                    background: 'rgba(59, 130, 246, 0.08)',
                    borderBottom: '1px solid rgba(59, 130, 246, 0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                }}>
                    <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></div>
                    <span style={{ fontSize: 12, color: '#3b82f6' }}>
                        Fetching market data, computing features, running AI model, analyzing sentiment...
                    </span>
                </div>
            )}

            {/* ── Dashboard Grid ───────────────────────────────── */}
            <main className="dashboard">
                <MarketPanel data={data} prediction={data} />
                <AIPanel data={data} />
                <RiskPanel data={data} ticker={data?.ticker || ticker} />
                <PortfolioPanel />
            </main>

            {/* ── Footer ───────────────────────────────────────── */}
            <footer style={{
                padding: '10px 24px',
                borderTop: '1px solid #1e293b',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: 11,
                color: '#475569',
            }}>
                <span>QuantAI v1.0 — Transformer + FinBERT + DQN</span>
                <span>
                    {data?.ticker && (
                        <span style={{ color: '#3b82f6' }}>
                            {data.ticker} | Last: ${data.current_price?.toFixed(2)} | {data.news_count} headlines
                        </span>
                    )}
                </span>
                <span>⚠ Not financial advice</span>
            </footer>
        </div>
    );
}
