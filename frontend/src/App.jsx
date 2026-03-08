/**
 * QuantAI Dashboard — Main Application
 * Bloomberg-style institutional stock prediction platform.
 */

import React, { useState, useCallback } from 'react';
import MarketPanel from './components/MarketPanel';
import AIPanel from './components/AIPanel';
import RiskPanel from './components/RiskPanel';
import PortfolioPanel from './components/PortfolioPanel';
import Header from './components/Header';
import Footer from './components/Footer';
import ErrorBoundary from './components/ErrorBoundary';
import { predictStock } from './services/api';

export default function App() {
    const [ticker, setTicker] = useState('');
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

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
            <Header
                ticker={ticker}
                setTicker={setTicker}
                handleAnalyze={handleAnalyze}
                loading={loading}
                error={error}
            />

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
                <ErrorBoundary>
                    <MarketPanel data={data} prediction={data} />
                    <AIPanel data={data} />
                    <RiskPanel data={data} ticker={data?.ticker || ticker} />
                    <PortfolioPanel />
                </ErrorBoundary>
            </main>

            {/* ── Footer ───────────────────────────────────────── */}
            <Footer data={data} ticker={ticker} />
        </div>
    );
}
