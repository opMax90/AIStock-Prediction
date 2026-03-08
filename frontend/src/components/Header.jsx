import React, { useState, useEffect } from 'react';

export default function Header({ ticker, setTicker, handleAnalyze, loading, error }) {
    const [currentTime, setCurrentTime] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') handleAnalyze();
    };

    return (
        <header className="header" style={{
            position: 'sticky',
            top: 0,
            zIndex: 100,
            transition: 'all 0.3s ease'
        }}>
            <div className="header-brand">
                <div className="header-logo" style={{
                    transition: 'transform 0.3s ease'
                }}>Q</div>
                <span className="header-title">QuantAI</span>
                <span className="header-subtitle" style={{ letterSpacing: '2px' }}>Institutional Stock Prediction</span>
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
                    <span style={{
                        color: '#ef4444',
                        fontSize: 12,
                        marginRight: 8,
                        animation: 'fadeIn 0.3s ease'
                    }}>
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
                    style={{
                        padding: '10px 16px',
                        fontSize: '14px',
                        transition: 'box-shadow 0.2s ease, border-color 0.2s ease'
                    }}
                />
                <button
                    id="analyze-btn"
                    className="btn btn-primary"
                    onClick={handleAnalyze}
                    disabled={loading || !ticker.trim()}
                    style={{
                        padding: '10px 20px',
                        fontSize: '13px',
                        transition: 'all 0.2s ease'
                    }}
                >
                    {loading ? (
                        <>
                            <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2, marginRight: 6 }}></span>
                            Analyzing...
                        </>
                    ) : (
                        '⚡ Analyze'
                    )}
                </button>
            </div>
        </header>
    );
}
