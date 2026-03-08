import React from 'react';

export default function Footer({ data, ticker }) {
    return (
        <footer style={{
            padding: '12px 24px',
            borderTop: '1px solid #1e293b',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: 11,
            color: '#475569',
            background: 'var(--bg-header)',
        }}>
            <span>QuantAI v1.0 — Transformer + FinBERT + DQN</span>
            <span>
                {data?.ticker ? (
                    <span style={{ color: '#3b82f6', fontWeight: '500' }}>
                        {data.ticker} | Last: ${data.current_price?.toFixed(2)} | {data.news_count} headlines
                    </span>
                ) : (
                    <span>Awaiting Analysis...</span>
                )}
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ color: '#f59e0b' }}>⚠</span>
                Not financial advice
            </span>
        </footer>
    );
}
