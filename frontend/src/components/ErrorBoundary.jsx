import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({
            error: error,
            errorInfo: errorInfo
        });
        console.error("ErrorBoundary caught an error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    padding: '24px',
                    margin: '16px',
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: '8px',
                    color: '#ef4444',
                    fontFamily: 'Inter, sans-serif'
                }}>
                    <h3 style={{ margin: '0 0 12px 0', fontSize: '16px' }}>Dashboard Render Error</h3>
                    <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: '#f87171' }}>
                        Something went wrong while displaying the dashboard.
                    </p>
                    <details style={{ whiteSpace: 'pre-wrap', fontSize: '11px', color: '#94a3b8', background: '#0a0e17', padding: '12px', borderRadius: '4px' }}>
                        <summary style={{ cursor: 'pointer', marginBottom: '8px', color: '#ef4444' }}>Technical Details</summary>
                        {this.state.error && this.state.error.toString()}
                        <br />
                        {this.state.errorInfo && this.state.errorInfo.componentStack}
                    </details>
                    <button
                        onClick={() => window.location.reload()}
                        style={{
                            marginTop: '16px',
                            background: '#ef4444',
                            color: 'white',
                            border: 'none',
                            padding: '8px 16px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '13px',
                            fontWeight: '600'
                        }}
                    >
                        Reload Dashboard
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
