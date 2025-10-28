-- Initialize database for Agentic Trading System

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(255) UNIQUE NOT NULL,
    ticker VARCHAR(50) NOT NULL,
    action VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    total_cost DECIMAL(20, 8) NOT NULL,
    fee DECIMAL(20, 8) NOT NULL,
    confidence DECIMAL(5, 4),
    reasoning TEXT,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on ticker and executed_at
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at DESC);

-- Create portfolio_snapshots table
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    balance_usdc DECIMAL(20, 8) NOT NULL,
    total_value_usdc DECIMAL(20, 8) NOT NULL,
    daily_pnl DECIMAL(20, 8) NOT NULL,
    total_pnl DECIMAL(20, 8) NOT NULL,
    holdings JSONB,
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on snapshot_at
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_at ON portfolio_snapshots(snapshot_at DESC);

-- Create agent_signals table
CREATE TABLE IF NOT EXISTS agent_signals (
    id SERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    ticker VARCHAR(50),
    action VARCHAR(10),
    confidence DECIMAL(5, 4),
    data JSONB,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on ticker and created_at
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON agent_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON agent_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_agent_type ON agent_signals(agent_type);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,
    win_rate DECIMAL(5, 4) NOT NULL,
    total_pnl DECIMAL(20, 8) NOT NULL,
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(5, 4) NOT NULL,
    current_drawdown DECIMAL(5, 4) NOT NULL,
    roi DECIMAL(10, 4) NOT NULL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on calculated_at
CREATE INDEX IF NOT EXISTS idx_performance_calculated_at ON performance_metrics(calculated_at DESC);

-- Create tracked_wallets table
CREATE TABLE IF NOT EXISTS tracked_wallets (
    id SERIAL PRIMARY KEY,
    address VARCHAR(255) UNIQUE NOT NULL,
    blockchain VARCHAR(50) NOT NULL,
    performance DECIMAL(10, 4) DEFAULT 0.0,
    trade_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5, 4) DEFAULT 0.0,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_checked_at TIMESTAMP
);

-- Create index on blockchain and address
CREATE INDEX IF NOT EXISTS idx_wallets_blockchain ON tracked_wallets(blockchain);
CREATE INDEX IF NOT EXISTS idx_wallets_performance ON tracked_wallets(performance DESC);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;

