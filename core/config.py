"""
Configuration management for the Agentic Trading System.
"""
from typing import List, Optional
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(extra="ignore")  # Ignore extra environment variables
    
    # API Configuration
    app_name: str = "Agentic Trading System"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="trading_system", env="POSTGRES_DB")
    postgres_user: str = Field(default="trading_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="trading_pass", env="POSTGRES_PASSWORD")
    
    # External APIs
    mcp_api_url: str = Field(default="https://forecasting.guidry-cloud.com", env="MCP_API_URL")
    mcp_api_key: Optional[str] = Field(default="mock_api_key", env="MCP_API_KEY")
    dex_simulator_url: str = Field(default="http://localhost:8001", env="DEX_SIMULATOR_URL")
    
    # Mock services
    use_mock_services: bool = Field(default=True, env="USE_MOCK_SERVICES")
    
    # Exchange API Keys
    mexc_api_key: Optional[str] = Field(default=None, env="MEXC_API_KEY")
    mexc_secret_key: Optional[str] = Field(default=None, env="MEXC_SECRET_KEY")
    
    # DEX Configuration
    private_key: Optional[str] = Field(default=None, env="PRIVATE_KEY")
    wallet_address: Optional[str] = Field(default=None, env="WALLET_ADDRESS")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    vllm_endpoint: Optional[str] = Field(default="http://localhost:8002/v1", env="VLLM_ENDPOINT")
    
    # Blockchain RPC URLs
    bsc_rpc_url: str = Field(default="https://bsc-dataseed.binance.org/", env="BSC_RPC_URL")
    eth_rpc_url: str = Field(default="https://eth.llamarpc.com", env="ETH_RPC_URL")
    sol_rpc_url: str = Field(default="https://api.mainnet-beta.solana.com", env="SOL_RPC_URL")
    
    # Trading Configuration
    initial_capital: float = Field(default=1000.0, env="INITIAL_CAPITAL")
    max_position_size: float = Field(default=0.20, env="MAX_POSITION_SIZE")  # 20% max per asset
    max_daily_loss: float = Field(default=0.05, env="MAX_DAILY_LOSS")  # 5% max daily loss
    max_drawdown: float = Field(default=0.15, env="MAX_DRAWDOWN")  # 15% max drawdown
    trading_fee: float = Field(default=0.001, env="TRADING_FEE")  # 0.1% trading fee
    min_confidence: float = Field(default=0.7, env="MIN_CONFIDENCE")  # Minimum confidence for DQN trades
    
    # Supported Assets
    supported_assets: List[str] = [
        "AAVE", "ADA", "AXS", "BTC", "CRO", "DOGE", "ETH", 
        "GALA", "IMX", "MANA", "PEPE", "POPCAT", "SAND", "SOL", "SUI"
    ]
    
    # Asset Risk Tiers (for position sizing)
    tier_1_assets: List[str] = ["BTC", "ETH", "SOL"]  # Major cryptos - higher allocation allowed
    tier_2_assets: List[str] = ["ADA", "AAVE", "CRO"]  # Mid-cap - moderate allocation
    tier_3_assets: List[str] = ["DOGE", "MANA", "SAND", "GALA", "AXS", "IMX", "SUI"]  # Higher risk
    tier_4_assets: List[str] = ["PEPE", "POPCAT"]  # Meme coins - lowest allocation
    
    # Trading Intervals
    observation_interval: str = "minutes"  # Observe market behavior
    decision_interval: str = "hours"  # Make trading decisions
    forecast_interval: str = "days"  # Long-term forecasting
    
    # Agent Configuration
    agent_heartbeat_interval: int = 30  # seconds
    agent_timeout: int = 300  # seconds
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="/app/logs/trading_system.log", env="LOG_FILE")
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_asset_tier(self, asset: str) -> int:
        """Get the risk tier for an asset."""
        if asset in self.tier_1_assets:
            return 1
        elif asset in self.tier_2_assets:
            return 2
        elif asset in self.tier_3_assets:
            return 3
        elif asset in self.tier_4_assets:
            return 4
        return 3  # Default to tier 3
    
    def get_max_position_for_asset(self, asset: str) -> float:
        """Get maximum position size for an asset based on its tier."""
        tier = self.get_asset_tier(asset)
        if tier == 1:
            return self.max_position_size  # 20% for tier 1
        elif tier == 2:
            return self.max_position_size * 0.75  # 15% for tier 2
        elif tier == 3:
            return self.max_position_size * 0.5  # 10% for tier 3
        else:  # tier 4
            return self.max_position_size * 0.25  # 5% for tier 4 (meme coins)
    


# Global settings instance
settings = Settings()

