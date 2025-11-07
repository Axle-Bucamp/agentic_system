"""
Configuration management for the Agentic Trading System.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from pydantic import Field, ConfigDict, model_validator
from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from core.models import AgentType


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
    
    # Blockscout MCP Configuration
    blockscout_mcp_url: Optional[str] = Field(
        default="http://blockscout-mcp:8080",
        env="BLOCKSCOUT_MCP_URL"
    )
    
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
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    vllm_endpoint: Optional[str] = Field(default="http://localhost:8002/v1", env="VLLM_ENDPOINT")
    
    # CAMEL Configuration (default to stable Gemini 1.5 Pro for best compatibility)
    camel_default_model: str = Field(default="gemini-1.5-pro", env="CAMEL_DEFAULT_MODEL")
    camel_coordinator_model: str = Field(default="gemini-1.5-pro", env="CAMEL_COORDINATOR_MODEL")
    camel_task_model: str = Field(default="gemini-1.5-pro", env="CAMEL_TASK_MODEL")
    camel_worker_model: str = Field(default="gemini-1.5-pro", env="CAMEL_WORKER_MODEL")
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field(default="trading_memory", env="QDRANT_COLLECTION_NAME")
    
    # Memory Configuration
    memory_chat_history_limit: int = Field(default=100, env="MEMORY_CHAT_HISTORY_LIMIT")
    memory_retrieve_limit: int = Field(default=3, env="MEMORY_RETRIEVE_LIMIT")
    memory_token_limit: int = Field(default=4096, env="MEMORY_TOKEN_LIMIT")
    memory_embedding_model: str = Field(default="nomic-embed-text", env="MEMORY_EMBEDDING_MODEL")
    memory_embedding_provider: str = Field(default="ollama", env="MEMORY_EMBEDDING_PROVIDER")  # "ollama" or "openai"
    
    # Ollama Configuration
    ollama_url: str = Field(default="http://ollama:11434", env="OLLAMA_URL")
    ollama_model: str = Field(default="nomic-embed-text", env="OLLAMA_MODEL")
    
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
    agent_schedule_profile: str = Field(default="minutes", env="AGENT_SCHEDULE_PROFILE")
    agent_schedule_profiles: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "minutes": {
                "memory": 600,
                "dqn": 300,
                "chart": 300,
                "risk": 120,
                "news": 900,
                "copytrade": 180,
                "orchestrator": 300,
                "workforce": 300,
            },
            "hours": {
                "memory": 3600,
                "dqn": 1800,
                "chart": 1800,
                "risk": 1200,
                "news": 3600,
                "copytrade": 900,
                "orchestrator": 1800,
                "workforce": 1800,
            },
            "days": {
                "memory": 21600,
                "dqn": 14400,
                "chart": 10800,
                "risk": 7200,
                "news": 43200,
                "copytrade": 3600,
                "orchestrator": 14400,
                "workforce": 14400,
            },
        }
    )
    agent_cycle_overrides: Dict[str, int] = Field(default_factory=dict, env="AGENT_CYCLE_OVERRIDES")
    default_agent_cycle_seconds: int = Field(default=300, env="DEFAULT_AGENT_CYCLE_SECONDS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="/app/logs/trading_system.log", env="LOG_FILE")

    @model_validator(mode="before")
    @classmethod
    def _coerce_blank_env_entries(cls, data: Dict[str, object]):
        """Ensure empty-string env overrides do not clobber numeric/string defaults."""
        if not isinstance(data, dict):
            return data

        numeric_fields = {
            "qdrant_port",
            "memory_retrieve_limit",
            "memory_token_limit",
            "redis_port",
            "postgres_port",
            "redis_db",
            "agent_heartbeat_interval",
            "agent_timeout",
            "default_agent_cycle_seconds",
        }

        string_fields = {
            "qdrant_host",
            "ollama_url",
            "mcp_api_url",
            "dex_simulator_url",
        }

        for field_name in numeric_fields:
            value = data.get(field_name)
            if isinstance(value, str) and not value.strip():
                data.pop(field_name, None)

        for field_name in string_fields:
            value = data.get(field_name)
            if isinstance(value, str) and not value.strip():
                data.pop(field_name, None)

        return data
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def qdrant_url(self) -> str:
        """Construct Qdrant URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
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

    def get_agent_cycle_seconds(self, agent: Union[str, "AgentType"]) -> int:
        """Resolve the cycle interval (seconds) for an agent based on configured profiles and overrides."""
        try:
            # Lazy import to avoid circular dependency during settings initialization
            from core.models import AgentType  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover - fallback if models not ready
            AgentType = None  # type: ignore

        if AgentType is not None and isinstance(agent, AgentType):
            agent_key = agent.value.lower()
        else:
            agent_key = str(agent).lower()

        # Explicit env override takes precedence (e.g., AGENT_CYCLE_DQN=120)
        env_override = os.getenv(f"AGENT_CYCLE_{agent_key.upper()}")
        if env_override:
            try:
                return int(env_override)
            except ValueError:
                pass

        # JSON overrides via settings field
        if agent_key in self.agent_cycle_overrides:
            return int(self.agent_cycle_overrides[agent_key])

        # Profile-based defaults
        profile = self.agent_schedule_profiles.get(self.agent_schedule_profile, {})
        if agent_key in profile:
            return profile[agent_key]

        return self.default_agent_cycle_seconds
    


# Global settings instance
settings = Settings()

