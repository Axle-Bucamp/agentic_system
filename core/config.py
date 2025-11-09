"""
Configuration management for the Agentic Trading System.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pydantic import Field, ConfigDict, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from core.models import AgentType


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        extra="ignore",
        env_ignore_empty=True,
    )
    
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
    mcp_api_key: Optional[str] = Field(default="sk_jDHFvVDCU8bF4caeenG96jnKbYIET4wcDm3qBzNWXVc", env="MCP_API_KEY")
    dex_simulator_url: str = Field(default="http://localhost:8001", env="DEX_SIMULATOR_URL")
    
    # Blockscout MCP Configuration
    blockscout_mcp_url: Optional[str] = Field(
        default="http://blockscout-mcp:8080",
        env="BLOCKSCOUT_MCP_URL"
    )
    
    # Mock services
    use_mock_services: bool = Field(default=False, env="USE_MOCK_SERVICES")
    
    # Exchange API Keys
    mexc_api_key: Optional[str] = Field(default=None, env="MEXC_API_KEY")
    mexc_secret_key: Optional[str] = Field(default=None, env="MEXC_SECRET_KEY")
    
    # DEX Configuration
    private_key: Optional[str] = Field(default=None, env="PRIVATE_KEY")
    wallet_address: Optional[str] = Field(default=None, env="WALLET_ADDRESS")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL")
    openrouter_title: str = Field(default="Agentic Trading System", env="OPENROUTER_TITLE")
    openrouter_referer: Optional[str] = Field(default="http://localhost", env="OPENROUTER_REFERER")
    vllm_endpoint: Optional[str] = Field(default="http://localhost:8002/v1", env="VLLM_ENDPOINT")
    
    # CAMEL Configuration (default to stable Gemini 1.5 Pro for best compatibility)
    camel_default_model: str = Field(default="auto", env="CAMEL_DEFAULT_MODEL")
    camel_coordinator_model: str = Field(default="auto", env="CAMEL_COORDINATOR_MODEL")
    camel_task_model: str = Field(default="auto", env="CAMEL_TASK_MODEL")
    camel_worker_model: str = Field(default="auto", env="CAMEL_WORKER_MODEL")
    camel_primary_model: str = Field(default="openrouter_llama_4_maverick_free", env="CAMEL_PRIMARY_MODEL")
    camel_fallback_model: str = Field(default="openai/gpt-4o-mini", env="CAMEL_FALLBACK_MODEL")
    camel_prefer_gemini: bool = Field(default=False, env="CAMEL_PREFER_GEMINI")
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field(default="trading_memory", env="QDRANT_COLLECTION_NAME")

    # Neo4j Configuration
    neo4j_uri: Optional[str] = Field(default=None, env="NEO4J_URI")
    neo4j_user: Optional[str] = Field(default=None, env="NEO4J_USER")
    neo4j_password: Optional[str] = Field(default=None, env="NEO4J_PASSWORD")
    
    # Memory Configuration
    memory_chat_history_limit: int = Field(default=100, env="MEMORY_CHAT_HISTORY_LIMIT")
    memory_retrieve_limit: int = Field(default=3, env="MEMORY_RETRIEVE_LIMIT")
    memory_token_limit: int = Field(default=4096, env="MEMORY_TOKEN_LIMIT")
    memory_embedding_model: str = Field(default="nomic-embed-text", env="MEMORY_EMBEDDING_MODEL")
    memory_embedding_provider: str = Field(default="ollama", env="MEMORY_EMBEDDING_PROVIDER")  # "ollama" or "openai"
    memory_prune_limit: int = Field(default=100, env="MEMORY_PRUNE_LIMIT")
    memory_prune_similarity_threshold: float = Field(default=0.82, env="MEMORY_PRUNE_SIMILARITY_THRESHOLD")
    review_interval_hours: int = Field(default=24, env="REVIEW_INTERVAL_HOURS")
    review_prompt_default: str = Field(
        default="Review recent agent performance, adjust coordination weights to maximize risk-adjusted returns, and surface rationale for any changes. Keep weights normalized to 1.0.",
        env="REVIEW_PROMPT_DEFAULT",
    )
    news_llm_model: str = Field(default="openai/gpt-4o-mini", env="NEWS_LLM_MODEL")
    
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
    trade_reward_window_seconds: int = Field(default=3600, env="TRADE_REWARD_WINDOW_SECONDS")
    trade_reward_min_confidence: float = Field(default=0.55, env="TRADE_REWARD_MIN_CONFIDENCE")
    trade_reward_max_pending: int = Field(default=500, env="TRADE_REWARD_MAX_PENDING")
    trade_reward_price_source: str = Field(default="chart", env="TRADE_REWARD_PRICE_SOURCE")
    deep_search_api_url: Optional[str] = Field(default=None, env="DEEP_SEARCH_API_URL")
    deep_search_api_key: Optional[str] = Field(default=None, env="DEEP_SEARCH_API_KEY")
    deep_search_sources: List[str] = Field(
        default_factory=lambda: ["coindesk", "cointelegraph", "decrypt"]
    )
    news_source_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "yahoo_finance": 0.35,
            "coin_bureau": 0.25,
            "arxiv": 0.20,
            "google_scholar": 0.20,
        },
        env="NEWS_SOURCE_WEIGHTS",
    )
    arxiv_enabled: bool = Field(default=True, env="ARXIV_ENABLED")
    deep_research_mcp_url: Optional[str] = Field(default=None, env="DEEP_RESEARCH_MCP_URL")
    deep_research_depth: int = Field(default=2, env="DEEP_RESEARCH_DEPTH")
    deep_research_breadth: int = Field(default=2, env="DEEP_RESEARCH_BREADTH")
    deep_research_model: Optional[str] = Field(default=None, env="DEEP_RESEARCH_MODEL")
    deep_research_source_preferences: Optional[str] = Field(default=None, env="DEEP_RESEARCH_SOURCE_PREFERENCES")
    deep_research_timeout_seconds: int = Field(default=120, env="DEEP_RESEARCH_TIMEOUT_SECONDS")
    agent_instance_id: str = Field(
        default_factory=lambda: os.getenv("AGENT_INSTANCE_ID")
        or os.getenv("HOSTNAME")
        or "agent-instance-1",
        env="AGENT_INSTANCE_ID",
    )
    cluster_name: str = Field(default="local-cluster", env="CLUSTER_NAME")
    
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
    pipeline_live_defaults: Dict[str, Dict[str, Union[bool, str]]] = Field(
        default_factory=lambda: {
            "trend": {"enabled": False, "interval": "hours"},
            "fact": {"enabled": False, "interval": "hours"},
            "fusion": {"enabled": False, "interval": "hours"},
            "prune": {"enabled": True, "interval": "days"},
        }
    )
    pipeline_live_interval_seconds: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "trend": {"minutes": 300, "hours": 1800, "days": 10800},
            "fact": {"minutes": 900, "hours": 3600, "days": 14400},
            "fusion": {"minutes": 600, "hours": 1800, "days": 7200},
            "prune": {"minutes": 1800, "hours": 7200, "days": 86400},
        }
    )
    agent_cycle_overrides: Dict[str, int] = Field(default_factory=dict, env="AGENT_CYCLE_OVERRIDES")
    default_agent_cycle_seconds: int = Field(default=300, env="DEFAULT_AGENT_CYCLE_SECONDS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="/app/logs/trading_system.log", env="LOG_FILE")
    logfire_token: Optional[str] = Field(default=None, env="LOGFIRE_TOKEN")
    log_redis_enabled: bool = Field(default=True, env="LOG_REDIS_ENABLED")
    log_redis_list_key: str = Field(default="logs:recent", env="LOG_REDIS_LIST_KEY")
    log_redis_max_entries: int = Field(default=1000, env="LOG_REDIS_MAX_ENTRIES")

    @model_validator(mode="after")
    def _apply_api_key_aliases(self) -> "Settings":
        """Populate API keys from alternate environment variable names when provided."""
        if not self.gemini_api_key:
            alias = (
                os.getenv("GOOGLE_API_KEY")
                or os.getenv("GOOGLE_STUDIO_API_KEY")
                or os.getenv("GOOGLE_GENAI_API_KEY")
                or os.getenv("GEMINI_APIKEY")
                or os.getenv("GEMINI_API_KEY")
            )
            if alias:
                self.gemini_api_key = alias.strip()

        if not self.openrouter_api_key:
            alias = os.getenv("OPEN_ROUTEUR_API")
            if alias:
                self.openrouter_api_key = alias.strip()

        if not self.openai_api_key and self.openrouter_api_key:
            self.openai_api_key = self.openrouter_api_key

        if self.openrouter_api_key and not self.openai_base_url:
            self.openai_base_url = self.openrouter_base_url
        if self.openrouter_api_key and not self.openrouter_referer:
            self.openrouter_referer = "https://openrouter.ai"

        # If no Gemini key available, ensure CAMEL defaults fall back to GPT
        if not self.gemini_api_key:
            self.camel_prefer_gemini = False
            if not (self.camel_primary_model.lower().startswith("gpt") or self.camel_primary_model.lower().startswith("openrouter")):
                self.camel_primary_model = "gpt-4o-mini"
            if not (self.camel_fallback_model.lower().startswith("gpt") or self.camel_fallback_model.lower().startswith("openrouter")):
                self.camel_fallback_model = "gpt-4o-mini"
        if not self.openrouter_api_key:
            if self.camel_primary_model.lower().startswith("openrouter"):
                self.camel_primary_model = "gpt-4o-mini"
            if self.camel_fallback_model.lower().startswith("openrouter"):
                self.camel_fallback_model = "gpt-4o-mini"

        return self

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
            "deep_research_depth",
            "deep_research_breadth",
            "deep_research_timeout_seconds",
        }

        string_fields = {
            "qdrant_host",
            "ollama_url",
            "mcp_api_url",
            "dex_simulator_url",
            "deep_search_api_url",
            "deep_research_mcp_url",
            "deep_research_model",
            "deep_research_source_preferences",
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

    @property
    def openai_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.openrouter_referer:
            headers["HTTP-Referer"] = self.openrouter_referer
        if self.openrouter_title:
            headers["X-Title"] = self.openrouter_title
        return headers

    @property
    def openai_client_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        headers = self.openai_headers
        if headers:
            kwargs["default_headers"] = headers
        return kwargs

    @property
    def neo4j_enabled(self) -> bool:
        """Determine whether Neo4j integration is configured."""
        return all([self.neo4j_uri, self.neo4j_user, self.neo4j_password])
    
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

