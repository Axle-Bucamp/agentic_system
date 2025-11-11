"""
CAMEL Model Configuration Factory

Supports multiple model providers: OpenAI, Gemini, and local models.
"""
from pathlib import Path
import sys
from typing import Optional, Dict, Any, Callable
import httpx
from core.config import settings
from core.logging import log

# Allow bundling the upstream CAMEL repository alongside this project
_local_camel_repo = Path(__file__).resolve().parents[2] / "camel"
if _local_camel_repo.exists():
    repo_path = str(_local_camel_repo)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

try:
    from camel.types import ModelType, ModelPlatformType
    from camel.models import ModelFactory
    from camel.configs import GeminiConfig, OpenRouterConfig
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    log.warning("CAMEL-AI not available. Install with: pip install camel-ai")
    GeminiConfig = None
    OpenRouterConfig = None


class CamelModelFactory:
    """Factory for creating CAMEL-compatible model instances."""
    
    _model_cache: Dict[str, Any] = {}
    _fallback_notice_logged: bool = False
    _gemini_status_checked: bool = False
    _gemini_key_valid: bool = False
    
    @classmethod
    def get_model_type(cls, model_name: str) -> Optional[ModelType]:
        """Convert model name string to CAMEL ModelType enum."""
        if not CAMEL_AVAILABLE:
            return None
        
        original_name = model_name.lower()
        normalized_name = original_name
        if normalized_name.startswith("openrouter"):
            normalized_name = normalized_name.split("/", 1)[-1] if "/" in normalized_name else normalized_name[len("openrouter"):].lstrip(":/")
        if normalized_name.startswith("openai/"):
            normalized_name = normalized_name.split("/", 1)[-1]
        if normalized_name.startswith("anthropic/"):
            normalized_name = normalized_name.split("/", 1)[-1]
        if normalized_name.startswith("google/"):
            normalized_name = normalized_name.split("/", 1)[-1]

        normalized_key = normalized_name.replace("/", "_").replace("-", "_")

        model_map: Dict[str, ModelType] = {}

        base_map = {
            "gpt-4o-mini": "GPT_4O_MINI",
            "gpt-4o": "GPT_4O",
            "gpt-4-turbo": "GPT_4_TURBO",
            "gpt-3.5-turbo": "GPT_3_5_TURBO",
            "claude-3-opus": "CLAUDE_3_OPUS",
            "claude-3-sonnet": "CLAUDE_3_SONNET",
            "claude-3-haiku": "CLAUDE_3_HAIKU",
            "gemini-pro": "GEMINI_1_5_PRO",
            "gemini-1.5-pro": "GEMINI_1_5_PRO",
            "gemini-1.5-flash": "GEMINI_1_5_FLASH",
            "gemini-2.0-flash": "GEMINI_2_0_FLASH",
            "gemini-2.5-pro": "GEMINI_2_5_PRO",
            "gemini-2.5-flash": "GEMINI_2_5_FLASH",
            "gemini-2.0-flash-thinking-exp": "GEMINI_2_0_FLASH_THINKING_EXP",
            "gemini-2.0-flash-lite": "GEMINI_2_0_FLASH_LITE",
            "gemini-pro-vision": "GEMINI_1_5_PRO",
            "openrouter_llama_3_1_405b": "OPENROUTER_LLAMA_3_1_405B",
            "openrouter_llama_3_1_70b": "OPENROUTER_LLAMA_3_1_70B",
            "openrouter_llama_4_maverick": "OPENROUTER_LLAMA_4_MAVERICK",
            "openrouter_llama_4_maverick_free": "OPENROUTER_LLAMA_4_MAVERICK_FREE",
            "openrouter_llama_4_scout": "OPENROUTER_LLAMA_4_SCOUT",
            "openrouter_olympicoder_7b": "OPENROUTER_OLYMPICODER_7B",
            "openrouter_horizon_alpha": "OPENROUTER_HORIZON_ALPHA",
        }

        for name, attr in base_map.items():
            enum_value = getattr(ModelType, attr, None)
            if enum_value:
                model_map[name] = enum_value

        candidate_keys = [
            normalized_name,
            normalized_key,
            f"openrouter_{normalized_name}",
            f"openrouter_{normalized_key}",
        ]

        for key in candidate_keys:
            if key in model_map:
                return model_map[key]

        return None
    
    @classmethod
    def create_model(
        cls,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create a CAMEL model instance.
        
        Args:
            model_name: Model name (defaults to settings.camel_default_model)
            api_key: API key for the model (defaults to OpenAI key from settings)
            **kwargs: Additional model-specific parameters
            
        Returns:
            CAMEL model instance
        """
        if not CAMEL_AVAILABLE:
            raise ImportError("CAMEL-AI is not installed. Install with: pip install camel-ai")
        
        model_name = cls._resolve_model_name(model_name or settings.camel_default_model)
        cache_key = f"{model_name}_{api_key or 'default'}"
        
        # Return cached model if available
        if cache_key in cls._model_cache:
            return cls._model_cache[cache_key]
        
        model_type = cls.get_model_type(model_name)
        if not model_type:
            log.warning(f"Unknown model type: {model_name}, using default")
            model_type = ModelType.GPT_4O_MINI
        
        # Determine API key based on model type
        if not api_key:
            api_key = cls._resolve_api_key(model_name)
            if not api_key:
                raise ValueError(f"No API key configured for resolved model '{model_name}'")
        
        if not api_key:
            raise ValueError(f"No API key provided for model {model_name}")
        
        def _build_model(
            selected_model_type: ModelType,
            selected_platform: ModelPlatformType,
            selected_key: str,
            selected_config: Optional[Dict[str, Any]] = None,
        ):
            return ModelFactory.create(
                model_platform=selected_platform,
                model_type=selected_model_type,
                api_key=selected_key,
                model_config_dict=selected_config,
            )

        try:
            model_platform = cls._resolve_platform(model_name)
            model_config = kwargs if kwargs else cls._default_model_config(model_platform)

            if model_platform == ModelPlatformType.GEMINI:
                if not cls._ensure_gemini_available():
                    log.warning(
                        "Gemini API unavailable; falling back to '%s' for CAMEL model.",
                        settings.camel_fallback_model,
                    )
                    fallback_name = settings.camel_fallback_model or "gpt-4o-mini"
                    fallback_key = cls._resolve_api_key(fallback_name)
                    fallback_platform = cls._resolve_platform(fallback_name)
                    model_type = cls.get_model_type(fallback_name) or ModelType.GPT_4O_MINI
                    fallback_config = cls._default_model_config(fallback_platform)
                    model = _build_model(model_type, fallback_platform, fallback_key, fallback_config)
                    cls._model_cache[cache_key] = model
                    return model

            model = _build_model(model_type, model_platform, api_key, model_config)

            cls._model_cache[cache_key] = model
            log.info(f"Created CAMEL model: {model_name}")
            return model

        except Exception as e:
            error_msg = str(e)
            log.error(f"Failed to create model {model_name}: {error_msg}")

            fallback = cls._attempt_immediate_fallback(
                cache_key=cache_key,
                build_fn=_build_model,
                original_error=error_msg,
                model_name=model_name,
            )
            if fallback:
                return fallback

            raise

    @classmethod
    def create_coordinator_model(cls) -> Any:
        """Create model for coordinator agents in CAMEL workforce."""
        model_name = settings.camel_coordinator_model or settings.camel_default_model
        api_key = cls._resolve_api_key(model_name)
        return cls.create_model(model_name=model_name, api_key=api_key)

    @classmethod
    def create_task_model(cls) -> Any:
        """Create model for task decomposition agents."""
        model_name = settings.camel_task_model or settings.camel_default_model
        api_key = cls._resolve_api_key(model_name)
        return cls.create_model(model_name=model_name, api_key=api_key)

    @classmethod
    def create_worker_model(cls) -> Any:
        """Create model for workforce workers."""
        model_name = settings.camel_worker_model or settings.camel_default_model
        api_key = cls._resolve_api_key(model_name)
        return cls.create_model(model_name=model_name, api_key=api_key)

    @classmethod
    def clear_cache(cls) -> None:
        cls._model_cache.clear()
        log.info("CAMEL model cache cleared")

    @classmethod
    def _resolve_model_name(cls, requested: str) -> str:
        candidate = (requested or "").strip() or "auto"
        if candidate.lower() != "auto":
            return candidate

        priorities = []
        primary = settings.camel_primary_model.strip() if settings.camel_primary_model else ""
        fallback = settings.camel_fallback_model.strip() if settings.camel_fallback_model else ""

        if settings.camel_prefer_gemini and primary:
            priorities.append(primary)
            if fallback:
                priorities.append(fallback)
        else:
            if fallback:
                priorities.append(fallback)
            if primary:
                priorities.append(primary)

        for name in priorities:
            lowered = name.lower()
            if lowered.startswith("gemini") and not settings.gemini_api_key:
                continue
            if (lowered.startswith("openai/") or lowered.startswith("openai")) and not settings.openai_api_key:
                continue
            if lowered.startswith("openrouter") and not settings.openrouter_api_key:
                if not settings.openai_api_key:
                    continue
            if lowered.startswith("gpt") and not settings.openai_api_key:
                continue
            return name

        raise ValueError(
            "No CAMEL model can be resolved. Configure GEMINI_API_KEY or OPENAI_API_KEY (or disable camel_prefer_gemini)."
        )

    @classmethod
    def _resolve_api_key(cls, model_name: str) -> Optional[str]:
        lowered = model_name.lower()
        if lowered.startswith("openrouter") or "openrouter/" in lowered:
            return settings.openrouter_api_key or settings.openai_api_key
        if lowered.startswith("openai/"):
            return settings.openai_api_key
        if lowered.startswith("gemini"):
            return settings.gemini_api_key
        if lowered.startswith("gpt") or lowered.startswith("claude"):
            return settings.openai_api_key
        return settings.openai_api_key

    @classmethod
    def _resolve_platform(cls, model_name: str) -> ModelPlatformType:
        lowered = model_name.lower()
        if lowered.startswith("openrouter") or "openrouter/" in lowered:
            return ModelPlatformType.OPENROUTER
        if lowered.startswith("openai/"):
            return ModelPlatformType.OPENAI
        if lowered.startswith("gemini"):
            return ModelPlatformType.GEMINI
        if lowered.startswith("claude"):
            return ModelPlatformType.ANTHROPIC
        return ModelPlatformType.OPENAI

    @classmethod
    def _attempt_immediate_fallback(
        cls,
        cache_key: str,
        build_fn: Callable[[ModelType, ModelPlatformType, str, Optional[Dict[str, Any]]], Any],
        original_error: str,
        model_name: str,
    ) -> Optional[Any]:
        fallback_name = settings.camel_fallback_model
        fallback_key = settings.openai_api_key
        if fallback_key and fallback_name:
            try:
                fallback_platform = cls._resolve_platform(fallback_name)
                fallback_config = cls._default_model_config(fallback_platform)
                fallback_model = build_fn(
                    cls.get_model_type(fallback_name) or ModelType.GPT_4O_MINI,
                    fallback_platform,
                    fallback_key,
                    fallback_config,
                )
                cls._model_cache[f"fallback_{cache_key}"] = fallback_model
                log.warning(
                    "Primary model '%s' unavailable (%s); using fallback '%s'",
                    model_name,
                    original_error,
                    fallback_name,
                )
                cls._fallback_notice_logged = True
                return fallback_model
            except Exception as fallback_error:
                log.error("Immediate fallback creation failed: %s", fallback_error)
        return None

    @staticmethod
    def _default_model_config(platform: ModelPlatformType) -> Optional[Dict[str, Any]]:
        if platform == ModelPlatformType.GEMINI and GeminiConfig:
            return GeminiConfig(temperature=0.2).as_dict()

        if platform == ModelPlatformType.OPENROUTER and OpenRouterConfig:
            return OpenRouterConfig(temperature=0.2).as_dict()

        if platform == ModelPlatformType.OPENAI:
            config: Dict[str, Any] = {"temperature": 0.2}
            return config

        return None

    @classmethod
    def _ensure_gemini_available(cls) -> bool:
        if cls._gemini_status_checked:
            return cls._gemini_key_valid

        cls._gemini_status_checked = True
        api_key = settings.gemini_api_key
        if not api_key:
            cls._gemini_key_valid = False
            return False

        url = "https://generativelanguage.googleapis.com/v1beta/models"
        try:
            response = httpx.get(url, params={"key": api_key}, timeout=8.0)
            if response.status_code == 200 and response.json().get("models"):
                cls._gemini_key_valid = True
            else:
                log.warning(
                    "Gemini models API returned %s: %s",
                    response.status_code,
                    response.text[:200],
                )
                cls._gemini_key_valid = False
        except Exception as exc:
            log.warning("Unable to validate Gemini API key: %s", exc)
            cls._gemini_key_valid = False

        return cls._gemini_key_valid
