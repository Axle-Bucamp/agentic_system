"""
CAMEL Model Configuration Factory

Supports multiple model providers: OpenAI, Gemini, and local models.
"""
from typing import Optional, Dict, Any, Callable
from core.config import settings
from core.logging import log

try:
    from camel.types import ModelType, ModelPlatformType
    from camel.models import ModelFactory
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    log.warning("CAMEL-AI not available. Install with: pip install camel-ai")


class CamelModelFactory:
    """Factory for creating CAMEL-compatible model instances."""
    
    _model_cache: Dict[str, Any] = {}
    _fallback_notice_logged: bool = False
    
    @classmethod
    def get_model_type(cls, model_name: str) -> Optional[ModelType]:
        """Convert model name string to CAMEL ModelType enum."""
        if not CAMEL_AVAILABLE:
            return None
        
        # Map common model names to CAMEL ModelType
        model_map = {
            "gpt-4o-mini": ModelType.GPT_4O_MINI,
            "gpt-4o": ModelType.GPT_4O,
            "gpt-4-turbo": ModelType.GPT_4_TURBO,
            "gpt-3.5-turbo": ModelType.GPT_3_5_TURBO,
            "claude-3-opus": ModelType.CLAUDE_3_OPUS,
            "claude-3-sonnet": ModelType.CLAUDE_3_SONNET,
            "claude-3-haiku": ModelType.CLAUDE_3_HAIKU,
            # Gemini models - using latest available versions
            "gemini-pro": ModelType.GEMINI_1_5_PRO,
            "gemini-1.5-pro": ModelType.GEMINI_1_5_PRO,
            "gemini-1.5-flash": ModelType.GEMINI_1_5_FLASH,
            "gemini-2.0-flash": ModelType.GEMINI_2_0_FLASH,
            "gemini-2.5-pro": ModelType.GEMINI_2_5_PRO_PREVIEW,
            "gemini-pro-vision": ModelType.GEMINI_1_5_PRO,  # Fallback to 1.5 Pro
        }
        
        return model_map.get(model_name.lower())
    
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
        
        def _build_model(selected_model_type: ModelType, selected_platform: ModelPlatformType, selected_key: str):
            return ModelFactory.create(
                model_platform=selected_platform,
                model_type=selected_model_type,
                api_key=selected_key,
                model_config_dict=kwargs if kwargs else None,
            )

        try:
            model_platform = cls._resolve_platform(model_name)
            model = _build_model(model_type, model_platform, api_key)

            cls._model_cache[cache_key] = model
            log.info(f"Created CAMEL model: {model_name}")
            return model

        except Exception as e:
            error_msg = str(e)
            log.error(f"Failed to create model {model_name}: {error_msg}")

            if "gemini" in model_name.lower():
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
            if lowered.startswith("gpt") and not settings.openai_api_key:
                continue
            return name

        raise ValueError(
            "No CAMEL model can be resolved. Configure GEMINI_API_KEY or OPENAI_API_KEY (or disable camel_prefer_gemini)."
        )

    @classmethod
    def _resolve_api_key(cls, model_name: str) -> Optional[str]:
        lowered = model_name.lower()
        if lowered.startswith("gemini"):
            return settings.gemini_api_key
        if lowered.startswith("gpt") or lowered.startswith("claude"):
            return settings.openai_api_key
        return settings.openai_api_key

    @classmethod
    def _resolve_platform(cls, model_name: str) -> ModelPlatformType:
        lowered = model_name.lower()
        if lowered.startswith("gemini"):
            return ModelPlatformType.GEMINI
        if lowered.startswith("claude"):
            return ModelPlatformType.ANTHROPIC
        return ModelPlatformType.OPENAI

    @classmethod
    def _should_prepare_fallback(cls, model_name: str) -> bool:
        return (
            "gemini" in model_name.lower()
            and bool(settings.camel_fallback_model)
            and bool(settings.openai_api_key)
        )

    @classmethod
    def _build_fallback_factory(cls, fallback_name: str, build_fn: Callable[[ModelType, ModelPlatformType, str], Any]) -> Optional[Callable[[], Any]]:
        if not fallback_name:
            return None

        fallback_type = cls.get_model_type(fallback_name) or ModelType.GPT_4O_MINI
        fallback_platform = cls._resolve_platform(fallback_name)
        fallback_key = cls._resolve_api_key(fallback_name)
        if not fallback_key:
            return None

        def factory() -> Any:
            log.warning(
                "Switching CAMEL model to fallback '%s' after primary failures.",
                fallback_name,
            )
            return build_fn(fallback_type, fallback_platform, fallback_key)

        return factory

    @classmethod
    def _attempt_immediate_fallback(
        cls,
        cache_key: str,
        build_fn: Callable[[ModelType, ModelPlatformType, str], Any],
        original_error: str,
        model_name: str,
    ) -> Optional[Any]:
        fallback_name = settings.camel_fallback_model
        fallback_key = settings.openai_api_key
        if fallback_key and fallback_name:
            try:
                fallback_model = build_fn(
                    cls.get_model_type(fallback_name) or ModelType.GPT_4O_MINI,
                    cls._resolve_platform(fallback_name),
                    fallback_key,
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
