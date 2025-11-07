"""
CAMEL Model Configuration Factory

Supports multiple model providers: OpenAI, Gemini, and local models.
"""
from typing import Optional, Dict, Any
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
        
        model_name = model_name or settings.camel_default_model
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
            if "gemini" in model_name.lower():
                api_key = settings.gemini_api_key
                if not api_key:
                    log.warning(f"No Gemini API key found, falling back to OpenAI")
                    model_type = ModelType.GPT_4O_MINI
                    api_key = settings.openai_api_key
            else:
                api_key = settings.openai_api_key
        
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
            # Determine model platform based on model type
            if "gemini" in model_name.lower():
                model_platform = ModelPlatformType.GEMINI
                model_config = kwargs if kwargs else None
            elif "gpt" in model_name.lower() or "claude" in model_name.lower():
                model_platform = ModelPlatformType.OPENAI if "gpt" in model_name.lower() else ModelPlatformType.ANTHROPIC
                model_config = kwargs if kwargs else None
            else:
                # Default to OpenAI platform
                model_platform = ModelPlatformType.OPENAI
                model_config = kwargs if kwargs else None

            model = _build_model(model_type, model_platform, api_key)

            cls._model_cache[cache_key] = model
            log.info(f"Created CAMEL model: {model_name}")
            return model

        except Exception as e:
            error_msg = str(e)
            log.error(f"Failed to create model {model_name}: {error_msg}")

            # Attempt graceful fallback for Gemini or other remote providers
            if "gemini" in model_name.lower():
                fallback_key = settings.openai_api_key
                if fallback_key:
                    try:
                        fallback_model = _build_model(
                            ModelType.GPT_4O_MINI,
                            ModelPlatformType.OPENAI,
                            fallback_key,
                        )
                        fallback_cache_key = f"fallback_{cache_key}"
                        cls._model_cache[fallback_cache_key] = fallback_model
                        log.warning(
                            "Gemini model '%s' unavailable (%s); using OpenAI fallback '%s'",
                            model_name,
                            error_msg,
                            ModelType.GPT_4O_MINI.value if hasattr(ModelType.GPT_4O_MINI, "value") else "gpt-4o-mini",
                        )
                        return fallback_model
                    except Exception as fallback_error:
                        log.error("Fallback OpenAI model creation failed: %s", fallback_error)

            raise
    
    @classmethod
    def create_coordinator_model(cls) -> Any:
        """Create model for coordinator agent."""
        model_name = settings.camel_coordinator_model
        # Use Gemini API key if model is Gemini, otherwise OpenAI
        api_key = settings.gemini_api_key if "gemini" in model_name.lower() else settings.openai_api_key
        return cls.create_model(
            model_name=model_name,
            api_key=api_key
        )
    
    @classmethod
    def create_task_model(cls) -> Any:
        """Create model for task decomposition agent."""
        model_name = settings.camel_task_model
        # Use Gemini API key if model is Gemini, otherwise OpenAI
        api_key = settings.gemini_api_key if "gemini" in model_name.lower() else settings.openai_api_key
        return cls.create_model(
            model_name=model_name,
            api_key=api_key
        )
    
    @classmethod
    def create_worker_model(cls) -> Any:
        """Create model for worker agents."""
        model_name = settings.camel_worker_model
        # Use Gemini API key if model is Gemini, otherwise OpenAI
        api_key = settings.gemini_api_key if "gemini" in model_name.lower() else settings.openai_api_key
        return cls.create_model(
            model_name=model_name,
            api_key=api_key
        )
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache."""
        cls._model_cache.clear()
        log.info("Model cache cleared")

