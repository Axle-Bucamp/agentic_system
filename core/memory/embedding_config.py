"""
Embedding Model Configuration for CAMEL Memory

Supports OpenAI embeddings and Ollama embeddings (lightweight, no PyTorch required).
"""
from typing import Optional
from core.config import settings
from core.logging import log

try:
    from camel.embeddings import OpenAIEmbedding, BaseEmbedding
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    log.warning("CAMEL embeddings not available. Install with: pip install camel-ai")

# Import Ollama embedding
try:
    from core.memory.ollama_embedding import OllamaEmbedding
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    log.warning("Ollama embedding not available")


class EmbeddingFactory:
    """Factory for creating embedding model instances."""
    
    _embedding_cache: Optional[BaseEmbedding] = None
    
    @classmethod
    def create_embedding(
        cls,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None
    ) -> BaseEmbedding:
        """
        Create an embedding model instance.
        
        Args:
            model_name: Embedding model name (defaults to settings.memory_embedding_model)
            api_key: OpenAI API key (defaults to settings.openai_api_key, only used for OpenAI)
            provider: Embedding provider ("ollama" or "openai", defaults to settings.memory_embedding_provider)
            
        Returns:
            BaseEmbedding instance
        """
        if cls._embedding_cache is not None:
            return cls._embedding_cache
        
        provider = provider or settings.memory_embedding_provider
        model_name = model_name or settings.memory_embedding_model
        
        try:
            # Use Ollama for lightweight embeddings (no PyTorch required)
            if provider.lower() == "ollama":
                if not OLLAMA_AVAILABLE:
                    raise ImportError("Ollama embedding not available. Check core.memory.ollama_embedding")
                
                embedding = OllamaEmbedding(
                    model=model_name or settings.ollama_model,
                    base_url=settings.ollama_url
                )
                log.info(f"Created Ollama embedding model: {model_name or settings.ollama_model}")
            
            # Use OpenAI embeddings (requires API key)
            elif provider.lower() == "openai":
                if not EMBEDDING_AVAILABLE:
                    raise ImportError("CAMEL embeddings not installed. Install with: pip install camel-ai")
                
                api_key = api_key or settings.openai_api_key
                if not api_key:
                    log.warning("No OpenAI API key provided for embeddings, falling back to Ollama")
                    if OLLAMA_AVAILABLE:
                        embedding = OllamaEmbedding(
                            model=settings.ollama_model,
                            base_url=settings.ollama_url
                        )
                        log.info(f"Fell back to Ollama embedding: {settings.ollama_model}")
                    else:
                        raise ValueError("No OpenAI API key and Ollama not available")
                else:
                    if "openai" in model_name.lower() or "text-embedding" in model_name.lower():
                        embedding = OpenAIEmbedding(model=model_name)
                    else:
                        embedding = OpenAIEmbedding()
                    embedding.api_key = api_key
                    log.info(f"Created OpenAI embedding model: {model_name}")
            
            else:
                raise ValueError(f"Unknown embedding provider: {provider}. Use 'ollama' or 'openai'")
            
            cls._embedding_cache = embedding
            return embedding
            
        except Exception as e:
            log.error(f"Failed to create embedding model: {e}")
            raise
    
    @classmethod
    def get_output_dim(cls, model_name: Optional[str] = None, provider: Optional[str] = None) -> int:
        """
        Get the output dimension for an embedding model.
        
        Args:
            model_name: Embedding model name
            provider: Embedding provider ("ollama" or "openai")
            
        Returns:
            Output dimension (vector size)
        """
        provider = provider or settings.memory_embedding_provider
        model_name = model_name or settings.memory_embedding_model
        
        # Ollama model dimensions
        if provider.lower() == "ollama":
            ollama_dim_map = {
                "nomic-embed-text": 768,
                "mxbai-embed-large": 1024,
            }
            for key, dim in ollama_dim_map.items():
                if key in model_name.lower():
                    return dim
            return 768  # Default for Ollama models
        
        # OpenAI model dimensions
        elif provider.lower() == "openai":
            openai_dim_map = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            }
            for key, dim in openai_dim_map.items():
                if key in model_name.lower():
                    return dim
            return 1536  # Default for OpenAI models
        
        # Fallback
        return 768
    
    @classmethod
    def clear_cache(cls):
        """Clear the embedding cache."""
        cls._embedding_cache = None
        log.info("Embedding cache cleared")

