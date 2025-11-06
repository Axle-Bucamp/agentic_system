"""
Embedding Model Configuration for CAMEL Memory

Supports OpenAI embeddings and alternative embedding models.
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


class EmbeddingFactory:
    """Factory for creating embedding model instances."""
    
    _embedding_cache: Optional[BaseEmbedding] = None
    
    @classmethod
    def create_embedding(
        cls,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseEmbedding:
        """
        Create an embedding model instance.
        
        Args:
            model_name: Embedding model name (defaults to settings.memory_embedding_model)
            api_key: OpenAI API key (defaults to settings.openai_api_key)
            
        Returns:
            BaseEmbedding instance
        """
        if not EMBEDDING_AVAILABLE:
            raise ImportError("CAMEL embeddings not installed. Install with: pip install camel-ai")
        
        if cls._embedding_cache is not None:
            return cls._embedding_cache
        
        model_name = model_name or settings.memory_embedding_model
        api_key = api_key or settings.openai_api_key
        
        if not api_key:
            log.warning("No OpenAI API key provided for embeddings")
        
        try:
            # Currently CAMEL primarily supports OpenAI embeddings
            # Support for other providers can be added here
            if "openai" in model_name.lower() or "text-embedding" in model_name.lower():
                embedding = OpenAIEmbedding(model=model_name)
                if api_key:
                    embedding.api_key = api_key
            else:
                # Default to OpenAI
                log.warning(f"Unknown embedding model {model_name}, using OpenAI default")
                embedding = OpenAIEmbedding()
                if api_key:
                    embedding.api_key = api_key
            
            cls._embedding_cache = embedding
            log.info(f"Created embedding model: {model_name}")
            
            return embedding
            
        except Exception as e:
            log.error(f"Failed to create embedding model: {e}")
            raise
    
    @classmethod
    def get_output_dim(cls, model_name: Optional[str] = None) -> int:
        """
        Get the output dimension for an embedding model.
        
        Args:
            model_name: Embedding model name
            
        Returns:
            Output dimension (vector size)
        """
        model_name = model_name or settings.memory_embedding_model
        
        # Known dimensions for common models
        dim_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        for key, dim in dim_map.items():
            if key in model_name.lower():
                return dim
        
        # Default dimension
        return 1536
    
    @classmethod
    def clear_cache(cls):
        """Clear the embedding cache."""
        cls._embedding_cache = None
        log.info("Embedding cache cleared")

