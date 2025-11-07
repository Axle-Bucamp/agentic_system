"""
Qdrant Storage Configuration for CAMEL Memory

Integrates Qdrant vector database with CAMEL's storage interface.
"""
from typing import Optional, Dict, Any
from core.config import settings
from core.logging import log

try:
    from camel.storages import QdrantStorage
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    log.warning("Qdrant dependencies not available. Install with: pip install qdrant-client camel-ai")


class QdrantStorageFactory:
    """Factory for creating Qdrant storage instances."""
    
    _client: Optional[QdrantClient] = None
    _storage_cache: Dict[str, Any] = {}
    
    @classmethod
    def get_client(cls) -> QdrantClient:
        """Get or create Qdrant client."""
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant dependencies not installed")
        
        if cls._client is None:
            configured_host = (settings.qdrant_host or "").strip()
            candidate_hosts = []

            if configured_host:
                candidate_hosts.append(configured_host)

            # Add sensible fallbacks for containerised and local development environments
            for fallback in ("qdrant", "localhost"):
                if fallback not in candidate_hosts:
                    candidate_hosts.append(fallback)

            last_error: Optional[Exception] = None
            for host in candidate_hosts:
                try:
                    client = QdrantClient(
                        host=host,
                        port=settings.qdrant_port,
                        timeout=30.0,
                    )
                    # Perform a lightweight request to ensure the connection succeeds
                    client.get_collections()
                    cls._client = client
                    log.info("Connected to Qdrant at http://%s:%s", host, settings.qdrant_port)
                    break
                except Exception as exc:  # pragma: no cover - network failures
                    last_error = exc
                    log.debug("Attempt to connect to Qdrant at %s:%s failed: %s", host, settings.qdrant_port, exc)

            if cls._client is None:
                error_message = f"Failed to connect to Qdrant using hosts {candidate_hosts}: {last_error}"
                log.error(error_message)
                raise RuntimeError(error_message) from last_error
        
        return cls._client
    
    @classmethod
    def create_storage(
        cls,
        collection_name: Optional[str] = None,
        vector_dim: int = 1536
    ) -> QdrantStorage:
        """
        Create a QdrantStorage instance for CAMEL.
        
        Args:
            collection_name: Name of the Qdrant collection
            vector_dim: Dimension of the embedding vectors
            
        Returns:
            QdrantStorage instance
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant dependencies not installed")
        
        collection_name = collection_name or settings.qdrant_collection_name
        cache_key = f"{collection_name}_{vector_dim}"
        
        if cache_key in cls._storage_cache:
            return cls._storage_cache[cache_key]
        
        try:
            client = cls.get_client()
            
            # Ensure collection exists
            try:
                client.get_collection(collection_name)
            except Exception:
                # Collection doesn't exist, create it
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_dim,
                        distance=Distance.COSINE
                    )
                )
                log.info(f"Created Qdrant collection: {collection_name}")
            
            # Create CAMEL storage wrapper
            storage = QdrantStorage(
                collection_name=collection_name,
                vector_dim=vector_dim,
                client=client
            )
            
            cls._storage_cache[cache_key] = storage
            log.info(f"Created QdrantStorage for collection: {collection_name}")
            
            return storage
            
        except Exception as e:
            log.error(f"Failed to create QdrantStorage: {e}")
            raise
    
    @classmethod
    def ensure_collection_exists(cls, collection_name: str, vector_dim: int = 1536):
        """Ensure a Qdrant collection exists, create if not."""
        if not QDRANT_AVAILABLE:
            return
        
        try:
            client = cls.get_client()
            try:
                client.get_collection(collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_dim,
                        distance=Distance.COSINE
                    )
                )
                log.info(f"Created Qdrant collection: {collection_name}")
        except Exception as e:
            log.error(f"Failed to ensure collection exists: {e}")
    
    @classmethod
    def clear_cache(cls):
        """Clear the storage cache."""
        cls._storage_cache.clear()
        log.info("Storage cache cleared")

