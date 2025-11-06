"""
Ollama Embedding Client for lightweight embeddings.

Uses Ollama API to generate embeddings without requiring PyTorch or large models.
"""
from typing import List, Optional
import httpx
from core.config import settings
from core.logging import log

try:
    from camel.embeddings import BaseEmbedding
    EMBEDDING_BASE_AVAILABLE = True
except ImportError:
    EMBEDDING_BASE_AVAILABLE = False
    # Create a minimal BaseEmbedding interface if CAMEL is not available
    class BaseEmbedding:
        """Minimal BaseEmbedding interface."""
        pass


class OllamaEmbedding(BaseEmbedding):
    """Ollama-based embedding client compatible with CAMEL BaseEmbedding interface."""
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Ollama embedding client.
        
        Args:
            model: Ollama model name (default: nomic-embed-text, ~137MB)
            base_url: Ollama API base URL (defaults to settings.ollama_url)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = (base_url or settings.ollama_url).rstrip('/')
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        log.info(f"Initialized OllamaEmbedding with model: {model}, URL: {self.base_url}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                follow_redirects=True
            )
        return self._client
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        client = await self._get_client()
        
        try:
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if "embedding" not in data:
                raise ValueError(f"Invalid response from Ollama: {data}")
            
            embedding = data["embedding"]
            log.debug(f"Generated embedding of dimension {len(embedding)} for text: {text[:50]}...")
            return embedding
            
        except httpx.HTTPError as e:
            log.error(f"HTTP error generating embedding: {e}")
            raise
        except Exception as e:
            log.error(f"Error generating embedding: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Ollama doesn't support batch embeddings natively, so we use asyncio.gather
        # for parallel requests (Ollama can handle multiple concurrent requests)
        import asyncio
        tasks = [self.embed(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        return list(embeddings)
    
    async def embed_list(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts (alias for embed_batch).
        
        This method is required by CAMEL's BaseEmbedding interface.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        return await self.embed_batch(texts)
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the embedding model.
        
        Returns:
            Dimension of the embedding vector
        """
        # nomic-embed-text has 768 dimensions
        # Other common Ollama embedding models:
        # - nomic-embed-text: 768
        # - mxbai-embed-large: 1024
        dim_map = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
        }
        return dim_map.get(self.model, 768)  # Default to 768
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    def __del__(self):
        """Cleanup on deletion."""
        # Note: This won't work for async cleanup, but it's a fallback
        if self._client is not None:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._client.aclose())
                else:
                    loop.run_until_complete(self._client.aclose())
            except Exception:
                pass

