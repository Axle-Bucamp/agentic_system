"""
Unit tests for CAMEL memory system.
"""
import pytest
from unittest.mock import MagicMock, patch
from core.memory.camel_memory_manager import CamelMemoryManager
from core.memory.qdrant_storage import QdrantStorageFactory
from core.memory.embedding_config import EmbeddingFactory


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = MagicMock()
    client.get_collection = MagicMock(side_effect=Exception("Collection not found"))
    client.create_collection = MagicMock()
    return client


@pytest.fixture
def mock_embedding():
    """Mock embedding model."""
    embedding = MagicMock()
    embedding.get_output_dim = MagicMock(return_value=1536)
    return embedding


@patch('core.memory.qdrant_storage.QdrantClient')
def test_qdrant_storage_factory(mock_qdrant_class, mock_qdrant_client):
    """Test Qdrant storage factory."""
    mock_qdrant_class.return_value = mock_qdrant_client
    
    storage = QdrantStorageFactory.create_storage(
        collection_name="test_collection",
        vector_dim=1536
    )
    
    assert storage is not None
    mock_qdrant_client.create_collection.assert_called_once()


@patch('core.memory.embedding_config.OpenAIEmbedding')
def test_embedding_factory(mock_embedding_class, mock_embedding):
    """Test embedding factory."""
    mock_embedding_class.return_value = mock_embedding
    
    embedding = EmbeddingFactory.create_embedding()
    
    assert embedding is not None


def test_embedding_output_dim():
    """Test embedding output dimension calculation."""
    dim = EmbeddingFactory.get_output_dim("text-embedding-3-small")
    assert dim == 1536
    
    dim = EmbeddingFactory.get_output_dim("text-embedding-3-large")
    assert dim == 3072
    
    dim = EmbeddingFactory.get_output_dim("unknown-model")
    assert dim == 1536  # Default


@patch('core.memory.camel_memory_manager.QdrantStorageFactory')
@patch('core.memory.camel_memory_manager.EmbeddingFactory')
def test_memory_manager_initialization(mock_embedding_factory, mock_storage_factory):
    """Test memory manager initialization."""
    mock_storage = MagicMock()
    mock_storage_factory.create_storage.return_value = mock_storage
    mock_storage_factory.ensure_collection_exists = MagicMock()
    
    mock_embedding = MagicMock()
    mock_embedding_factory.create_embedding.return_value = mock_embedding
    mock_embedding_factory.get_output_dim.return_value = 1536
    
    manager = CamelMemoryManager(agent_id="test_agent")
    
    assert manager.agent_id == "test_agent"
    assert manager.memory is not None


def test_memory_manager_stats():
    """Test memory manager statistics."""
    # This test would require mocking the full memory system
    # For now, just verify the method exists
    assert hasattr(CamelMemoryManager, 'get_stats')

