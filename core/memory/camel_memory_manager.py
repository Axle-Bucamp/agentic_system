"""
CAMEL Memory Manager

Wraps CAMEL's LongtermAgentMemory with Qdrant storage and Redis chat history.
"""
from typing import Optional, List, Dict, Any
from core.config import settings
from core.logging import log
from core.memory.qdrant_storage import QdrantStorageFactory
from core.memory.embedding_config import EmbeddingFactory

try:
    from camel.memories import (
        LongtermAgentMemory,
        ChatHistoryBlock,
        VectorDBBlock,
        MemoryRecord,
        ScoreBasedContextCreator,
    )
    from camel.messages import BaseMessage
    from camel.types import ModelType, OpenAIBackendRole
    from camel.utils import OpenAITokenCounter
    from camel.storages import InMemoryKeyValueStorage
    CAMEL_MEMORY_AVAILABLE = True
except ImportError:
    CAMEL_MEMORY_AVAILABLE = False
    log.warning("CAMEL memory not available. Install with: pip install camel-ai")


class CamelMemoryManager:
    """Manager for CAMEL long-term agent memory."""
    
    def __init__(
        self,
        agent_id: str,
        collection_name: Optional[str] = None,
        model_type: Optional[ModelType] = None
    ):
        """
        Initialize CAMEL memory manager.
        
        Args:
            agent_id: Unique identifier for the agent
            collection_name: Qdrant collection name (defaults to settings)
            model_type: Model type for token counting (defaults to GPT_4O_MINI)
        """
        if not CAMEL_MEMORY_AVAILABLE:
            raise ImportError("CAMEL memory not installed")
        
        self.agent_id = agent_id
        self.collection_name = collection_name or f"{settings.qdrant_collection_name}_{agent_id}"
        self.model_type = model_type or ModelType.GPT_4O_MINI
        
        # Initialize components
        self._memory: Optional[LongtermAgentMemory] = None
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize the CAMEL memory system."""
        try:
            # Get embedding model and dimension
            embedding = EmbeddingFactory.create_embedding()
            vector_dim = EmbeddingFactory.get_output_dim()
            
            # Ensure Qdrant collection exists
            QdrantStorageFactory.ensure_collection_exists(
                collection_name=self.collection_name,
                vector_dim=vector_dim
            )
            
            # Create Qdrant storage
            qdrant_storage = QdrantStorageFactory.create_storage(
                collection_name=self.collection_name,
                vector_dim=vector_dim
            )
            
            # Create chat history block (using Redis-compatible in-memory for now)
            # In production, this could be backed by Redis
            chat_history_block = ChatHistoryBlock(
                storage=InMemoryKeyValueStorage(),
                keep_rate=0.9
            )
            
            # Create vector DB block
            vector_db_block = VectorDBBlock(
                storage=qdrant_storage,
                embedding=embedding,
                retrieve_limit=settings.memory_retrieve_limit
            )
            
            # Create context creator
            context_creator = ScoreBasedContextCreator(
                token_counter=OpenAITokenCounter(self.model_type),
                token_limit=settings.memory_token_limit
            )
            
            # Create long-term memory
            self._memory = LongtermAgentMemory(
                context_creator=context_creator,
                chat_history_block=chat_history_block,
                vector_db_block=vector_db_block,
                retrieve_limit=settings.memory_retrieve_limit
            )
            
            log.info(f"Initialized CAMEL memory for agent: {self.agent_id}")
            
        except Exception as e:
            log.error(f"Failed to initialize CAMEL memory: {e}")
            raise
    
    @property
    def memory(self) -> LongtermAgentMemory:
        """Get the memory instance."""
        if self._memory is None:
            raise RuntimeError("Memory not initialized")
        return self._memory
    
    def write_record(
        self,
        message: BaseMessage,
        role: Optional[OpenAIBackendRole] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        Write a single memory record.
        
        Args:
            message: The message to store
            role: Backend role (USER or ASSISTANT). If None, inferred from message role.
            extra_info: Optional metadata
        """
        if role is None:
            # Infer role from message role name
            if hasattr(message, 'role_name'):
                if message.role_name.lower() in ['user', 'task coordinator']:
                    role = OpenAIBackendRole.USER
                else:
                    role = OpenAIBackendRole.ASSISTANT
            else:
                role = OpenAIBackendRole.USER
        
        record = MemoryRecord(
            message=message,
            role_at_backend=role,
            extra_info=extra_info or {}
        )
        self.memory.write_records([record])
    
    def write_records(
        self,
        records: List[MemoryRecord]
    ):
        """Write multiple memory records."""
        self.memory.write_records(records)
    
    def get_context(self) -> tuple[List[BaseMessage], int]:
        """
        Get context from memory.
        
        Returns:
            Tuple of (messages, token_count)
        """
        return self.memory.get_context()
    
    def retrieve(self, query: str) -> List:
        """
        Retrieve relevant records from memory.
        
        Args:
            query: Query string for semantic search
            
        Returns:
            List of ContextRecord objects
        """
        return self.memory.retrieve(query)
    
    def clear(self):
        """Clear all memory."""
        self.memory.clear()
        log.info(f"Cleared memory for agent: {self.agent_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            # Get context to see current state
            context, token_count = self.get_context()
            
            return {
                "agent_id": self.agent_id,
                "collection_name": self.collection_name,
                "context_message_count": len(context),
                "context_token_count": token_count,
                "retrieve_limit": settings.memory_retrieve_limit,
            }
        except Exception as e:
            log.error(f"Failed to get memory stats: {e}")
            return {
                "agent_id": self.agent_id,
                "error": str(e)
            }

