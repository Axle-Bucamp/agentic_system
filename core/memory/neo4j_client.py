"""Optional Neo4j client for persisting graph memory to a graph database."""
from __future__ import annotations

from typing import Optional

from core.config import settings
from core.logging import log
from core.models import GraphMemoryEdge, GraphMemoryNode

try:
    from neo4j import AsyncGraphDatabase  # type: ignore

    NEO4J_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AsyncGraphDatabase = None
    NEO4J_AVAILABLE = False
    log.debug("Neo4j driver not installed; graph persistence will rely on Redis only.")


class Neo4jGraphClient:
    """Thin async wrapper around the Neo4j driver."""

    def __init__(self):
        self._driver = None

    async def connect(self):
        if not settings.neo4j_enabled or not NEO4J_AVAILABLE:
            return
        if self._driver is not None:
            return
        try:
            self._driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=300,
            )
            log.info("Connected to Neo4j at %s", settings.neo4j_uri)
        except Exception as exc:  # pragma: no cover - connection failure
            log.warning("Unable to connect to Neo4j: %s", exc)
            self._driver = None

    async def close(self):
        if self._driver:
            try:
                await self._driver.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._driver = None

    async def upsert_node(self, node: GraphMemoryNode):
        if not self._driver:
            return
        query = """
        MERGE (n {node_id: $node_id})
        SET n.label = $label,
            n.node_type = $node_type,
            n.weight = $weight,
            n.metadata = $metadata,
            n.updated_at = $updated_at,
            n.created_at = coalesce(n.created_at, $created_at)
        """
        params = {
            "node_id": node.node_id,
            "label": node.label,
            "node_type": node.node_type,
            "weight": node.weight,
            "metadata": dict(node.metadata or {}),
            "updated_at": node.updated_at.isoformat(),
            "created_at": node.created_at.isoformat(),
        }
        try:
            async with self._driver.session() as session:
                await session.run(query, **params)
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("Neo4j node upsert failed for %s: %s", node.node_id, exc)

    async def upsert_edge(self, edge: GraphMemoryEdge):
        if not self._driver:
            return
        query = """
        MERGE (source {node_id: $source_id})
        MERGE (target {node_id: $target_id})
        MERGE (source)-[r:REL {edge_id: $edge_id}]->(target)
        SET r.relation = $relation,
            r.weight = $weight,
            r.metadata = $metadata,
            r.updated_at = $updated_at,
            r.created_at = coalesce(r.created_at, $created_at)
        """
        params = {
            "source_id": edge.source,
            "target_id": edge.target,
            "edge_id": edge.edge_id,
            "relation": edge.relation,
            "weight": edge.weight,
            "metadata": dict(edge.metadata or {}),
            "updated_at": edge.updated_at.isoformat(),
            "created_at": edge.created_at.isoformat(),
        }
        try:
            async with self._driver.session() as session:
                await session.run(query, **params)
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("Neo4j edge upsert failed for %s: %s", edge.edge_id, exc)


async def get_neo4j_client() -> Optional[Neo4jGraphClient]:
    """Factory to return a connected Neo4j client when configured."""
    if not settings.neo4j_enabled:
        return None
    client = Neo4jGraphClient()
    await client.connect()
    if client._driver is None:
        return None
    return client

