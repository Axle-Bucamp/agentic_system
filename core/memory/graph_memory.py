"""Graph-based long-term memory manager for user knowledge."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from core.logging import log
from core.models import GraphMemoryEdge, GraphMemoryNode


class GraphMemoryManager:
    """Manage a lightweight knowledge graph stored in Redis (optionally mirrored to Neo4j)."""

    def __init__(self, redis_client, namespace: str = "memory:graph", neo4j_client=None):
        self.redis = redis_client
        self.namespace = namespace
        self.neo4j_client = neo4j_client

    def _nodes_key(self) -> str:
        return f"{self.namespace}:nodes"

    def _edges_key(self) -> str:
        return f"{self.namespace}:edges"

    async def _delete_hash_field(self, key: str, field: str):
        """Best-effort deletion of a hash field without requiring a dedicated client helper."""
        redis_raw = getattr(self.redis, "redis", None)
        if redis_raw is None:
            return
        try:
            await redis_raw.hdel(key, field)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning("Failed to delete field %s from %s: %s", field, key, exc)

    @staticmethod
    def build_edge_id(source: str, relation: str, target: str) -> str:
        """Create a deterministic edge identifier."""
        joined = f"{source}|{relation}|{target}"
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
        return f"edge:{digest}"

    async def upsert_node(self, node: GraphMemoryNode) -> GraphMemoryNode:
        """Insert or update a node within the graph."""
        try:
            stored = await self.redis.hget(self._nodes_key(), node.node_id)
            if stored:
                existing = GraphMemoryNode.parse_raw(stored)
                node.created_at = existing.created_at
                node.weight = max(existing.weight, node.weight)
            node.updated_at = datetime.utcnow()
            await self.redis.hset(self._nodes_key(), node.node_id, node.json())
            if self.neo4j_client:
                await self.neo4j_client.upsert_node(node)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("Failed to upsert graph node %s: %s", node.node_id, exc)
        return node

    async def upsert_edge(self, edge: GraphMemoryEdge) -> GraphMemoryEdge:
        """Insert or update an edge within the graph."""
        try:
            stored = await self.redis.hget(self._edges_key(), edge.edge_id)
            if stored:
                existing = GraphMemoryEdge.parse_raw(stored)
                edge.created_at = existing.created_at
                edge.weight = max(existing.weight, edge.weight)
            edge.updated_at = datetime.utcnow()
            await self.redis.hset(self._edges_key(), edge.edge_id, edge.json())
            if self.neo4j_client:
                await self.neo4j_client.upsert_edge(edge)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("Failed to upsert graph edge %s: %s", edge.edge_id, exc)
        return edge

    async def connect(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphMemoryEdge:
        """Create or reinforce an edge between two nodes."""
        edge_id = self.build_edge_id(source_id, relation, target_id)
        edge = GraphMemoryEdge(
            edge_id=edge_id,
            source=source_id,
            target=target_id,
            relation=relation,
            weight=weight,
            metadata=metadata or {},
        )
        return await self.upsert_edge(edge)

    async def get_node(self, node_id: str) -> Optional[GraphMemoryNode]:
        stored = await self.redis.hget(self._nodes_key(), node_id)
        if not stored:
            return None
        return GraphMemoryNode.parse_raw(stored)

    async def get_neighbors(
        self, node_id: str, relation: Optional[str] = None, limit: int = 25
    ) -> Dict[str, List[GraphMemoryEdge]]:
        """Return incident edges keyed by direction for a node."""
        edges = await self.redis.hgetall(self._edges_key())
        outgoing: List[GraphMemoryEdge] = []
        incoming: List[GraphMemoryEdge] = []
        for payload in edges.values():
            try:
                edge = GraphMemoryEdge.parse_raw(payload)
            except Exception:
                continue
            if relation and edge.relation != relation:
                continue
            if edge.source == node_id:
                outgoing.append(edge)
            if edge.target == node_id:
                incoming.append(edge)
        return {
            "outgoing": outgoing[:limit],
            "incoming": incoming[:limit],
        }

    async def get_snapshot(self, node_limit: int = 200, edge_limit: int = 500) -> Dict[str, Iterable]:
        """Return a snapshot of the current graph state for UI or analytics."""
        nodes_raw = await self.redis.hgetall(self._nodes_key())
        edges_raw = await self.redis.hgetall(self._edges_key())
        nodes = []
        edges = []
        for payload in nodes_raw.values():
            try:
                nodes.append(GraphMemoryNode.parse_raw(payload))
            except Exception:
                continue
        for payload in edges_raw.values():
            try:
                edges.append(GraphMemoryEdge.parse_raw(payload))
            except Exception:
                continue
        return {
            "nodes": nodes[:node_limit],
            "edges": edges[:edge_limit],
        }

    async def record_user_input(
        self,
        user_id: str,
        content: str,
        tags: Optional[List[str]] = None,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Capture user-provided knowledge as graph facts."""
        safe_tags = tags or []
        metadata = metadata or {}

        user_node = GraphMemoryNode(
            node_id=f"user:{user_id}",
            label=user_id,
            node_type="user",
            weight=1.0,
        )
        await self.upsert_node(user_node)

        fact_hash = hashlib.sha1(f"{user_id}:{content}".encode("utf-8")).hexdigest()
        fact_node = GraphMemoryNode(
            node_id=f"fact:{fact_hash}",
            label=content[:120],
            node_type="fact",
            weight=weight,
            metadata={"content": content, "tags": safe_tags, **metadata},
        )
        await self.upsert_node(fact_node)

        await self.connect(user_node.node_id, fact_node.node_id, "PROVIDED", weight)

        for tag in safe_tags:
            topic_node = GraphMemoryNode(
                node_id=f"topic:{tag.lower()}",
                label=tag,
                node_type="topic",
                weight=weight,
            )
            await self.upsert_node(topic_node)
            await self.connect(fact_node.node_id, topic_node.node_id, "TAGGED", weight)

    async def prune(self, max_nodes: int = 2000, min_weight: float = 0.05):
        """Trim low-value nodes/edges to keep the graph manageable."""
        nodes_raw = await self.redis.hgetall(self._nodes_key())
        if len(nodes_raw) <= max_nodes:
            return

        parsed_nodes: List[GraphMemoryNode] = []
        for payload in nodes_raw.values():
            try:
                parsed_nodes.append(GraphMemoryNode.parse_raw(payload))
            except Exception:
                continue

        sorted_nodes = sorted(parsed_nodes, key=lambda node: node.weight)

        removable = [node for node in sorted_nodes if node.weight < min_weight]
        for node in removable:
            await self._delete_hash_field(self._nodes_key(), node.node_id)
            nodes_raw.pop(node.node_id, None)

        edges_raw = await self.redis.hgetall(self._edges_key())
        for edge_id, edge_payload in edges_raw.items():
            try:
                edge = GraphMemoryEdge.parse_raw(edge_payload)
            except Exception:
                continue
            if edge.source not in nodes_raw or edge.target not in nodes_raw:
                await self._delete_hash_field(self._edges_key(), edge_id)

