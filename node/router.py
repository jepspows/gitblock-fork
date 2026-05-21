"""Smart routing — pick the best node for each request."""
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class NodeInfo:
    """Information about a network node."""
    node_id: str
    endpoint: str
    models: list[str]
    reputation: float
    avg_latency_ms: float
    current_load: int
    max_load: int
    region: str = "unknown"
    last_seen: float = 0.0

    @property
    def available(self) -> bool:
        """Whether the node can accept requests."""
        return (self.current_load < self.max_load and
                time.time() - self.last_seen < 120)

    @property
    def load_ratio(self) -> float:
        """Current load as a fraction of capacity."""
        return self.current_load / max(self.max_load, 1)


class Router:
    """Routes requests to the best available node."""

    def __init__(self):
        self.nodes: dict[str, NodeInfo] = {}
        self._latency_cache: dict[str, float] = {}

    def register_node(self, node: NodeInfo) -> None:
        """Register or update a node."""
        node.last_seen = time.time()
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the registry."""
        self.nodes.pop(node_id, None)

    def pick_node(
        self,
        model: str,
        region: Optional[str] = None,
    ) -> Optional[NodeInfo]:
        """Pick the best node for a request.

        Scoring: 40% reputation + 30% latency + 20% load + 10% region match
        """
        candidates = [
            n for n in self.nodes.values()
            if n.available and model in n.models
        ]

        if not candidates:
            return None

        def score(node: NodeInfo) -> float:
            s = 0.0
            # Reputation (0-100 -> 0-1)
            s += 0.4 * (node.reputation / 100)
            # Latency (lower is better, 0-2000ms range)
            lat = self._latency_cache.get(node.node_id, node.avg_latency_ms)
            s += 0.3 * max(0, 1 - lat / 2000)
            # Load (lower is better)
            s += 0.2 * (1 - node.load_ratio)
            # Region match
            if region and node.region == region:
                s += 0.1
            return s

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def get_healthy_nodes(self) -> list[NodeInfo]:
        """Get all healthy, available nodes."""
        return [n for n in self.nodes.values() if n.available]

    def cleanup_stale(self, max_age: float = 120) -> int:
        """Remove nodes not seen within max_age seconds."""
        now = time.time()
        stale = [nid for nid, n in self.nodes.items()
                 if now - n.last_seen > max_age]
        for nid in stale:
            del self.nodes[nid]
        return len(stale)
