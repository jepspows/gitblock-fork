"""Node reputation scoring system."""
import time
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ReputationScore:
    """Tracks a node's reputation based on performance metrics."""
    uptime_score: float = 50.0
    quality_score: float = 50.0
    latency_score: float = 50.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    _latencies: deque = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def overall(self) -> float:
        """Weighted overall reputation score (0-100)."""
        return (self.uptime_score * 0.3 +
                self.quality_score * 0.4 +
                self.latency_score * 0.3)

    @property
    def uptime_hours(self) -> float:
        """Hours since node started."""
        return (time.time() - self.start_time) / 3600

    def record_success(self, latency_ms: float, quality: float = 1.0) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self._latencies.append(latency_ms)
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

        # Update scores with exponential moving average
        alpha = 0.05
        success_rate = self.successful_requests / max(self.total_requests, 1)
        self.quality_score = (1 - alpha) * self.quality_score + alpha * (quality * 100)

        # Latency score: <100ms = 100, >2000ms = 0
        lat_score = max(0, min(100, 100 - (latency_ms - 100) * 100 / 1900))
        self.latency_score = (1 - alpha) * self.latency_score + alpha * lat_score

        # Uptime score increases with time
        hours = self.uptime_hours
        self.uptime_score = min(100, 50 + hours * 2)

    def record_failure(self) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        alpha = 0.1
        success_rate = self.successful_requests / max(self.total_requests, 1)
        self.quality_score = (1 - alpha) * self.quality_score + alpha * (success_rate * 100)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "overall": round(self.overall, 2),
            "uptime_score": round(self.uptime_score, 2),
            "quality_score": round(self.quality_score, 2),
            "latency_score": round(self.latency_score, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "uptime_hours": round(self.uptime_hours, 2),
        }
