"""GitBlock Agent Memory — Persistent state for autonomous agents across runs.

Agents need memory between sessions. This module provides a lightweight
filesystem-backed memory store. In production, this can be swapped for
IPFS/Arweave-backed storage for true decentralization.

Memory structure (per agent):
    ~/.gitblock/agents/<agent_id>/
        memory.json       # agent state, goals, active topics
        sessions/         # past session logs
        health.json       # quality scores, failure tracking
        config.json       # agent configuration
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def _agent_dir(agent_id: str) -> Path:
    """Get the agent's data directory."""
    base = Path(os.environ.get("GITBLOCK_AGENT_DIR", Path.home() / ".gitblock" / "agents"))
    return base / agent_id


@dataclass
class AgentMemory:
    """Persistent memory for a single agent."""
    agent_id: str
    goals: list[str] = field(default_factory=list)
    active_topics: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)  # key → value
    last_run: Optional[str] = None
    total_runs: int = 0
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    @classmethod
    def load(cls, agent_id: str) -> "AgentMemory":
        """Load memory from disk, or create fresh."""
        path = _agent_dir(agent_id) / "memory.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(agent_id=agent_id, **{k: v for k, v in data.items() if k != "agent_id"})
            except (json.JSONDecodeError, KeyError):
                pass
        return cls(agent_id=agent_id)

    def save(self) -> None:
        """Persist memory to disk."""
        d = _agent_dir(self.agent_id)
        d.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        (d / "memory.json").write_text(json.dumps(data, indent=2))

    def add_note(self, key: str, value: str) -> None:
        """Add or update a note."""
        self.notes[key] = value
        self.total_runs += 1
        self.last_run = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    def summary(self) -> str:
        """Human-readable memory summary for injection into agent context."""
        parts = []
        if self.goals:
            parts.append(f"Goals: {', '.join(self.goals)}")
        if self.active_topics:
            parts.append(f"Active topics: {', '.join(self.active_topics)}")
        if self.notes:
            parts.append("Notes:")
            for k, v in self.notes.items():
                parts.append(f"  - {k}: {v}")
        if self.last_run:
            parts.append(f"Last run: {self.last_run} (total runs: {self.total_runs})")
        return "\n".join(parts) if parts else "(no memory yet)"


@dataclass
class AgentHealth:
    """Quality tracking for an agent — similar to Aeon's skill-health system."""
    agent_id: str
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0
    consecutive_failures: int = 0
    avg_quality_score: float = 0.0
    last_quality_score: float = 0.0
    scores: list[dict] = field(default_factory=list)  # last 30 runs

    @classmethod
    def load(cls, agent_id: str) -> "AgentHealth":
        path = _agent_dir(agent_id) / "health.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(agent_id=agent_id, **{k: v for k, v in data.items() if k != "agent_id"})
            except (json.JSONDecodeError, KeyError):
                pass
        return cls(agent_id=agent_id)

    def save(self) -> None:
        d = _agent_dir(self.agent_id)
        d.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        (d / "health.json").write_text(json.dumps(data, indent=2))

    def record_session(self, success: bool, quality: float = 0.0, error: str = "") -> None:
        """Record a session result."""
        self.total_sessions += 1
        if success:
            self.successful_sessions += 1
            self.consecutive_failures = 0
        else:
            self.failed_sessions += 1
            self.consecutive_failures += 1

        self.last_quality_score = quality
        self.scores.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "success": success,
            "quality": quality,
            "error": error[:200] if error else "",
        })

        # Rolling window — keep last 30
        if len(self.scores) > 30:
            self.scores = self.scores[-30:]

        # Recalculate average
        valid = [s["quality"] for s in self.scores if s["success"]]
        self.avg_quality_score = round(sum(valid) / len(valid), 2) if valid else 0.0

    def needs_repair(self) -> bool:
        """Check if agent needs auto-repair intervention (3+ consecutive failures)."""
        return self.consecutive_failures >= 3

    def summary(self) -> str:
        """Human-readable health summary."""
        rate = (self.successful_sessions / max(self.total_sessions, 1)) * 100
        return (
            f"Sessions: {self.total_sessions} ({rate:.0f}% success) | "
            f"Quality: {self.last_quality_score:.1f}/5 (avg {self.avg_quality_score:.1f}) | "
            f"Consecutive failures: {self.consecutive_failures}"
        )


@dataclass
class AgentConfig:
    """Agent configuration — mirrors Aeon's aeon.yml per-skill config."""
    agent_id: str
    skill: str  # skill name (e.g., "token-alert", "morning-brief")
    schedule: str = "0 9 * * *"  # cron expression
    model: str = "llama-3.3-70b"
    var: str = ""  # focus parameter
    channels: list[str] = field(default_factory=list)  # ["telegram", "discord", "email"]
    max_turns: int = 20  # max LLM turns per session
    enabled: bool = True

    @classmethod
    def load(cls, agent_id: str) -> Optional["AgentConfig"]:
        path = _agent_dir(agent_id) / "config.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(agent_id=agent_id, **{k: v for k, v in data.items() if k != "agent_id"})
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def save(self) -> None:
        d = _agent_dir(self.agent_id)
        d.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        (d / "config.json").write_text(json.dumps(data, indent=2))
