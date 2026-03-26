"""Tiered memory system for Version B dynamic mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections import deque
import json


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    timestamp: datetime
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    source: str = "conversation"
    access_count: int = 0
    last_accessed: datetime | None = None

    def access(self) -> None:
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def calculate_relevance(self, keywords: list[str], decay_rate: float = 0.95) -> float:
        """Calculate current relevance score."""
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        time_decay = decay_rate ** age_hours

        keyword_match = sum(1 for kw in keywords if kw.lower() in self.content.lower())
        keyword_bonus = min(keyword_match * 0.2, 0.6)

        access_bonus = min(self.access_count * 0.05, 0.3)

        return (self.importance * time_decay) + keyword_bonus + access_bonus

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "tags": self.tags,
            "source": self.source,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        entry = cls(
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            source=data.get("source", "conversation"),
            access_count=data.get("access_count", 0),
        )
        if data.get("last_accessed"):
            entry.last_accessed = datetime.fromisoformat(data["last_accessed"])
        return entry


@dataclass
class EpisodicMemory:
    """A complete interaction episode."""
    summary: str
    key_events: list[str]
    emotional_arc: str
    timestamp: datetime
    turn_count: int
    outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "key_events": self.key_events,
            "emotional_arc": self.emotional_arc,
            "timestamp": self.timestamp.isoformat(),
            "turn_count": self.turn_count,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicMemory:
        return cls(
            summary=data["summary"],
            key_events=data.get("key_events", []),
            emotional_arc=data.get("emotional_arc", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            turn_count=data.get("turn_count", 0),
            outcome=data.get("outcome", ""),
        )


class TieredMemory:
    """Manages three tiers of memory: short-term, long-term, and episodic."""

    def __init__(
        self,
        short_term_limit: int = 10,
        long_term_limit: int = 50,
        episodic_limit: int = 20,
        consolidation_threshold: int = 5,
        relevance_decay: float = 0.95,
    ):
        self.short_term_limit = short_term_limit
        self.long_term_limit = long_term_limit
        self.episodic_limit = episodic_limit
        self.consolidation_threshold = consolidation_threshold
        self.relevance_decay = relevance_decay

        self.short_term: deque[MemoryEntry] = deque(maxlen=short_term_limit)
        self.long_term: list[MemoryEntry] = []
        self.episodic: list[EpisodicMemory] = []

        self._consolidation_counter = 0

    def add_short_term(self, content: str, importance: float = 0.5, tags: list[str] | None = None) -> None:
        """Add a memory to short-term storage."""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or [],
            source="conversation",
        )
        self.short_term.append(entry)
        self._consolidation_counter += 1

        if self._consolidation_counter >= self.consolidation_threshold:
            self._consolidate()
            self._consolidation_counter = 0

    def add_long_term(self, content: str, importance: float = 0.7, tags: list[str] | None = None) -> None:
        """Directly add a memory to long-term storage."""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or [],
            source="direct",
        )
        self.long_term.append(entry)
        self._prune_long_term()

    def add_episode(self, summary: str, key_events: list[str], emotional_arc: str, turn_count: int, outcome: str = "") -> None:
        """Add an episodic memory."""
        episode = EpisodicMemory(
            summary=summary,
            key_events=key_events,
            emotional_arc=emotional_arc,
            timestamp=datetime.now(),
            turn_count=turn_count,
            outcome=outcome,
        )
        self.episodic.append(episode)
        if len(self.episodic) > self.episodic_limit:
            self.episodic = self.episodic[-self.episodic_limit:]

    def retrieve(self, keywords: list[str], limit: int = 5) -> list[MemoryEntry]:
        """Retrieve most relevant memories based on keywords."""
        all_memories = list(self.short_term) + self.long_term

        scored = []
        for mem in all_memories:
            score = mem.calculate_relevance(keywords, self.relevance_decay)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, mem in scored[:limit]:
            mem.access()
            results.append(mem)

        return results

    def retrieve_recent(self, limit: int = 5) -> list[MemoryEntry]:
        """Retrieve most recent short-term memories."""
        recent = list(self.short_term)[-limit:]
        for mem in recent:
            mem.access()
        return recent

    def retrieve_episodes(self, limit: int = 3) -> list[EpisodicMemory]:
        """Retrieve most recent episodic memories."""
        return self.episodic[-limit:]

    def _consolidate(self) -> None:
        """Move important short-term memories to long-term."""
        threshold = 0.6
        for mem in list(self.short_term):
            if mem.importance >= threshold or mem.access_count >= 2:
                if mem not in self.long_term:
                    self.long_term.append(mem)
        self._prune_long_term()

    def _prune_long_term(self) -> None:
        """Prune long-term memory to stay within limits."""
        if len(self.long_term) > self.long_term_limit:
            self.long_term.sort(
                key=lambda m: m.calculate_relevance([], self.relevance_decay),
                reverse=True
            )
            self.long_term = self.long_term[:self.long_term_limit]

    def build_context_block(self, keywords: list[str] | None = None, include_episodic: bool = True) -> str:
        """Build a memory context block for prompt assembly."""
        lines = ["## Memory Context"]

        recent = self.retrieve_recent(3)
        if recent:
            lines.append("\n### Recent")
            for mem in recent:
                lines.append(f"- {mem.content}")

        if keywords:
            relevant = self.retrieve(keywords, 3)
            relevant = [m for m in relevant if m not in recent]
            if relevant:
                lines.append("\n### Relevant")
                for mem in relevant:
                    lines.append(f"- {mem.content}")

        if include_episodic:
            episodes = self.retrieve_episodes(2)
            if episodes:
                lines.append("\n### Past Interactions")
                for ep in episodes:
                    lines.append(f"- {ep.summary}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "short_term": [m.to_dict() for m in self.short_term],
            "long_term": [m.to_dict() for m in self.long_term],
            "episodic": [e.to_dict() for e in self.episodic],
            "consolidation_counter": self._consolidation_counter,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        short_term_limit: int = 10,
        long_term_limit: int = 50,
        episodic_limit: int = 20,
        consolidation_threshold: int = 5,
        relevance_decay: float = 0.95,
    ) -> TieredMemory:
        """Deserialize memory from dictionary."""
        memory = cls(
            short_term_limit=short_term_limit,
            long_term_limit=long_term_limit,
            episodic_limit=episodic_limit,
            consolidation_threshold=consolidation_threshold,
            relevance_decay=relevance_decay,
        )

        for m in data.get("short_term", []):
            entry = MemoryEntry.from_dict(m)
            memory.short_term.append(entry)

        for m in data.get("long_term", []):
            memory.long_term.append(MemoryEntry.from_dict(m))

        for e in data.get("episodic", []):
            memory.episodic.append(EpisodicMemory.from_dict(e))

        memory._consolidation_counter = data.get("consolidation_counter", 0)
        return memory

    def save(self, path: str) -> None:
        """Save memory to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> TieredMemory:
        """Load memory from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, **kwargs)
