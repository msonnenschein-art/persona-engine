"""Conversation state tracking for Version B dynamic mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EmotionalState:
    """Tracks character's emotional state."""
    primary: str = "neutral"
    intensity: float = 0.5
    secondary: str | None = None

    def update(self, emotion: str, intensity: float, secondary: str | None = None) -> None:
        self.primary = emotion
        self.intensity = max(0.0, min(1.0, intensity))
        self.secondary = secondary

    def decay(self, rate: float = 0.1) -> None:
        """Decay emotional intensity toward neutral."""
        self.intensity = max(0.0, self.intensity - rate)
        if self.intensity < 0.2:
            self.primary = "neutral"
            self.secondary = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "intensity": self.intensity,
            "secondary": self.secondary,
        }


@dataclass
class RelationshipState:
    """Tracks relationship with the user."""
    trust_level: float = 0.3
    familiarity: float = 0.0
    rapport: float = 0.5
    tension: float = 0.0

    def adjust_trust(self, delta: float) -> None:
        self.trust_level = max(0.0, min(1.0, self.trust_level + delta))

    def adjust_familiarity(self, delta: float) -> None:
        self.familiarity = max(0.0, min(1.0, self.familiarity + delta))

    def adjust_rapport(self, delta: float) -> None:
        self.rapport = max(0.0, min(1.0, self.rapport + delta))

    def adjust_tension(self, delta: float) -> None:
        self.tension = max(0.0, min(1.0, self.tension + delta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust_level": self.trust_level,
            "familiarity": self.familiarity,
            "rapport": self.rapport,
            "tension": self.tension,
        }


@dataclass
class ConversationState:
    """Complete conversation state for Version B."""
    turn_count: int = 0
    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    relationship: RelationshipState = field(default_factory=RelationshipState)
    active_topics: list[str] = field(default_factory=list)
    mentioned_entities: set[str] = field(default_factory=set)
    revealed_secrets: list[str] = field(default_factory=list)
    conversation_flags: dict[str, bool] = field(default_factory=dict)
    custom_state: dict[str, Any] = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)

    def advance_turn(self) -> None:
        """Advance conversation by one turn."""
        self.turn_count += 1
        self.last_interaction = datetime.now()
        self.emotional_state.decay()

    def add_topic(self, topic: str) -> None:
        """Add a topic to active topics, keeping recent ones."""
        if topic in self.active_topics:
            self.active_topics.remove(topic)
        self.active_topics.insert(0, topic)
        self.active_topics = self.active_topics[:5]

    def add_entity(self, entity: str) -> None:
        """Track a mentioned entity."""
        self.mentioned_entities.add(entity)

    def set_flag(self, flag: str, value: bool = True) -> None:
        """Set a conversation flag."""
        self.conversation_flags[flag] = value

    def get_flag(self, flag: str, default: bool = False) -> bool:
        """Get a conversation flag."""
        return self.conversation_flags.get(flag, default)

    def record_secret_reveal(self, secret: str) -> None:
        """Record that a secret was revealed."""
        if secret not in self.revealed_secrets:
            self.revealed_secrets.append(secret)

    def build_context_block(self) -> str:
        """Build a context block for dynamic prompt assembly."""
        lines = [
            "## Current State",
            f"- Turn: {self.turn_count}",
            f"- Emotional state: {self.emotional_state.primary} (intensity: {self.emotional_state.intensity:.1f})",
        ]

        if self.emotional_state.secondary:
            lines.append(f"- Secondary emotion: {self.emotional_state.secondary}")

        lines.extend([
            f"- Trust level: {self.relationship.trust_level:.1f}",
            f"- Familiarity: {self.relationship.familiarity:.1f}",
            f"- Rapport: {self.relationship.rapport:.1f}",
        ])

        if self.relationship.tension > 0.2:
            lines.append(f"- Tension: {self.relationship.tension:.1f}")

        if self.active_topics:
            lines.append(f"- Active topics: {', '.join(self.active_topics)}")

        if self.revealed_secrets:
            lines.append(f"- Secrets revealed: {len(self.revealed_secrets)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "turn_count": self.turn_count,
            "emotional_state": self.emotional_state.to_dict(),
            "relationship": self.relationship.to_dict(),
            "active_topics": self.active_topics,
            "mentioned_entities": list(self.mentioned_entities),
            "revealed_secrets": self.revealed_secrets,
            "conversation_flags": self.conversation_flags,
            "custom_state": self.custom_state,
            "session_start": self.session_start.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationState:
        """Deserialize state from dictionary."""
        state = cls()
        state.turn_count = data.get("turn_count", 0)

        if "emotional_state" in data:
            es = data["emotional_state"]
            state.emotional_state = EmotionalState(
                primary=es.get("primary", "neutral"),
                intensity=es.get("intensity", 0.5),
                secondary=es.get("secondary"),
            )

        if "relationship" in data:
            rel = data["relationship"]
            state.relationship = RelationshipState(
                trust_level=rel.get("trust_level", 0.3),
                familiarity=rel.get("familiarity", 0.0),
                rapport=rel.get("rapport", 0.5),
                tension=rel.get("tension", 0.0),
            )

        state.active_topics = data.get("active_topics", [])
        state.mentioned_entities = set(data.get("mentioned_entities", []))
        state.revealed_secrets = data.get("revealed_secrets", [])
        state.conversation_flags = data.get("conversation_flags", {})
        state.custom_state = data.get("custom_state", {})

        if "session_start" in data:
            state.session_start = datetime.fromisoformat(data["session_start"])
        if "last_interaction" in data:
            state.last_interaction = datetime.fromisoformat(data["last_interaction"])

        return state
