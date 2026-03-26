"""Schema definitions for character configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class RevealCondition(Enum):
    """Conditions under which secrets can be revealed."""
    NEVER = "never"
    TRUST_THRESHOLD = "trust_threshold"
    EXPLICIT_ASK = "explicit_ask"
    EMOTIONAL_STATE = "emotional_state"
    KEYWORD_TRIGGER = "keyword_trigger"


@dataclass
class SecretEntry:
    """A single secret with reveal conditions."""
    content: str
    reveal_condition: RevealCondition = RevealCondition.NEVER
    threshold: float = 0.8
    triggers: list[str] = field(default_factory=list)
    revealed: bool = False


@dataclass
class CharacterSecrets:
    """Manages character secrets and their reveal logic."""
    entries: list[SecretEntry] = field(default_factory=list)

    def get_revealable(self, trust_level: float, emotional_state: str, keywords: list[str]) -> list[str]:
        """Return secrets that can be revealed given current conditions."""
        revealable = []
        for entry in self.entries:
            if entry.revealed:
                continue
            if entry.reveal_condition == RevealCondition.TRUST_THRESHOLD:
                if trust_level >= entry.threshold:
                    revealable.append(entry.content)
            elif entry.reveal_condition == RevealCondition.EMOTIONAL_STATE:
                if emotional_state in entry.triggers:
                    revealable.append(entry.content)
            elif entry.reveal_condition == RevealCondition.KEYWORD_TRIGGER:
                if any(kw.lower() in [k.lower() for k in keywords] for kw in entry.triggers):
                    revealable.append(entry.content)
            elif entry.reveal_condition == RevealCondition.EXPLICIT_ASK:
                if "secret" in [k.lower() for k in keywords] or "truth" in [k.lower() for k in keywords]:
                    revealable.append(entry.content)
        return revealable

    def mark_revealed(self, content: str) -> None:
        """Mark a secret as revealed."""
        for entry in self.entries:
            if entry.content == content:
                entry.revealed = True
                break


@dataclass
class MemoryConfig:
    """Configuration for tiered memory system."""
    short_term_limit: int = 10
    long_term_limit: int = 50
    episodic_limit: int = 20
    consolidation_threshold: int = 5
    relevance_decay: float = 0.95


@dataclass
class Character:
    """Complete character definition."""
    name: str
    description: str
    personality: str
    background: str
    speaking_style: str
    goals: list[str] = field(default_factory=list)
    quirks: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)
    secrets: CharacterSecrets = field(default_factory=CharacterSecrets)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    initial_state: dict[str, Any] = field(default_factory=dict)
    system_prefix: str = ""
    version_a_prompt: str = ""
    # Dramatic register fields
    fundamental_desire: str = ""
    subtextuality: int = 5       # 1 = blunt/direct, 10 = almost never says what they mean
    lived_in_genre: str = ""     # the character's internal emotional/tonal world
    digressiveness: int = 5      # 1 = terse/on-point, 10 = constantly spiraling into tangents

    @classmethod
    def from_yaml(cls, path: str | Path) -> Character:
        """Load character from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        secrets = CharacterSecrets()
        if "secrets" in data:
            for s in data["secrets"]:
                secrets.entries.append(
                    SecretEntry(
                        content=s["content"],
                        reveal_condition=RevealCondition(s.get("reveal_condition", "never")),
                        threshold=s.get("threshold", 0.8),
                        triggers=s.get("triggers", []),
                    )
                )

        memory_config = MemoryConfig(**data.get("memory_config", {}))

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            personality=data.get("personality", ""),
            background=data.get("background", ""),
            speaking_style=data.get("speaking_style", ""),
            goals=data.get("goals", []),
            quirks=data.get("quirks", []),
            relationships=data.get("relationships", {}),
            secrets=secrets,
            memory_config=memory_config,
            initial_state=data.get("initial_state", {}),
            system_prefix=data.get("system_prefix", ""),
            version_a_prompt=data.get("version_a_prompt", ""),
            fundamental_desire=data.get("fundamental_desire", ""),
            subtextuality=int(data.get("subtextuality", 5)),
            lived_in_genre=data.get("lived_in_genre", ""),
            digressiveness=int(data.get("digressiveness", 5)),
        )

    def build_static_prompt(self) -> str:
        """Build Version A static system prompt."""
        if self.version_a_prompt:
            return self.version_a_prompt

        sections = [
            f"You are {self.name}.",
            "",
            f"## Description\n{self.description}",
            f"## Personality\n{self.personality}",
            f"## Background\n{self.background}",
            f"## Speaking Style\n{self.speaking_style}",
        ]

        if self.goals:
            sections.append(f"## Goals\n" + "\n".join(f"- {g}" for g in self.goals))

        if self.quirks:
            sections.append(f"## Quirks\n" + "\n".join(f"- {q}" for q in self.quirks))

        if self.system_prefix:
            sections.insert(0, self.system_prefix)

        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Serialize character to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "personality": self.personality,
            "background": self.background,
            "speaking_style": self.speaking_style,
            "goals": self.goals,
            "quirks": self.quirks,
            "relationships": self.relationships,
        }
