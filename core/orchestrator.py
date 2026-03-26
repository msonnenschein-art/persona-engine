"""Orchestrator for persona conversations - handles both Version A and B modes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from .schema import Character
from .state import ConversationState
from .memory import TieredMemory
from .llm_adapter import LLMAdapter, Message, LLMResponse, create_adapter
from .rag_manager import RAGManager


class Mode(Enum):
    """Operating mode for the persona engine."""
    VERSION_A = "a"  # Static prompt
    VERSION_B = "b"  # Dynamic context assembly


def _subtext_directive(score: int) -> str:
    """Return a subtext instruction scaled to the 1-10 subtextuality score."""
    if score <= 2:
        return (
            "Subtext: minimal — this character says what they mean, plainly and directly."
        )
    if score <= 4:
        return (
            "Subtext: low — this character is mostly direct but occasionally hints "
            "rather than states outright."
        )
    if score <= 6:
        return (
            "Subtext: moderate — feelings are implied as often as named; "
            "the character hedges and deflects on emotional topics."
        )
    if score <= 8:
        return (
            "Subtext: high — this character rarely names their feelings directly; "
            "prefer implication, deflection, and indirection over plain statement."
        )
    return (
        "Subtext: very high — almost nothing this character says should be taken "
        "at face value; emotions and intentions are buried under misdirection, "
        "deflection, and pointed silence."
    )


def _digression_directive(score: int) -> str:
    """Return a pacing/structure instruction scaled to the 1-10 digressiveness score."""
    if score <= 2:
        return (
            "Pacing: respond concisely and on-point — cut to what matters, "
            "avoid tangents, don't linger."
        )
    if score <= 4:
        return (
            "Pacing: mostly direct; brief asides are fine when natural "
            "but return to the thread quickly."
        )
    if score <= 6:
        return (
            "Pacing: balanced — occasional digressions and asides are welcome "
            "when they add texture."
        )
    if score <= 8:
        return (
            "Pacing: digressive — feel free to wander, circle back, and follow "
            "tangents before returning to the point."
        )
    return (
        "Pacing: highly digressive — lean into spiraling asides and tangential "
        "detours; this character's mind rarely travels in a straight line and "
        "frequently loses — then rediscovers — the thread."
    )


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    user_input: str
    assistant_response: str
    state_snapshot: dict[str, Any] | None = None


class PersonaOrchestrator:
    """Orchestrates character conversations with state and memory management."""

    def __init__(
        self,
        character: Character,
        adapter: LLMAdapter,
        mode: Mode = Mode.VERSION_B,
        state: ConversationState | None = None,
        memory: TieredMemory | None = None,
        rag: RAGManager | None = None,
    ):
        self.character = character
        self.adapter = adapter
        self.mode = mode
        self.rag = rag

        if mode == Mode.VERSION_B:
            self.state = state or ConversationState()
            if character.initial_state:
                self.state.custom_state.update(character.initial_state)

            mc = character.memory_config
            self.memory = memory or TieredMemory(
                short_term_limit=mc.short_term_limit,
                long_term_limit=mc.long_term_limit,
                episodic_limit=mc.episodic_limit,
                consolidation_threshold=mc.consolidation_threshold,
                relevance_decay=mc.relevance_decay,
            )
        else:
            self.state = None
            self.memory = None

        self.history: list[ConversationTurn] = []
        self.messages: list[Message] = []

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on mode."""
        if self.mode == Mode.VERSION_A:
            return self.character.build_static_prompt()

        sections = []

        if self.character.system_prefix:
            sections.append(self.character.system_prefix)

        sections.append(f"You are {self.character.name}.")

        identity = self.character.description
        if self.character.fundamental_desire:
            identity += f"\n\nFundamental desire: {self.character.fundamental_desire}"
        sections.append(f"\n## Core Identity\n{identity}")

        sections.append(f"\n## Personality\n{self.character.personality}")
        sections.append(f"\n## Background\n{self.character.background}")
        sections.append(f"\n## Speaking Style\n{self.character.speaking_style}")

        register_lines = []
        if self.character.lived_in_genre:
            register_lines.append(
                f"- Tonal register: {self.character.lived_in_genre} — "
                "let this emotional world shape your diction, pacing, and mood."
            )
        if self.character.subtextuality:
            register_lines.append(f"- {_subtext_directive(self.character.subtextuality)}")
        if self.character.digressiveness:
            register_lines.append(f"- {_digression_directive(self.character.digressiveness)}")
        if register_lines:
            sections.append("\n## Dramatic Register\n" + "\n".join(register_lines))

        if self.character.goals:
            sections.append("\n## Current Goals\n" + "\n".join(f"- {g}" for g in self.character.goals))

        if self.character.quirks:
            sections.append("\n## Quirks & Mannerisms\n" + "\n".join(f"- {q}" for q in self.character.quirks))

        if self.state:
            sections.append("\n" + self.state.build_context_block())

        keywords = self._extract_keywords()
        if self.memory:
            memory_context = self.memory.build_context_block(keywords)
            if memory_context:
                sections.append("\n" + memory_context)

        revealable = self._check_revealable_secrets(keywords)
        if revealable:
            sections.append("\n## Available to Reveal (if appropriate)\n" + "\n".join(f"- {s}" for s in revealable))

        # Cold memory: semantic retrieval from the knowledge base (RAG tier)
        if self.rag and self.messages:
            query = self.messages[-1].content
            rag_block = self.rag.build_context_block(query)
            if rag_block:
                sections.append("\n" + rag_block)

        sections.append("\n## Guidelines")
        sections.append("- Stay in character at all times")
        sections.append("- Respond naturally based on your personality and current emotional state")
        sections.append("- Reference past interactions when relevant")
        sections.append("- Reveal secrets only when conditions feel right")

        return "\n".join(sections)

    def _extract_keywords(self) -> list[str]:
        """Extract keywords from recent conversation."""
        keywords = list(self.state.active_topics) if self.state else []

        if self.messages:
            last_msg = self.messages[-1].content if self.messages else ""
            words = re.findall(r'\b\w{4,}\b', last_msg.lower())
            stopwords = {"what", "when", "where", "which", "that", "this", "have", "with", "from", "your", "about"}
            keywords.extend([w for w in words if w not in stopwords][:5])

        return keywords

    def _check_revealable_secrets(self, keywords: list[str]) -> list[str]:
        """Check which secrets can be revealed."""
        if not self.state:
            return []

        return self.character.secrets.get_revealable(
            trust_level=self.state.relationship.trust_level,
            emotional_state=self.state.emotional_state.primary,
            keywords=keywords,
        )

    def _update_state_from_input(self, user_input: str) -> None:
        """Update state based on user input."""
        if not self.state:
            return

        self.state.relationship.adjust_familiarity(0.02)

        positive_signals = ["thank", "great", "love", "appreciate", "awesome", "amazing"]
        negative_signals = ["hate", "awful", "terrible", "worst", "stupid", "annoying"]

        input_lower = user_input.lower()

        if any(sig in input_lower for sig in positive_signals):
            self.state.relationship.adjust_trust(0.05)
            self.state.relationship.adjust_rapport(0.05)
            self.state.relationship.adjust_tension(-0.03)

        if any(sig in input_lower for sig in negative_signals):
            self.state.relationship.adjust_tension(0.1)
            self.state.relationship.adjust_rapport(-0.05)

        words = re.findall(r'\b\w{4,}\b', input_lower)
        important_words = [w for w in words if len(w) > 5][:2]
        for word in important_words:
            self.state.add_topic(word)

        if self.memory:
            importance = 0.4
            if "?" in user_input:
                importance += 0.1
            if len(user_input) > 100:
                importance += 0.1
            self.memory.add_short_term(f"User said: {user_input[:100]}", importance=importance)

    def _update_state_from_response(self, response: str) -> None:
        """Update state based on assistant response."""
        if not self.state:
            return

        self.state.advance_turn()

        keywords = self._extract_keywords()
        revealed = self._check_revealable_secrets(keywords)
        for secret in revealed:
            if secret.lower() in response.lower():
                self.state.record_secret_reveal(secret)
                self.character.secrets.mark_revealed(secret)
                if self.memory:
                    self.memory.add_long_term(f"Revealed: {secret[:50]}", importance=0.9, tags=["secret"])

        if self.memory:
            self.memory.add_short_term(f"I responded about: {response[:80]}", importance=0.3)

    def chat(self, user_input: str, **kwargs) -> str:
        """Process a chat message and return response."""
        if self.mode == Mode.VERSION_B:
            self._update_state_from_input(user_input)

        self.messages.append(Message(role="user", content=user_input))

        system_prompt = self._build_system_prompt()
        response = self.adapter.complete(self.messages, system=system_prompt, **kwargs)

        self.messages.append(Message(role="assistant", content=response.content))

        if self.mode == Mode.VERSION_B:
            self._update_state_from_response(response.content)

        self.history.append(ConversationTurn(
            user_input=user_input,
            assistant_response=response.content,
            state_snapshot=self.state.to_dict() if self.state else None,
        ))

        return response.content

    def chat_stream(self, user_input: str, **kwargs) -> Iterator[str]:
        """Stream a chat response."""
        if self.mode == Mode.VERSION_B:
            self._update_state_from_input(user_input)

        self.messages.append(Message(role="user", content=user_input))

        system_prompt = self._build_system_prompt()

        full_response = ""
        for chunk in self.adapter.stream(self.messages, system=system_prompt, **kwargs):
            full_response += chunk
            yield chunk

        self.messages.append(Message(role="assistant", content=full_response))

        if self.mode == Mode.VERSION_B:
            self._update_state_from_response(full_response)

        self.history.append(ConversationTurn(
            user_input=user_input,
            assistant_response=full_response,
            state_snapshot=self.state.to_dict() if self.state else None,
        ))

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of current state."""
        if not self.state:
            return {"mode": "version_a", "turns": len(self.history)}

        return {
            "mode": "version_b",
            "turns": self.state.turn_count,
            "emotional_state": self.state.emotional_state.to_dict(),
            "relationship": self.state.relationship.to_dict(),
            "active_topics": self.state.active_topics,
            "secrets_revealed": len(self.state.revealed_secrets),
        }

    def end_session(self, summary: str | None = None) -> None:
        """End the current session and create an episodic memory."""
        if not self.state or not self.memory:
            return

        if not summary:
            topics = ", ".join(self.state.active_topics[:3]) or "general conversation"
            summary = f"Conversation about {topics}"

        key_events = []
        for secret in self.state.revealed_secrets:
            key_events.append(f"Revealed: {secret[:30]}...")

        emotional_arc = f"{self.state.emotional_state.primary} (intensity: {self.state.emotional_state.intensity:.1f})"

        self.memory.add_episode(
            summary=summary,
            key_events=key_events,
            emotional_arc=emotional_arc,
            turn_count=self.state.turn_count,
        )

    def save_state(self, path: str | Path) -> None:
        """Save orchestrator state to file."""
        data = {
            "character_name": self.character.name,
            "mode": self.mode.value,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
        }

        if self.state:
            data["state"] = self.state.to_dict()
        if self.memory:
            data["memory"] = self.memory.to_dict()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self, path: str | Path) -> None:
        """Load orchestrator state from file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.messages = [Message(role=m["role"], content=m["content"]) for m in data.get("messages", [])]

        if "state" in data and self.mode == Mode.VERSION_B:
            self.state = ConversationState.from_dict(data["state"])

        if "memory" in data and self.mode == Mode.VERSION_B:
            mc = self.character.memory_config
            self.memory = TieredMemory.from_dict(
                data["memory"],
                short_term_limit=mc.short_term_limit,
                long_term_limit=mc.long_term_limit,
                episodic_limit=mc.episodic_limit,
                consolidation_threshold=mc.consolidation_threshold,
                relevance_decay=mc.relevance_decay,
            )

    def reset(self) -> None:
        """Reset conversation state."""
        self.messages.clear()
        self.history.clear()

        if self.mode == Mode.VERSION_B:
            self.state = ConversationState()
            if self.character.initial_state:
                self.state.custom_state.update(self.character.initial_state)

            mc = self.character.memory_config
            self.memory = TieredMemory(
                short_term_limit=mc.short_term_limit,
                long_term_limit=mc.long_term_limit,
                episodic_limit=mc.episodic_limit,
                consolidation_threshold=mc.consolidation_threshold,
                relevance_decay=mc.relevance_decay,
            )


def create_orchestrator(
    character_path: str | Path,
    provider: str = "anthropic",
    mode: str = "b",
    rag: RAGManager | None = None,
    **adapter_kwargs,
) -> PersonaOrchestrator:
    """Factory function to create a fully configured orchestrator.

    Pass *rag* to enable the cold memory tier.  If *rag* is provided and its
    collection is empty, :meth:`RAGManager.ingest_directory` will be called
    automatically to populate it from the knowledge base directory.
    """
    character = Character.from_yaml(character_path)
    adapter = create_adapter(provider, **adapter_kwargs)

    if rag is not None and rag.document_count == 0:
        rag.ingest_directory()

    return PersonaOrchestrator(
        character=character,
        adapter=adapter,
        mode=Mode.VERSION_A if mode.lower() == "a" else Mode.VERSION_B,
        rag=rag,
    )
