"""Baseline Comparison Mode - side-by-side Version A vs Version B for client demos."""

from __future__ import annotations

import textwrap
from typing import Iterator, Optional

from .orchestrator import PersonaOrchestrator, Mode, create_orchestrator


class ComparisonResult:
    """Side-by-side response from Version A (static) and Version B (dynamic)."""

    def __init__(
        self,
        user_input: str,
        response_a: str,
        response_b: str,
        turn: int,
    ):
        self.user_input = user_input
        self.response_a = response_a
        self.response_b = response_b
        self.turn = turn

    def format(self, terminal_width: int = 100) -> str:
        """Return a formatted side-by-side string suitable for terminal output."""
        col = max(30, (terminal_width - 3) // 2)
        bar = "─" * terminal_width

        label_a = "VERSION A — STATIC PROMPT"
        label_b = "VERSION B — DYNAMIC CONTEXT"

        lines = [
            "",
            bar,
            f"  Turn {self.turn}  |  You: {self.user_input[:terminal_width - 20]}",
            bar,
            f"  {label_a:<{col - 2}}│  {label_b}",
            "─" * terminal_width,
        ]

        wrapped_a = textwrap.wrap(self.response_a, col - 2) or ["(no response)"]
        wrapped_b = textwrap.wrap(self.response_b, col - 2) or ["(no response)"]
        max_rows = max(len(wrapped_a), len(wrapped_b))

        for i in range(max_rows):
            left = wrapped_a[i] if i < len(wrapped_a) else ""
            right = wrapped_b[i] if i < len(wrapped_b) else ""
            lines.append(f"  {left:<{col - 2}}│  {right}")

        lines.append(bar)
        return "\n".join(lines)


class BaselineComparison:
    """
    Runs the same user message through both Version A and Version B orchestrators
    and returns a :class:`ComparisonResult` with both responses.

    Intended for client-facing demos that illustrate the architectural difference
    between a static-prompt approach and the full dynamic context/memory system.
    """

    def __init__(
        self,
        orchestrator_a: PersonaOrchestrator,
        orchestrator_b: PersonaOrchestrator,
    ):
        if orchestrator_a.mode != Mode.VERSION_A:
            raise ValueError("orchestrator_a must be initialised in VERSION_A mode")
        if orchestrator_b.mode != Mode.VERSION_B:
            raise ValueError("orchestrator_b must be initialised in VERSION_B mode")

        self.orchestrator_a = orchestrator_a
        self.orchestrator_b = orchestrator_b
        self._turn = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_input: str, **adapter_kwargs) -> ComparisonResult:
        """Send *user_input* to both orchestrators and return the comparison."""
        self._turn += 1
        response_a = self.orchestrator_a.chat(user_input, **adapter_kwargs)
        response_b = self.orchestrator_b.chat(user_input, **adapter_kwargs)
        return ComparisonResult(
            user_input=user_input,
            response_a=response_a,
            response_b=response_b,
            turn=self._turn,
        )

    def reset(self) -> None:
        """Reset both orchestrators to their initial state."""
        self.orchestrator_a.reset()
        self.orchestrator_b.reset()
        self._turn = 0

    @property
    def turn(self) -> int:
        """Current turn number."""
        return self._turn

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_character_path(
        cls,
        character_path: str,
        provider: str = "anthropic",
        **adapter_kwargs,
    ) -> "BaselineComparison":
        """Create a comparison pair from a single character YAML path."""
        orc_a = create_orchestrator(
            character_path=character_path,
            provider=provider,
            mode="a",
            **adapter_kwargs,
        )
        orc_b = create_orchestrator(
            character_path=character_path,
            provider=provider,
            mode="b",
            **adapter_kwargs,
        )
        return cls(orc_a, orc_b)
