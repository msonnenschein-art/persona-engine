"""Rubric loader - reads YAML rubric files and converts them to DeepEval metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


@dataclass
class RubricCriterion:
    """A single evaluation criterion within a rubric."""
    id: str
    label: str
    prompt: str


@dataclass
class Rubric:
    """A fully parsed rubric definition loaded from a YAML file."""
    name: str
    version: str
    description: str
    weight: float
    criteria: list[RubricCriterion]
    scale: dict[int, str]       # 1-5 anchor descriptions
    notes: str = ""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Rubric":
        """Load a rubric from a YAML file."""
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        criteria = [
            RubricCriterion(
                id=c["id"],
                label=c["label"],
                prompt=c["prompt"],
            )
            for c in data.get("criteria", [])
        ]

        # YAML keys may be ints or strings depending on the parser
        scale = {int(k): str(v) for k, v in data.get("scale", {}).items()}

        return cls(
            name=data["name"],
            version=str(data.get("version", "1.0")),
            description=data.get("description", ""),
            weight=float(data.get("weight", 1.0)),
            criteria=criteria,
            scale=scale,
            notes=data.get("notes", ""),
        )

    # ------------------------------------------------------------------
    # DeepEval integration
    # ------------------------------------------------------------------

    def to_deepeval_metric(self) -> GEval:
        """Convert this rubric to a DeepEval :class:`GEval` metric.

        The combined criteria text and the 1-5 scale anchors are assembled into
        a single evaluation prompt so that the judge LLM can score holistically.
        """
        criteria_lines = "\n".join(
            f"  [{c.id}] {c.label}: {c.prompt}" for c in self.criteria
        )
        scale_lines = "\n".join(
            f"  {score}: {anchor}" for score, anchor in sorted(self.scale.items())
        )

        combined_criteria = (
            f"{self.description}\n\n"
            f"Evaluate the following dimensions:\n{criteria_lines}\n\n"
            f"Score on a 1-5 scale where:\n{scale_lines}"
        )

        evaluation_steps = [
            f"Review the conversation for '{c.label}': {c.prompt}"
            for c in self.criteria
        ] + [
            "Assign an overall integer score from 1 to 5 using the provided scale anchors.",
            "Provide a concise justification that references specific evidence from the conversation.",
        ]

        return GEval(
            name=self.name,
            criteria=combined_criteria,
            evaluation_steps=evaluation_steps,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
        )


# ------------------------------------------------------------------
# Directory-level helpers
# ------------------------------------------------------------------

def load_rubrics_from_dir(
    rubrics_dir: str | Path,
    names: list[str] | None = None,
) -> list[Rubric]:
    """Load all (or selected) rubrics from *rubrics_dir*.

    Args:
        rubrics_dir: Path to the folder containing YAML rubric files.
        names: Optional list of rubric ``name`` values to load.  When *None*,
               every YAML file in the directory is loaded.

    Returns:
        A list of :class:`Rubric` instances, sorted by file name.
    """
    rubrics_dir = Path(rubrics_dir)
    if not rubrics_dir.exists():
        return []

    rubrics: list[Rubric] = []
    for path in sorted(rubrics_dir.glob("*.yaml")) + sorted(rubrics_dir.glob("*.yml")):
        try:
            rubric = Rubric.from_yaml(path)
        except (KeyError, yaml.YAMLError) as exc:
            raise ValueError(f"Failed to load rubric from {path}: {exc}") from exc

        if names is None or rubric.name in names:
            rubrics.append(rubric)

    return rubrics
