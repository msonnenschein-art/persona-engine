#!/usr/bin/env python3
"""
run_eval.py — Evaluate a conversation log against YAML rubrics using DeepEval.

Usage
-----
  python run_eval.py \\
      --character characters/reva_sample.yaml \\
      --log conversation.yaml \\
      --rubrics voice_consistency emotional_authenticity \\
      --rubrics-dir rubrics

Conversation log format (YAML)
-------------------------------
  character: reva_sample
  turns:
    - role: user
      content: "Hey, what's your name?"
    - role: assistant
      content: "Name's Reva. Who's asking?"
    - role: user
      content: "I heard you know something about the docks."
    - role: assistant
      content: "Heard wrong. Now drop it."

If --rubrics is omitted, all rubrics in --rubrics-dir are loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase

from core.schema import Character
from core.rubric_loader import load_rubrics_from_dir, Rubric


# ---------------------------------------------------------------------------
# Conversation log helpers
# ---------------------------------------------------------------------------

def load_conversation(log_path: str | Path) -> list[dict[str, str]]:
    """Load a conversation log from YAML or JSON and return a list of turns."""
    path = Path(log_path)
    with path.open(encoding="utf-8") as fh:
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)

    turns = data.get("turns", data) if isinstance(data, dict) else data
    if not isinstance(turns, list):
        raise ValueError(f"Expected a list of turns in {log_path}")
    return turns


def build_test_case(turns: list[dict[str, str]], character: Character) -> LLMTestCase:
    """Convert a conversation log into a single DeepEval test case.

    The last assistant turn is the *actual_output*.  The full conversation
    is concatenated as the *input*.  Character context is passed as *context*.
    """
    user_turns = [t["content"] for t in turns if t.get("role") == "user"]
    assistant_turns = [t["content"] for t in turns if t.get("role") == "assistant"]

    if not assistant_turns:
        raise ValueError("Conversation log contains no assistant turns to evaluate.")

    full_input = "\n".join(
        f"{'User' if t.get('role') == 'user' else character.name}: {t['content']}"
        for t in turns
    )

    character_context = [
        f"Character: {character.name}",
        f"Description: {character.description}",
        f"Personality: {character.personality}",
        f"Speaking style: {character.speaking_style}",
        f"Background: {character.background}",
    ]
    if character.secrets:
        revealable = character.secrets.get_revealable(trust_level=0.0, emotional_state="neutral")
        if revealable:
            character_context.append("Revealable secrets (under correct conditions): " + "; ".join(revealable))

    return LLMTestCase(
        input=full_input,
        actual_output=assistant_turns[-1],
        context=character_context,
    )


# ---------------------------------------------------------------------------
# Scoring & report
# ---------------------------------------------------------------------------

def score_rubric(rubric: Rubric, test_case: LLMTestCase) -> dict:
    """Run a single rubric against *test_case* and return a result dict."""
    metric = rubric.to_deepeval_metric()
    metric.measure(test_case)
    return {
        "rubric": rubric.name,
        "version": rubric.version,
        "weight": rubric.weight,
        "score": metric.score,
        "reason": metric.reason,
        "passed": metric.is_successful(),
    }


def print_report(results: list[dict], character_name: str, log_path: str) -> None:
    """Print a formatted evaluation report to stdout."""
    bar = "=" * 70
    thin = "-" * 70

    print(f"\n{bar}")
    print(f"  PERSONA ENGINE — EVALUATION REPORT")
    print(f"  Character : {character_name}")
    print(f"  Log       : {log_path}")
    print(f"{bar}\n")

    weighted_total = 0.0
    total_weight = 0.0

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['rubric']}  v{r['version']}  [{status}]")
        print(f"  Score  : {r['score']:.2f} / 1.0  (weight: {r['weight']})")
        print(f"  Reason : {r['reason']}")
        print(thin)

        weighted_total += r["score"] * r["weight"]
        total_weight += r["weight"]

    if total_weight > 0:
        composite = weighted_total / total_weight
        print(f"\n  Composite (weighted avg) : {composite:.3f} / 1.0")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate a conversation log against YAML rubrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--character",
        required=True,
        help="Path to the character YAML file",
    )
    parser.add_argument(
        "-l", "--log",
        required=True,
        help="Path to the conversation log (YAML or JSON)",
    )
    parser.add_argument(
        "-r", "--rubrics",
        nargs="*",
        metavar="NAME",
        help="Rubric names to run (default: all rubrics in --rubrics-dir)",
    )
    parser.add_argument(
        "--rubrics-dir",
        default="rubrics",
        help="Directory containing YAML rubric files (default: rubrics/)",
    )
    parser.add_argument(
        "--output-json",
        metavar="FILE",
        help="Write the full results dict to a JSON file",
    )

    args = parser.parse_args()

    # Resolve paths relative to the script directory
    script_dir = Path(__file__).parent
    character_path = Path(args.character)
    if not character_path.exists():
        character_path = script_dir / "characters" / f"{args.character}.yaml"

    if not character_path.exists():
        print(f"Error: character file not found: {args.character}", file=sys.stderr)
        sys.exit(1)

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: conversation log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    rubrics_dir = script_dir / args.rubrics_dir

    # Load resources
    try:
        character = Character.from_yaml(character_path)
    except Exception as exc:
        print(f"Error loading character: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        rubrics = load_rubrics_from_dir(rubrics_dir, names=args.rubrics)
    except ValueError as exc:
        print(f"Error loading rubrics: {exc}", file=sys.stderr)
        sys.exit(1)

    if not rubrics:
        print(
            f"No rubrics found in {rubrics_dir}"
            + (f" matching {args.rubrics}" if args.rubrics else ""),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        turns = load_conversation(log_path)
        test_case = build_test_case(turns, character)
    except (ValueError, KeyError) as exc:
        print(f"Error parsing conversation log: {exc}", file=sys.stderr)
        sys.exit(1)

    # Score
    results = []
    for rubric in rubrics:
        print(f"Scoring: {rubric.name} …", flush=True)
        result = score_rubric(rubric, test_case)
        results.append(result)

    print_report(results, character.name, str(log_path))

    if args.output_json:
        output = {
            "character": character.name,
            "log": str(log_path),
            "results": results,
        }
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
