#!/usr/bin/env python3
"""
ingest_transcript.py — Reverse-engineer a character YAML from a chat transcript.

Transcript format (plain text, alternating speakers):
    User: Hey, what's your name?
    Reva: Name's Reva. Who's asking?
    User: Where are you from?
    Reva: Ganymede. Long time ago.

Usage:
    python ingest_transcript.py transcript.txt
    python ingest_transcript.py transcript.txt -o characters/aria.yaml
    python ingest_transcript.py transcript.txt --notes "Client session 2024-01-15"
    python ingest_transcript.py transcript.txt --character-label Aria
    python ingest_transcript.py transcript.txt --model claude-sonnet-4-6
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ── Extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert narrative designer and character analyst. Analyse the
transcript provided and extract a detailed character profile for the
character speaker (not the user).

Return ONLY a single JSON object — no markdown fences, no commentary.

Every top-level scalar field is an object with "value" and "inferred" keys.
Set "inferred": false only when the transcript contains a direct, explicit
statement of that fact. Set "inferred": true when you are synthesising from
tone, implication, subtext, or context.

Return exactly this schema (all keys required; use empty arrays/objects
rather than null for missing data):

{
  "character_name": {"value": "string", "inferred": bool},
  "description": {
    "value": "2-4 sentence physical and demographic description",
    "inferred": bool
  },
  "personality": {
    "value": "3-5 sentences: temperament, emotional response patterns, how they handle conflict, relationship dynamics",
    "inferred": bool
  },
  "background": {
    "value": "apparent backstory: origin, history, current situation and role",
    "inferred": bool
  },
  "speaking_style": {
    "value": "vocabulary register, sentence rhythm, idioms, tone shifts, notable speech patterns and verbal tics",
    "inferred": bool
  },
  "goals": [
    {"value": "string", "inferred": bool}
  ],
  "quirks": [
    {"value": "observable behavioural quirk or mannerism", "inferred": bool}
  ],
  "topics_avoided": [
    {"value": "topic or question the character deflects, evades or shuts down", "inferred": bool}
  ],
  "topics_embraced": [
    {"value": "topic the character engages with warmly, at length, or with clear interest", "inferred": bool}
  ],
  "relationships": [
    {"name": "person or group name", "description": "nature of the relationship", "inferred": bool}
  ],
  "secrets": [
    {
      "content": "likely hidden information or guarded truth inferred from evasions and subtext",
      "reveal_condition": "trust_threshold|keyword_trigger|emotional_state|explicit_ask|never",
      "threshold": 0.75,
      "triggers": ["keyword or emotion name"],
      "inferred": true
    }
  ],
  "initial_state": {
    "value": {"key": "value"},
    "inferred": bool
  },
  "system_prefix": {
    "value": "1-2 sentence roleplay instruction (e.g. You are X. Stay in character at all times.)",
    "inferred": true
  },
  "sample_dialogue": [
    {"user": "exact user line from transcript", "character": "exact character line from transcript"}
  ]
}

Guidelines:
- Extract at minimum: 2 goals, 2 quirks, 2 topics_avoided, 2 topics_embraced.
- Extract 1-3 plausible secrets inferred from evasions, contradictions, or
  charged reactions. Choose reveal conditions that match the character's
  guardedness: suspicious or guarded characters → high thresholds (0.7-0.9)
  or keyword triggers; open characters → explicit_ask or lower thresholds.
- sample_dialogue: include 3-5 representative exchange pairs verbatim.
- Do NOT invent details that contradict the transcript. When genuinely
  unknown, say so in the value text and set inferred: true.
- initial_state should reflect the character's situation at the start of
  the transcript (location, mood, current activity, etc.).
"""


# ── Transcript parsing ────────────────────────────────────────────────────────

def parse_transcript(
    path: Path,
    char_label: str | None = None,
) -> tuple[str, str, list[dict]]:
    """
    Parse a plain-text transcript into turns.

    Returns (user_label, character_label, turns) where turns is a list of
    {"speaker": str, "text": str} dicts.

    Transcript lines must be formatted as:
        Speaker Name: message text here
    """
    text = path.read_text(encoding="utf-8")

    # Match "Speaker Name: rest of line" — label may include spaces, hyphens, apostrophes
    line_re = re.compile(r"^([A-Za-z][A-Za-z0-9 '_-]*):\s*(.+)", re.MULTILINE)

    turns: list[dict] = []
    seen: list[str] = []  # ordered unique speakers

    for m in line_re.finditer(text):
        speaker = m.group(1).strip()
        content = m.group(2).strip()
        turns.append({"speaker": speaker, "text": content})
        if speaker not in seen:
            seen.append(speaker)

    if not turns:
        raise ValueError(
            "No turns detected. Ensure each line is formatted as 'Speaker: text'."
        )
    if len(seen) < 2:
        raise ValueError(
            f"Only one speaker detected ({seen}). "
            "Transcripts must have at least two speakers."
        )

    if char_label:
        if char_label not in seen:
            raise ValueError(
                f"--character-label '{char_label}' not found in transcript. "
                f"Detected speakers: {seen}"
            )
        user_label = next(s for s in seen if s != char_label)
    else:
        # Heuristic: prefer explicit "User" or "Human" label as the user side
        user_candidates = [s for s in seen if s.lower() in ("user", "human")]
        if user_candidates:
            user_label = user_candidates[0]
            char_label = next(s for s in seen if s != user_label)
        else:
            user_label, char_label = seen[0], seen[1]
            print(
                f"[info] Assuming '{user_label}' = user, '{char_label}' = character. "
                "Use --character-label to override.",
                file=sys.stderr,
            )

    return user_label, char_label, turns


def format_transcript(turns: list[dict], user_label: str, char_label: str) -> str:
    """Normalise speaker labels to 'User' / char_label for the API prompt."""
    lines = []
    for t in turns:
        label = "User" if t["speaker"] == user_label else char_label
        lines.append(f"{label}: {t['text']}")
    return "\n".join(lines)


# ── API call ──────────────────────────────────────────────────────────────────

def extract_character(transcript_text: str, char_label: str, model: str) -> dict:
    """Send transcript to the Anthropic API and return parsed extraction JSON."""
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("anthropic package not installed — run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env or export it.")

    client = Anthropic(api_key=api_key)

    user_message = (
        f"CHARACTER LABEL IN TRANSCRIPT: {char_label}\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    print(f"[info] Sending transcript to {model}...", file=sys.stderr)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences the model might add
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[error] Could not parse API response as JSON: {exc}", file=sys.stderr)
        print("[debug] First 800 chars of response:", file=sys.stderr)
        print(raw[:800], file=sys.stderr)
        sys.exit(1)


# ── YAML rendering ────────────────────────────────────────────────────────────

_INFERRED_TAG = "  # inferred - verify"


def _ind(text: str, spaces: int) -> str:
    """Indent every non-empty line; leave blank lines truly blank."""
    pad = " " * spaces
    return "\n".join(
        pad + line if line.strip() else ""
        for line in text.strip().splitlines()
    )


def _blk(inferred: bool) -> str:
    """Return YAML block scalar indicator with optional inferred comment."""
    return f"|{_INFERRED_TAG if inferred else ''}"


def _safe_scalar(value: str) -> str:
    """
    Return value as a YAML plain scalar or double-quoted string.
    Quotes if the value contains characters that would break plain YAML.
    """
    needs_quotes = any(c in str(value) for c in ':#{}[]|>&*!,\'"\\')
    if needs_quotes:
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)


def render_yaml(data: dict, notes: str | None, transcript_path: str) -> str:
    """Render extracted character data as a Persona Engine–compatible YAML string."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    char_name = data.get("character_name", {}).get("value", "Unknown")
    name_inferred = data.get("character_name", {}).get("inferred", True)

    # ── File header ───────────────────────────────────────────────────────────
    lines += [
        f"# {char_name}",
        f"# Generated by ingest_transcript.py",
        f"# Source: {Path(transcript_path).name}",
        f"# Generated: {now}",
    ]
    if notes:
        lines.append(f"# Provenance: {notes}")
    lines += [
        f"# Fields marked '{_INFERRED_TAG.strip()}' were deduced from context.",
        "# Review and edit all inferred fields before use in production.",
        "",
    ]

    # ── name ──────────────────────────────────────────────────────────────────
    name_comment = _INFERRED_TAG if name_inferred else ""
    lines.append(f"name: {char_name}{name_comment}")
    lines.append("")

    # ── Block scalar fields ───────────────────────────────────────────────────
    for key in ("description", "personality", "background", "speaking_style"):
        field = data.get(key, {})
        value = field.get("value", "")
        inferred = field.get("inferred", True)
        lines.append(f"{key}: {_blk(inferred)}")
        lines.append(_ind(value, 2))
        lines.append("")

    # ── goals ─────────────────────────────────────────────────────────────────
    lines.append("goals:")
    for item in data.get("goals", []):
        tag = _INFERRED_TAG if item.get("inferred", True) else ""
        lines.append(f"  - {item['value']}{tag}")
    lines.append("")

    # ── quirks ────────────────────────────────────────────────────────────────
    lines.append("quirks:")
    for item in data.get("quirks", []):
        tag = _INFERRED_TAG if item.get("inferred", True) else ""
        lines.append(f"  - {item['value']}{tag}")
    lines.append("")

    # ── relationships ─────────────────────────────────────────────────────────
    rels = data.get("relationships", [])
    if rels:
        lines.append("relationships:")
        for rel in rels:
            lines.append(f"  {rel['name']}: {_blk(rel.get('inferred', True))}")
            lines.append(_ind(rel["description"], 4))
        lines.append("")

    # ── secrets ───────────────────────────────────────────────────────────────
    secrets = data.get("secrets", [])
    if secrets:
        lines.append("secrets:")
        for s in secrets:
            lines.append(f"  - content: {_blk(s.get('inferred', True))}")
            lines.append(_ind(s["content"], 6))
            rc = s.get("reveal_condition", "trust_threshold")
            lines.append(f"    reveal_condition: {rc}")
            if rc == "trust_threshold" and s.get("threshold") is not None:
                lines.append(f"    threshold: {s['threshold']}")
            if rc in ("keyword_trigger", "emotional_state") and s.get("triggers"):
                lines.append("    triggers:")
                for t in s["triggers"]:
                    lines.append(f"      - {t}")
        lines.append("")

    # ── initial_state ─────────────────────────────────────────────────────────
    initial = data.get("initial_state", {})
    state_val = initial.get("value", {})
    if state_val:
        state_tag = _INFERRED_TAG if initial.get("inferred", True) else ""
        lines.append(f"initial_state:{state_tag}")
        for k, v in state_val.items():
            lines.append(f"  {k}: {_safe_scalar(v)}")
        lines.append("")

    # ── memory_config — sensible defaults ─────────────────────────────────────
    lines += [
        "memory_config:",
        "  short_term_limit: 10",
        "  long_term_limit: 50",
        "  episodic_limit: 20",
        "  consolidation_threshold: 5",
        "  relevance_decay: 0.95",
        "",
    ]

    # ── system_prefix ─────────────────────────────────────────────────────────
    sp = data.get("system_prefix", {})
    sp_val = sp.get("value", "")
    if sp_val:
        lines.append(f"system_prefix: {_blk(sp.get('inferred', True))}")
        lines.append(_ind(sp_val, 2))
        lines.append("")

    # ── version_a_prompt placeholder ──────────────────────────────────────────
    lines += [
        "# version_a_prompt: |",
        f"#   You are {char_name}.",
        "#   [Replace this with a complete static prompt for Version A mode.]",
        "",
    ]

    # ── Analysis notes (commented — not parsed by engine) ────────────────────
    lines.append("# --- Analysis Notes (not parsed by engine) ---")

    avoided = data.get("topics_avoided", [])
    if avoided:
        lines.append("#")
        lines.append("# Topics avoided:")
        for item in avoided:
            lines.append(f"#   - {item['value']}")

    embraced = data.get("topics_embraced", [])
    if embraced:
        lines.append("#")
        lines.append("# Topics embraced:")
        for item in embraced:
            lines.append(f"#   - {item['value']}")

    dialogue = data.get("sample_dialogue", [])
    if dialogue:
        lines.append("#")
        lines.append("# Sample dialogue from transcript:")
        for turn in dialogue[:5]:
            lines.append(f"#   User:      {turn.get('user', '')}")
            lines.append(f"#   Character: {turn.get('character', '')}")
            lines.append("#")

    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse-engineer a Persona Engine character YAML from a chat transcript."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transcript format — plain text, one turn per line:
  User: Hey, what's your name?
  Aria: Aria. And you are?
  User: Just passing through.
  Aria: Nobody just passes through here.

Examples:
  python ingest_transcript.py transcript.txt
  python ingest_transcript.py transcript.txt -o characters/aria.yaml
  python ingest_transcript.py transcript.txt --character-label Aria --notes "Session 1"
  python ingest_transcript.py transcript.txt --model claude-sonnet-4-6
        """,
    )
    parser.add_argument(
        "transcript",
        help="Path to the plain-text transcript file",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Write YAML to this file (default: stdout)",
    )
    parser.add_argument(
        "--character-label",
        metavar="LABEL",
        help=(
            "Speaker label for the character side of the transcript. "
            "Auto-detected if omitted."
        ),
    )
    parser.add_argument(
        "--notes",
        metavar="TEXT",
        help="Provenance note added as a comment at the top of the output YAML.",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Anthropic model to use (default: claude-opus-4-6)",
    )

    args = parser.parse_args()
    transcript_path = Path(args.transcript)

    if not transcript_path.exists():
        sys.exit(f"[error] File not found: {transcript_path}")

    # Parse transcript
    try:
        user_label, char_label, turns = parse_transcript(
            transcript_path, args.character_label
        )
    except ValueError as exc:
        sys.exit(f"[error] {exc}")

    print(
        f"[info] Parsed {len(turns)} turns — "
        f"user: '{user_label}', character: '{char_label}'",
        file=sys.stderr,
    )

    # Build normalised transcript text for the API
    transcript_text = format_transcript(turns, user_label, char_label)

    # Extract via API
    data = extract_character(transcript_text, char_label, args.model)

    # Render YAML
    yaml_output = render_yaml(data, args.notes, args.transcript)

    # Write output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_output, encoding="utf-8")
        print(f"[info] Written to: {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(yaml_output)


if __name__ == "__main__":
    main()
