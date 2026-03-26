#!/usr/bin/env python3
"""Persona Engine CLI - Interactive character AI conversations."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from core.orchestrator import create_orchestrator, Mode
from core.schema import Character
from core.comparison import BaselineComparison
from core.rag_manager import RAGManager


def print_header(character_name: str, mode: str) -> None:
    """Print the CLI header."""
    print("\n" + "=" * 60)
    print(f"  PERSONA ENGINE - {character_name}")
    print(f"  Mode: Version {'A (Static)' if mode == 'a' else 'B (Dynamic)'}")
    print("=" * 60)
    print("\nCommands:")
    print("  /quit, /exit  - End conversation")
    print("  /state        - Show current state (Version B only)")
    print("  /reset        - Reset conversation")
    print("  /save <file>  - Save session state")
    print("  /load <file>  - Load session state")
    print("  /mode <a|b>   - Switch mode")
    print("-" * 60 + "\n")


def run_compare(comparison: BaselineComparison, character_name: str) -> None:
    """Run the interactive baseline-comparison loop."""
    width = 100
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  PERSONA ENGINE — BASELINE COMPARISON  |  {character_name}")
    print(f"  Version A (static) vs Version B (dynamic) — same input, both responses")
    print(f"{bar}")
    print("  Type your message.  /quit to exit.  /reset to restart both sides.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("\nGoodbye!")
            break

        if user_input.lower() == "/reset":
            comparison.reset()
            print("\nBoth sides reset.\n")
            continue

        result = comparison.chat(user_input)
        print(result.format(terminal_width=width))


def print_state(orchestrator) -> None:
    """Print current conversation state."""
    summary = orchestrator.get_state_summary()
    print("\n--- Current State ---")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("-------------------\n")


def handle_command(cmd: str, orchestrator, args) -> bool:
    """Handle a CLI command. Returns True if should continue, False to exit."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if command in ("/quit", "/exit"):
        orchestrator.end_session()
        print("\nGoodbye!")
        return False

    elif command == "/state":
        print_state(orchestrator)

    elif command == "/reset":
        orchestrator.reset()
        print("\nConversation reset.\n")

    elif command == "/save":
        if not arg:
            print("\nUsage: /save <filename>\n")
        else:
            try:
                orchestrator.save_state(arg)
                print(f"\nSession saved to {arg}\n")
            except Exception as e:
                print(f"\nError saving: {e}\n")

    elif command == "/load":
        if not arg:
            print("\nUsage: /load <filename>\n")
        else:
            try:
                orchestrator.load_state(arg)
                print(f"\nSession loaded from {arg}\n")
            except Exception as e:
                print(f"\nError loading: {e}\n")

    elif command == "/mode":
        if arg not in ("a", "b"):
            print("\nUsage: /mode <a|b>\n")
        else:
            new_mode = Mode.VERSION_A if arg == "a" else Mode.VERSION_B
            orchestrator.mode = new_mode
            if new_mode == Mode.VERSION_B and orchestrator.state is None:
                from core.state import ConversationState
                from core.memory import TieredMemory
                orchestrator.state = ConversationState()
                mc = orchestrator.character.memory_config
                orchestrator.memory = TieredMemory(
                    short_term_limit=mc.short_term_limit,
                    long_term_limit=mc.long_term_limit,
                    episodic_limit=mc.episodic_limit,
                )
            print(f"\nSwitched to Version {'A' if arg == 'a' else 'B'}\n")

    else:
        print(f"\nUnknown command: {command}\n")

    return True


def run_interactive(orchestrator, args) -> None:
    """Run interactive conversation loop."""
    print_header(orchestrator.character.name, args.mode)

    intro = f"*{orchestrator.character.name} is here*"
    print(f"{orchestrator.character.name}: {intro}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            orchestrator.end_session()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, orchestrator, args):
                break
            continue

        print(f"\n{orchestrator.character.name}: ", end="", flush=True)

        if args.stream:
            for chunk in orchestrator.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            response = orchestrator.chat(user_input)
            print(f"{response}\n")


def run_single(orchestrator, message: str, stream: bool = False) -> None:
    """Run a single message exchange."""
    if stream:
        for chunk in orchestrator.chat_stream(message):
            print(chunk, end="", flush=True)
        print()
    else:
        response = orchestrator.chat(message)
        print(response)


def list_characters(characters_dir: Path) -> None:
    """List available character files."""
    print("\nAvailable characters:")
    print("-" * 40)

    yaml_files = list(characters_dir.glob("*.yaml")) + list(characters_dir.glob("*.yml"))

    if not yaml_files:
        print("  No character files found.")
        print(f"  Add YAML files to: {characters_dir}")
    else:
        for f in sorted(yaml_files):
            try:
                char = Character.from_yaml(f)
                print(f"  {f.stem}: {char.name}")
                if char.description:
                    desc = char.description[:60] + "..." if len(char.description) > 60 else char.description
                    print(f"    {desc}")
            except Exception as e:
                print(f"  {f.stem}: (error loading: {e})")

    print()


def main() -> None:
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Persona Engine - Character AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py -c reva_sample              # Interactive chat with Reva
  python cli.py -c reva_sample -m a         # Use Version A (static prompt)
  python cli.py -c reva_sample --message "Hello"  # Single message
  python cli.py --list                      # List available characters
  python cli.py -c reva_sample --provider openai  # Use OpenAI
        """,
    )

    parser.add_argument(
        "-c", "--character",
        help="Character file name (without .yaml) or full path",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["a", "b"],
        default="b",
        help="Version A (static) or B (dynamic). Default: b",
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider. Default: anthropic",
    )
    parser.add_argument(
        "--model",
        help="Model name (e.g., claude-sonnet-4-20250514, gpt-4o)",
    )
    parser.add_argument(
        "--message",
        help="Single message to send (non-interactive mode)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream responses",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available characters",
    )
    parser.add_argument(
        "--characters-dir",
        default="characters",
        help="Directory containing character YAML files",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in response",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Baseline comparison mode: run same input through Version A and B side by side",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG cold memory tier (indexes knowledge/ on first run)",
    )
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="Directory containing knowledge base documents (default: knowledge/)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    characters_dir = script_dir / args.characters_dir

    if args.list:
        list_characters(characters_dir)
        return

    if not args.character:
        parser.print_help()
        print("\nError: --character is required (or use --list to see available characters)")
        sys.exit(1)

    character_path = Path(args.character)
    if not character_path.exists():
        character_path = characters_dir / f"{args.character}.yaml"
        if not character_path.exists():
            character_path = characters_dir / f"{args.character}.yml"

    if not character_path.exists():
        print(f"Error: Character file not found: {args.character}")
        print(f"Searched in: {characters_dir}")
        list_characters(characters_dir)
        sys.exit(1)

    adapter_kwargs = {"max_tokens": args.max_tokens}
    if args.model:
        adapter_kwargs["model"] = args.model

    # Optional RAG manager (cold memory tier)
    rag = None
    if args.rag:
        knowledge_dir = script_dir / args.knowledge_dir
        rag = RAGManager(knowledge_dir=knowledge_dir)
        if rag.document_count == 0:
            count = rag.ingest_directory()
            if count:
                print(f"[RAG] Indexed {count} chunks from {knowledge_dir}")
        else:
            print(f"[RAG] Loaded collection ({rag.document_count} chunks)")

    try:
        if args.compare:
            comparison = BaselineComparison.from_character_path(
                character_path=str(character_path),
                provider=args.provider,
                **adapter_kwargs,
            )
            # Attach RAG to the Version B orchestrator only
            if rag is not None:
                comparison.orchestrator_b.rag = rag
            run_compare(comparison, character_name=comparison.orchestrator_b.character.name)
        else:
            orchestrator = create_orchestrator(
                character_path=character_path,
                provider=args.provider,
                mode=args.mode,
                rag=rag,
                **adapter_kwargs,
            )
            if args.message:
                run_single(orchestrator, args.message, stream=args.stream)
            else:
                run_interactive(orchestrator, args)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nMake sure your API key is set:")
        print("  - ANTHROPIC_API_KEY for Anthropic")
        print("  - OPENAI_API_KEY for OpenAI")
        sys.exit(1)


if __name__ == "__main__":
    main()
