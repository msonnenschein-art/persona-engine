"""Discord bot interface for the persona engine.

Routes messages through the same orchestrator the Streamlit chat page uses —
Version B (dynamic) mode, one PersonaOrchestrator per conversation. This
module only wraps core/orchestrator.py; it does not duplicate any engine
logic.

Run with: python discord_bot.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

import discord
from dotenv import load_dotenv

from core.orchestrator import PersonaOrchestrator, create_orchestrator
from core.schema import Character

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_bot")

CHARS_DIR = Path("characters")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHARACTER = os.getenv("DISCORD_CHARACTER", "reva_sample")
PROVIDER = os.getenv("PERSONA_DEFAULT_PROVIDER", "anthropic")
CHARACTER_PATH = CHARS_DIR / f"{DISCORD_CHARACTER}.yaml"

MAX_DISCORD_MESSAGE_LENGTH = 2000
APOLOGY_LINE = "...sorry, I lost my train of thought there. Give me a moment and try again?"
HELP_TEXT = (
    "I'm a character from the persona engine, dropped into Discord. "
    "DM me or @-mention me in a channel and I'll reply in character, "
    "remembering our conversation as we go.\n\n"
    "Commands:\n"
    "`!reset` — wipe our conversation and start fresh\n"
    "`!help` — show this message"
)


class SessionStore:
    """Per-user conversation sessions, keyed by Discord user ID.

    Sessions live in a plain dict for now (in-memory only, lost on
    restart). All access goes through get_or_create()/reset() so a real
    persistence layer can be dropped in later without touching call sites.
    """

    def __init__(self, character_path: Path, provider: str):
        self._character_path = character_path
        self._provider = provider
        self._sessions: dict[int, PersonaOrchestrator] = {}

    def get_or_create(self, user_id: int) -> PersonaOrchestrator:
        if user_id not in self._sessions:
            self._sessions[user_id] = create_orchestrator(
                self._character_path, provider=self._provider, mode="b",
            )
        return self._sessions[user_id]

    def reset(self, user_id: int) -> None:
        self._sessions.pop(user_id, None)


def split_message(text: str, limit: int = MAX_DISCORD_MESSAGE_LENGTH) -> list[str]:
    """Split text into chunks under Discord's message length cap, breaking at sentence boundaries."""
    if len(text) <= limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), limit):
                chunks.append(sentence[i:i + limit])
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
sessions = SessionStore(CHARACTER_PATH, PROVIDER)


def _strip_mention(content: str) -> str:
    if client.user is None:
        return content.strip()
    return re.sub(rf"<@!?{client.user.id}>", "", content).strip()


@client.event
async def on_ready():
    print(f"Logged in as {client.user} — serving character '{DISCORD_CHARACTER}'")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user is not None and client.user in message.mentions

    if not (is_dm or is_mentioned):
        return

    content = _strip_mention(message.content) if is_mentioned else message.content.strip()
    if not content:
        return

    lowered = content.lower()
    if lowered == "!help":
        await message.channel.send(HELP_TEXT)
        return

    if lowered == "!reset":
        sessions.reset(message.author.id)
        await message.channel.send("Session reset — starting fresh.")
        return

    async with message.channel.typing():
        try:
            orc = sessions.get_or_create(message.author.id)
            reply = await asyncio.to_thread(orc.chat, content)
        except Exception:
            logger.exception("Engine call failed for user %s", message.author.id)
            await message.channel.send(APOLOGY_LINE)
            return

    for chunk in split_message(reply):
        await message.channel.send(chunk)


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN not set — add it to .env")
    if not CHARACTER_PATH.exists():
        raise SystemExit(f"Character file not found: {CHARACTER_PATH}")

    Character.from_yaml(CHARACTER_PATH)  # fail fast on a broken character file
    client.run(DISCORD_BOT_TOKEN)
