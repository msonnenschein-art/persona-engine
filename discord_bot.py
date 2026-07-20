"""Discord bot interface for the persona engine.

Routes messages through the same orchestrator the Streamlit chat page uses.
Each Discord user is permanently assigned 50/50 to Version A (static engine)
or Version B (dynamic engine).  All per-user data is stored by hashed ID only
— raw Discord IDs and usernames are never persisted.

Conversation state (messages, memory, emotional state) is written to SQLite
after every turn and restored on next contact, so conversations survive
bot restarts.

Commands:
    !help   — show usage
    !reset  — wipe the sender's session and engine state
    !status — show uptime, user counts, and session stats (public)

Run with: python discord_bot.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord.ext import tasks
from dotenv import load_dotenv

from bot_db import BotDB, DB_PATH
from core.llm_adapter import Message
from core.memory import TieredMemory
from core.orchestrator import PersonaOrchestrator, create_orchestrator
from core.schema import Character
from core.state import ConversationState

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("discord_bot")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHARS_DIR = Path("characters")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHARACTER = os.getenv("DISCORD_CHARACTER", "reva_sample")
PROVIDER = os.getenv("PERSONA_DEFAULT_PROVIDER", "anthropic")
MODEL = os.getenv("PERSONA_DEFAULT_MODEL", "claude-sonnet-4-6")
CHARACTER_PATH = CHARS_DIR / f"{DISCORD_CHARACTER}.yaml"

_hc_raw = os.getenv("HEARTBEAT_CHANNEL_ID", "")
HEARTBEAT_CHANNEL_ID: int | None = int(_hc_raw) if _hc_raw.strip() else None

MAX_DISCORD_MESSAGE_LENGTH = 2000
APOLOGY_LINE = (
    "...sorry, I lost my train of thought there. Give me a moment and try again?"
)
HELP_TEXT = (
    "I'm a character from the persona engine, dropped into Discord. "
    "DM me or @-mention me in a channel and I'll reply in character, "
    "remembering our conversation as we go.\n\n"
    "Commands:\n"
    "`!reset`  — wipe our conversation and start fresh\n"
    "`!status` — show bot uptime and usage stats\n"
    "`!help`   — show this message"
)

_bot_start_time: datetime = datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Orchestrator serialization helpers
# ---------------------------------------------------------------------------


def _orc_to_json(orc: PersonaOrchestrator) -> str:
    """Serialize orchestrator conversation state to a JSON string."""
    data: dict = {
        "character_name": orc.character.name,
        "mode": orc.mode.value,
        "messages": [{"role": m.role, "content": m.content} for m in orc.messages],
    }
    if orc.state:
        data["state"] = orc.state.to_dict()
    if orc.memory:
        data["memory"] = orc.memory.to_dict()
    return json.dumps(data)


def _orc_from_json(orc: PersonaOrchestrator, state_json: str) -> None:
    """Restore orchestrator conversation state from a JSON string (in-place)."""
    data = json.loads(state_json)
    orc.messages = [
        Message(role=m["role"], content=m["content"])
        for m in data.get("messages", [])
    ]
    if "state" in data and orc.mode.value == "b":
        orc.state = ConversationState.from_dict(data["state"])
    if "memory" in data and orc.mode.value == "b":
        mc = orc.character.memory_config
        orc.memory = TieredMemory.from_dict(
            data["memory"],
            short_term_limit=mc.short_term_limit,
            long_term_limit=mc.long_term_limit,
            episodic_limit=mc.episodic_limit,
            consolidation_threshold=mc.consolidation_threshold,
            relevance_decay=mc.relevance_decay,
        )


# ---------------------------------------------------------------------------
# Session store backed by SQLite
# ---------------------------------------------------------------------------


class SessionStore:
    """Per-user orchestrator sessions, persisted to SQLite.

    Keyed internally by hashed_id (never by raw Discord user ID).
    """

    def __init__(
        self,
        character_path: Path,
        provider: str,
        model: str,
        db: BotDB,
    ) -> None:
        self._character_path = character_path
        self._provider = provider
        self._model = model
        self._db = db
        # In-process cache — hashed_id → orchestrator
        self._cache: dict[str, PersonaOrchestrator] = {}
        # Current session ids — hashed_id → session_id
        self._session_ids: dict[str, int] = {}

    def get_or_create(
        self, discord_id: int
    ) -> tuple[PersonaOrchestrator, str, int]:
        """Return (orchestrator, hashed_id, session_id) for this Discord user.

        Creates the user record and assigns an engine version on first contact.
        Opens a new session if the previous one has timed out.
        Loads persisted engine state from the DB if present.
        """
        hashed_id, version = self._db.get_or_create_user(discord_id)

        session_id = self._db.get_active_session(hashed_id)
        if session_id is None:
            session_id = self._db.start_session(hashed_id, version)
        self._session_ids[hashed_id] = session_id

        if hashed_id not in self._cache:
            mode = "a" if version == "A" else "b"
            orc = create_orchestrator(
                self._character_path,
                provider=self._provider,
                mode=mode,
                model=self._model,
            )
            saved = self._db.load_engine_state(hashed_id)
            if saved:
                try:
                    _orc_from_json(orc, saved)
                except Exception:
                    logger.exception(
                        "Failed to restore engine state for hashed_id=%s...; "
                        "starting fresh.",
                        hashed_id[:8],
                    )
            self._cache[hashed_id] = orc

        return self._cache[hashed_id], hashed_id, session_id

    def after_turn(self, hashed_id: str, session_id: int) -> None:
        """Persist engine state and record the turn in the DB."""
        orc = self._cache[hashed_id]
        self._db.save_engine_state(hashed_id, _orc_to_json(orc))
        self._db.record_turn(session_id, hashed_id)

    def reset(self, discord_id: int) -> None:
        """Wipe the user's engine state and end their current session."""
        hashed_id = self._db.hash_id(discord_id)
        session_id = self._session_ids.pop(hashed_id, None)
        if session_id is not None:
            self._db.end_session(session_id)
        self._cache.pop(hashed_id, None)
        self._db.clear_engine_state(hashed_id)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def split_message(text: str, limit: int = MAX_DISCORD_MESSAGE_LENGTH) -> list[str]:
    """Split text into chunks under Discord's message-length cap at sentence boundaries."""
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
                chunks.append(sentence[i : i + limit])
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _uptime_str() -> str:
    delta = datetime.now(tz=timezone.utc) - _bot_start_time
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# ---------------------------------------------------------------------------
# Discord client
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

db = BotDB(DB_PATH)
sessions = SessionStore(CHARACTER_PATH, PROVIDER, MODEL, db)


def _strip_mention(content: str) -> str:
    if client.user is None:
        return content.strip()
    return re.sub(rf"<@!?{client.user.id}>", "", content).strip()


# ---------------------------------------------------------------------------
# Daily heartbeat task
# ---------------------------------------------------------------------------


@tasks.loop(hours=24)
async def daily_heartbeat() -> None:
    if HEARTBEAT_CHANNEL_ID is None:
        return
    channel = client.get_channel(HEARTBEAT_CHANNEL_ID)
    if channel is None:
        logger.warning("Heartbeat: channel %s not found.", HEARTBEAT_CHANNEL_ID)
        return
    try:
        stats = db.get_stats_summary()
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        line = (
            f"**Heartbeat** {now} | uptime {_uptime_str()} | "
            f"users {stats['total_users']} | "
            f"sessions (24h) {stats['sessions_24h']}"
        )
        db_file = discord.File(DB_PATH)
        await channel.send(line, file=db_file)
    except Exception:
        logger.exception("Heartbeat failed.")


@daily_heartbeat.before_loop
async def _before_heartbeat() -> None:
    await client.wait_until_ready()


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


@client.event
async def on_ready() -> None:
    print(
        f"Logged in as {client.user} — serving character '{DISCORD_CHARACTER}'",
        flush=True,
    )
    if not daily_heartbeat.is_running():
        daily_heartbeat.start()


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user is not None and client.user in message.mentions

    if not (is_dm or is_mentioned):
        return

    content = (
        _strip_mention(message.content) if is_mentioned else message.content.strip()
    )
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

    if lowered == "!status":
        try:
            stats = db.get_stats_summary()
            status_msg = (
                f"**Status**\n"
                f"Uptime: {_uptime_str()}\n"
                f"Total users: {stats['total_users']} "
                f"(A: {stats['users_a']}, B: {stats['users_b']})\n"
                f"Total sessions: {stats['total_sessions']}\n"
                f"Sessions (last 24h): {stats['sessions_24h']}\n"
                f"Model: {MODEL}"
            )
        except Exception:
            logger.exception("!status query failed.")
            status_msg = "Could not retrieve status right now."
        await message.channel.send(status_msg)
        return

    async with message.channel.typing():
        try:
            orc, hashed_id, session_id = sessions.get_or_create(message.author.id)
            reply = await asyncio.to_thread(orc.chat, content)
            await asyncio.to_thread(sessions.after_turn, hashed_id, session_id)
        except Exception:
            logger.exception("Engine call failed for user %s", message.author.id)
            await message.channel.send(APOLOGY_LINE)
            return

    for chunk in split_message(reply):
        await message.channel.send(chunk)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN not set — add it to .env")
    if not CHARACTER_PATH.exists():
        raise SystemExit(f"Character file not found: {CHARACTER_PATH}")

    Character.from_yaml(CHARACTER_PATH)  # fail fast on a broken character file

    try:
        client.run(DISCORD_BOT_TOKEN)
    except Exception:
        logger.exception("Bot crashed — restarting is handled by the process manager.")
        sys.exit(1)
