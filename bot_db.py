"""SQLite persistence layer for the Discord bot.

Tables
------
users         — hashed Discord IDs, A/B assignments, first_seen, days_active (JSON list)
sessions      — per-conversation session records with turn counts and timestamps
engine_states — serialized PersonaOrchestrator state so conversations survive restarts

Raw Discord IDs and usernames are never stored.  All user records are keyed by
SHA-256(HASH_SALT + discord_id) so the link between hash and identity is
unrecoverable without the salt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_MINUTES = 30

_HASH_SALT: str = os.getenv("HASH_SALT", "")
if not _HASH_SALT:
    logger.warning(
        "HASH_SALT env var is not set — user IDs will be hashed without a salt. "
        "Set HASH_SALT in .env for any deployment beyond local testing."
    )

DB_PATH: str = os.getenv("DB_PATH", "./data/bot.db")


class BotDB:
    """Thread-safe SQLite wrapper for Discord bot persistence."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    hashed_id  TEXT PRIMARY KEY,
                    ab_version TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    days_active TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    hashed_id   TEXT NOT NULL,
                    version     TEXT NOT NULL,
                    started_at  TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    ended_at    TEXT,
                    turn_count  INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_hashed_id
                    ON sessions (hashed_id);

                CREATE TABLE IF NOT EXISTS engine_states (
                    hashed_id  TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # ID hashing
    # ------------------------------------------------------------------

    def hash_id(self, discord_id: int) -> str:
        """Return the SHA-256 hash of HASH_SALT+discord_id."""
        raw = (_HASH_SALT + str(discord_id)).encode()
        return hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------------
    # User records
    # ------------------------------------------------------------------

    def get_or_create_user(self, discord_id: int) -> tuple[str, str]:
        """Return (hashed_id, ab_version).  Creates the user record on first contact."""
        hashed_id = self.hash_id(discord_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT ab_version FROM users WHERE hashed_id = ?",
                (hashed_id,),
            ).fetchone()
            if row:
                return hashed_id, row["ab_version"]

            version = "A" if random.random() < 0.5 else "B"
            now = datetime.utcnow().isoformat()
            self._conn.execute(
                "INSERT INTO users (hashed_id, ab_version, first_seen, days_active) "
                "VALUES (?, ?, ?, '[]')",
                (hashed_id, version, now),
            )
            self._conn.commit()
            logger.info("New user registered — version=%s", version)
            return hashed_id, version

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def get_active_session(self, hashed_id: str) -> int | None:
        """Return the session_id of the current open session, or None if timed out."""
        cutoff = (
            datetime.utcnow() - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
        ).isoformat()
        row = self._conn.execute(
            "SELECT id FROM sessions "
            "WHERE hashed_id = ? AND ended_at IS NULL AND last_active > ? "
            "ORDER BY started_at DESC LIMIT 1",
            (hashed_id, cutoff),
        ).fetchone()
        return int(row["id"]) if row else None

    def start_session(self, hashed_id: str, version: str) -> int:
        """Open a new session and return its id."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions "
                "(hashed_id, version, started_at, last_active, turn_count) "
                "VALUES (?, ?, ?, ?, 0)",
                (hashed_id, version, now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        """Mark a session as ended."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (now, session_id),
            )
            self._conn.commit()

    def record_turn(self, session_id: int, hashed_id: str) -> None:
        """Increment turn count, refresh last_active, and track today in days_active."""
        today = datetime.utcnow().date().isoformat()
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET turn_count = turn_count + 1, last_active = ? "
                "WHERE id = ?",
                (now, session_id),
            )
            row = self._conn.execute(
                "SELECT days_active FROM users WHERE hashed_id = ?",
                (hashed_id,),
            ).fetchone()
            if row:
                days: list[str] = json.loads(row["days_active"])
                if today not in days:
                    days.append(today)
                    self._conn.execute(
                        "UPDATE users SET days_active = ? WHERE hashed_id = ?",
                        (json.dumps(days), hashed_id),
                    )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Engine state persistence
    # ------------------------------------------------------------------

    def save_engine_state(self, hashed_id: str, state_json: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO engine_states (hashed_id, state_json, updated_at) "
                "VALUES (?, ?, ?)",
                (hashed_id, state_json, now),
            )
            self._conn.commit()

    def load_engine_state(self, hashed_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT state_json FROM engine_states WHERE hashed_id = ?",
            (hashed_id,),
        ).fetchone()
        return row["state_json"] if row else None

    def clear_engine_state(self, hashed_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM engine_states WHERE hashed_id = ?",
                (hashed_id,),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Quick stats for !status and heartbeat
    # ------------------------------------------------------------------

    def get_stats_summary(self) -> dict[str, int]:
        """Return a dict of headline numbers — no user-identifiable data."""
        total_users: int = self._conn.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]
        users_a: int = self._conn.execute(
            "SELECT COUNT(*) FROM users WHERE ab_version = 'A'"
        ).fetchone()[0]
        users_b: int = self._conn.execute(
            "SELECT COUNT(*) FROM users WHERE ab_version = 'B'"
        ).fetchone()[0]
        total_sessions: int = self._conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        sessions_24h: int = self._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE started_at > ?",
            (cutoff_24h,),
        ).fetchone()[0]
        return {
            "total_users": total_users,
            "users_a": users_a,
            "users_b": users_b,
            "total_sessions": total_sessions,
            "sessions_24h": sessions_24h,
        }
