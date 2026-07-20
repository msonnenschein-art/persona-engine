"""Stats query module for the Discord bot persistence layer.

Functions here are shared between the bot (e.g. for the !status command) and
standalone analysis.  The module is self-contained — it opens its own read-only
SQLite connection rather than requiring a BotDB instance.

CLI usage
---------
    python stats.py               # reads DB_PATH env var (default: ./data/bot.db)
    python stats.py /path/to/db   # explicit path
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta

DB_PATH: str = os.getenv("DB_PATH", "./data/bot.db")


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def users_per_version(db_path: str | None = None) -> dict[str, int]:
    """Return {'A': n, 'B': n} user counts per engine version."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT ab_version, COUNT(*) AS cnt FROM users GROUP BY ab_version"
        ).fetchall()
    return {row["ab_version"]: row["cnt"] for row in rows}


def sessions_per_user(db_path: str | None = None) -> float:
    """Return average number of sessions per distinct user."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT hashed_id) AS users, COUNT(*) AS sessions "
            "FROM sessions"
        ).fetchone()
    if not row or not row["users"]:
        return 0.0
    return round(row["sessions"] / row["users"], 2)


def avg_turns_per_session(db_path: str | None = None) -> float:
    """Return average turn count across all sessions that have at least one turn."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT AVG(turn_count) AS avg_turns FROM sessions WHERE turn_count > 0"
        ).fetchone()
    if not row or row["avg_turns"] is None:
        return 0.0
    return round(float(row["avg_turns"]), 2)


def d1_return_rate(db_path: str | None = None) -> float:
    """Fraction of eligible users who were active on the day after first contact.

    Denominator: users registered >= 1 day ago (they've had a chance to return).
    Numerator:   those whose days_active list contains (first_seen_date + 1 day).
    """
    cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT first_seen, days_active FROM users WHERE first_seen <= ?",
            (cutoff,),
        ).fetchall()
    if not rows:
        return 0.0
    returned = sum(
        1
        for row in rows
        if (datetime.fromisoformat(row["first_seen"]).date() + timedelta(days=1)).isoformat()
        in json.loads(row["days_active"])
    )
    return round(returned / len(rows), 4)


def d7_return_rate(db_path: str | None = None) -> float:
    """Fraction of eligible users who were active on day 7 after first contact.

    Denominator: users registered >= 7 days ago.
    Numerator:   those whose days_active list contains (first_seen_date + 7 days).
    """
    cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT first_seen, days_active FROM users WHERE first_seen <= ?",
            (cutoff,),
        ).fetchall()
    if not rows:
        return 0.0
    returned = sum(
        1
        for row in rows
        if (datetime.fromisoformat(row["first_seen"]).date() + timedelta(days=7)).isoformat()
        in json.loads(row["days_active"])
    )
    return round(returned / len(rows), 4)


def print_summary(db_path: str | None = None) -> None:
    """Print a formatted stats table to stdout."""
    upv = users_per_version(db_path)
    total_users = sum(upv.values())
    a_count = upv.get("A", 0)
    b_count = upv.get("B", 0)
    spu = sessions_per_user(db_path)
    atps = avg_turns_per_session(db_path)
    d1 = d1_return_rate(db_path)
    d7 = d7_return_rate(db_path)

    print("=" * 52)
    print("  Persona Engine — Discord Bot Stats")
    print("=" * 52)
    print(f"  {'Total users':<28} {total_users}")
    print(f"  {'  Version A (static)':<28} {a_count}")
    print(f"  {'  Version B (dynamic)':<28} {b_count}")
    print(f"  {'Sessions per user':<28} {spu}")
    print(f"  {'Avg turns per session':<28} {atps}")
    print(f"  {'D1 return rate':<28} {d1:.1%}")
    print(f"  {'D7 return rate':<28} {d7:.1%}")
    print("=" * 52)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    db_arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        print_summary(db_arg)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
