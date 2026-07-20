"""Automated checks for bot_db.py and stats.py.

Tests (without running the Discord bot or touching any LLM API):
  1. Two fake user IDs get stable, independent A/B assignments.
  2. Sessions are created and isolated per user.
  3. Turn counts are recorded correctly.
  4. The stats module computes correct numbers from that data.
  5. Engine state survives a simulated restart (new BotDB instance, same file).

Run with: python test_bot.py
"""
from __future__ import annotations

import json
import os
import tempfile
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Point at a temporary DB so this test never touches the real data/bot.db
_tmpdir = tempfile.mkdtemp()
_TEST_DB = str(Path(_tmpdir) / "test_bot.db")
os.environ["DB_PATH"] = _TEST_DB
os.environ["HASH_SALT"] = "test-salt-abc123"

from bot_db import BotDB  # noqa: E402 — env vars must be set first
import stats as stats_mod  # noqa: E402


FAKE_USER_A = 111111111111111111
FAKE_USER_B = 222222222222222222

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures: list[str] = []


def check(label: str, condition: bool) -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        _failures.append(label)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

db = BotDB(_TEST_DB)


# ---------------------------------------------------------------------------
# Test 1 — A/B assignment stability
# ---------------------------------------------------------------------------

print("\n[1] A/B assignment stability")

hid_a1, ver_a1 = db.get_or_create_user(FAKE_USER_A)
hid_a2, ver_a2 = db.get_or_create_user(FAKE_USER_A)
check("Same hashed_id on repeated calls", hid_a1 == hid_a2)
check("Same version on repeated calls", ver_a1 == ver_a2)

hid_b1, ver_b1 = db.get_or_create_user(FAKE_USER_B)
check("Different hashed_id for different users", hid_a1 != hid_b1)
check("Version is A or B", ver_a1 in ("A", "B") and ver_b1 in ("A", "B"))


# ---------------------------------------------------------------------------
# Test 2 — Independent sessions
# ---------------------------------------------------------------------------

print("\n[2] Session isolation")

sess_a = db.get_active_session(hid_a1)
check("No active session before first start", sess_a is None)

sid_a = db.start_session(hid_a1, ver_a1)
sid_b = db.start_session(hid_b1, ver_b1)
check("Session IDs are distinct", sid_a != sid_b)
check("User A has an active session", db.get_active_session(hid_a1) == sid_a)
check("User B has an active session", db.get_active_session(hid_b1) == sid_b)


# ---------------------------------------------------------------------------
# Test 3 — Turn counts
# ---------------------------------------------------------------------------

print("\n[3] Turn counts")

db.record_turn(sid_a, hid_a1)
db.record_turn(sid_a, hid_a1)
db.record_turn(sid_a, hid_a1)
db.record_turn(sid_b, hid_b1)

import sqlite3
conn = sqlite3.connect(_TEST_DB)
conn.row_factory = sqlite3.Row
turns_a = conn.execute("SELECT turn_count FROM sessions WHERE id = ?", (sid_a,)).fetchone()["turn_count"]
turns_b = conn.execute("SELECT turn_count FROM sessions WHERE id = ?", (sid_b,)).fetchone()["turn_count"]
conn.close()

check("User A has 3 turns", turns_a == 3)
check("User B has 1 turn", turns_b == 1)


# ---------------------------------------------------------------------------
# Test 4 — Stats module
# ---------------------------------------------------------------------------

print("\n[4] Stats module")

db.end_session(sid_a)
db.end_session(sid_b)

upv = stats_mod.users_per_version(_TEST_DB)
check("users_per_version returns both versions", sum(upv.values()) == 2)

spu = stats_mod.sessions_per_user(_TEST_DB)
check("sessions_per_user == 1.0 (one session each)", spu == 1.0)

atps = stats_mod.avg_turns_per_session(_TEST_DB)
check("avg_turns_per_session == 2.0 ((3+1)/2)", atps == 2.0)

d1 = stats_mod.d1_return_rate(_TEST_DB)
check("D1 rate is 0.0 (no day-1 return data yet)", d1 == 0.0)

d7 = stats_mod.d7_return_rate(_TEST_DB)
check("D7 rate is 0.0 (no day-7 return data yet)", d7 == 0.0)


# ---------------------------------------------------------------------------
# Test 5 — Engine state survives restart
# ---------------------------------------------------------------------------

print("\n[5] Engine state persistence across restart")

dummy_state = json.dumps({
    "character_name": "TestChar",
    "mode": "b",
    "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
    "state": {"turn_count": 1, "emotional_state": {"primary": "neutral", "intensity": 0.5, "secondary": None},
               "relationship": {"trust_level": 0.3, "familiarity": 0.0, "rapport": 0.5, "tension": 0.0},
               "active_topics": [], "mentioned_entities": [], "revealed_secrets": [],
               "conversation_flags": {}, "custom_state": {},
               "session_start": datetime.utcnow().isoformat(),
               "last_interaction": datetime.utcnow().isoformat()},
})

db.save_engine_state(hid_a1, dummy_state)

# Simulate restart: new BotDB instance pointing at the same file
db2 = BotDB(_TEST_DB)
loaded = db2.load_engine_state(hid_a1)
check("Engine state is loadable after restart", loaded is not None)
if loaded:
    parsed = json.loads(loaded)
    check("Message history survives restart", len(parsed.get("messages", [])) == 2)
    check("Turn count survives restart", parsed.get("state", {}).get("turn_count") == 1)

# Clear state and confirm deletion
db2.clear_engine_state(hid_a1)
check("Engine state cleared after reset", db2.load_engine_state(hid_a1) is None)


# ---------------------------------------------------------------------------
# Test 6 — get_stats_summary
# ---------------------------------------------------------------------------

print("\n[6] get_stats_summary")

summary = db2.get_stats_summary()
check("total_users == 2", summary["total_users"] == 2)
check("total_sessions == 2", summary["total_sessions"] == 2)
check("sessions_24h == 2", summary["sessions_24h"] == 2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
if _failures:
    print(f"FAILED: {len(_failures)} check(s) — {', '.join(_failures)}")
    sys.exit(1)
else:
    print("All checks passed.")
