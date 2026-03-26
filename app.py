"""
app.py — Persona Engine multi-page UI

Pages (sidebar):
  1. Character Creator  — build / edit / save character YAML
  2. Knowledge Base     — upload documents, rebuild RAG index
  3. Rubric Builder     — create / edit evaluation rubrics
  4. Chat               — interactive conversation with any character

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Directory constants ───────────────────────────────────────────────────────

CHARS_DIR     = Path("characters")
KNOWLEDGE_DIR = Path("knowledge")
RUBRICS_DIR   = Path("rubrics")

for _d in (CHARS_DIR, KNOWLEDGE_DIR, RUBRICS_DIR):
    _d.mkdir(exist_ok=True)

# ── YAML helpers ──────────────────────────────────────────────────────────────

class _Literal(str):
    """Emitted as a YAML literal block scalar (|)."""

class _Folded(str):
    """Emitted as a YAML folded block scalar (>)."""

def _literal_rep(dumper: yaml.Dumper, data: _Literal) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

def _folded_rep(dumper: yaml.Dumper, data: _Folded) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")

class _Dumper(yaml.Dumper):
    pass

_Dumper.add_representer(_Literal, _literal_rep)
_Dumper.add_representer(_Folded,  _folded_rep)


def _block(text: str) -> _Literal | str:
    """Use literal block scalar for multiline / long text; plain scalar otherwise."""
    text = text.strip()
    if not text:
        return text
    if "\n" in text or len(text) > 72:
        return _Literal(text + "\n")
    return text


def _folded(text: str) -> _Folded | str:
    """Use folded block scalar for prose paragraphs."""
    text = text.strip()
    if not text:
        return text
    return _Folded(text + "\n")


def _yaml_dump(data: dict) -> str:
    return yaml.dump(
        data,
        Dumper=_Dumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=88,
    )


# ── Session-state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    """Seed session state keys that must exist before widgets render."""
    char_defaults: dict = {
        "name": "", "description": "", "personality": "", "background": "",
        "speaking_style": "", "fundamental_desire": "", "lived_in_genre": "",
        "system_prefix": "", "subtextuality": 5, "digressiveness": 5,
        # Dynamic-row counters
        "vs_ids": [0],    "vs_next":    1,
        "goal_ids": [0],  "goal_next":  1,
        "quirk_ids": [0], "quirk_next": 1,
        "rel_ids": [0],   "rel_next":   1,
        "sec_ids": [0],   "sec_next":   1,
        # Initial widget values for the first row of each dynamic section
        "vs_ctx_0": "", "vs_exc_0": "",
        "goal_0": "",   "quirk_0": "",
        "rel_name_0": "", "rel_desc_0": "",
        "sec_content_0": "", "sec_condition_0": "trust_threshold",
        "sec_threshold_0": 0.75, "sec_triggers_0": "",
    }
    rubric_defaults: dict = {
        "rl_name": "", "rl_version": "1.0", "rl_description": "",
        "rl_weight": 1.0, "rl_notes": "",
        "rl_crit_ids": [0], "rl_crit_next": 1,
        "rl_crit_id_0": "", "rl_crit_label_0": "", "rl_crit_prompt_0": "",
        **{f"rl_scale_{i}": "" for i in range(1, 6)},
    }
    chat_defaults: dict = {
        "chat_messages": [],
        "chat_orc": None,
        "chat_comp": None,
        "chat_active_config": None,
    }
    ingester_defaults: dict = {
        "ingester_yaml": None,
        "ingester_char_name": "character",
    }
    for defaults in (char_defaults, rubric_defaults, chat_defaults, ingester_defaults):
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v


# ── Dynamic-row helpers (shared by all pages) ─────────────────────────────────

def _add(id_key: str, next_key: str) -> None:
    st.session_state[id_key] = list(st.session_state[id_key]) + [st.session_state[next_key]]
    st.session_state[next_key] += 1

def _remove(id_key: str, item_id: int) -> None:
    st.session_state[id_key] = [i for i in st.session_state[id_key] if i != item_id]


# ── Character helpers ─────────────────────────────────────────────────────────

def _list_chars() -> list[str]:
    return sorted(p.stem for p in CHARS_DIR.glob("*.yaml"))


def _load_character(path: Path) -> None:
    """Populate character session state from a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Scalar fields
    for key in ("name", "description", "personality", "background",
                "speaking_style", "fundamental_desire", "lived_in_genre",
                "system_prefix"):
        st.session_state[key] = str(data.get(key, "") or "").strip()

    st.session_state["subtextuality"]  = int(data.get("subtextuality", 5))
    st.session_state["digressiveness"] = int(data.get("digressiveness", 5))

    # Voice samples
    vs_data = data.get("voice_samples", [])
    ids = list(range(max(len(vs_data), 1)))
    st.session_state["vs_ids"] = ids
    st.session_state["vs_next"] = len(ids)
    for i in ids:
        vs = vs_data[i] if i < len(vs_data) else {}
        st.session_state[f"vs_ctx_{i}"] = str(vs.get("context", "") or "").strip()
        st.session_state[f"vs_exc_{i}"] = str(vs.get("exchange", "") or "").strip()

    # Goals
    goals = data.get("goals", [])
    ids = list(range(max(len(goals), 1)))
    st.session_state["goal_ids"] = ids
    st.session_state["goal_next"] = len(ids)
    for i in ids:
        st.session_state[f"goal_{i}"] = goals[i] if i < len(goals) else ""

    # Quirks
    quirks = data.get("quirks", [])
    ids = list(range(max(len(quirks), 1)))
    st.session_state["quirk_ids"] = ids
    st.session_state["quirk_next"] = len(ids)
    for i in ids:
        st.session_state[f"quirk_{i}"] = quirks[i] if i < len(quirks) else ""

    # Relationships (dict → list of (name, desc))
    rels = list((data.get("relationships") or {}).items())
    ids = list(range(max(len(rels), 1)))
    st.session_state["rel_ids"] = ids
    st.session_state["rel_next"] = len(ids)
    for i in ids:
        if i < len(rels):
            st.session_state[f"rel_name_{i}"] = str(rels[i][0])
            st.session_state[f"rel_desc_{i}"] = str(rels[i][1] or "").strip()
        else:
            st.session_state[f"rel_name_{i}"] = ""
            st.session_state[f"rel_desc_{i}"] = ""

    # Secrets
    secrets = data.get("secrets", [])
    ids = list(range(max(len(secrets), 1)))
    st.session_state["sec_ids"] = ids
    st.session_state["sec_next"] = len(ids)
    for i in ids:
        s = secrets[i] if i < len(secrets) else {}
        st.session_state[f"sec_content_{i}"]   = str(s.get("content", "") or "").strip()
        st.session_state[f"sec_condition_{i}"]  = s.get("reveal_condition", "trust_threshold")
        st.session_state[f"sec_threshold_{i}"]  = float(s.get("threshold", 0.75) or 0.75)
        triggers = s.get("triggers", [])
        st.session_state[f"sec_triggers_{i}"] = (
            ", ".join(triggers) if isinstance(triggers, list) else str(triggers or "")
        )


def _build_char_yaml() -> str:
    ss = st.session_state
    data: dict = {}

    name = ss.get("name", "").strip()
    data["name"] = name or "Unnamed Character"

    for key in ("description", "personality", "background", "speaking_style"):
        if val := ss.get(key, "").strip():
            data[key] = _block(val)

    # Voice samples
    samples = []
    for sid in ss.get("vs_ids", []):
        exc = ss.get(f"vs_exc_{sid}", "").strip()
        ctx = ss.get(f"vs_ctx_{sid}", "").strip()
        if exc:
            entry: dict = {}
            if ctx:
                entry["context"] = ctx
            entry["exchange"] = _block(exc)
            samples.append(entry)
    if samples:
        data["voice_samples"] = samples

    # Goals / quirks
    goals = [g for gid in ss.get("goal_ids", []) if (g := ss.get(f"goal_{gid}", "").strip())]
    if goals:
        data["goals"] = goals
    quirks = [q for qid in ss.get("quirk_ids", []) if (q := ss.get(f"quirk_{qid}", "").strip())]
    if quirks:
        data["quirks"] = quirks

    # Relationships
    rels: dict = {}
    for rid in ss.get("rel_ids", []):
        rname = ss.get(f"rel_name_{rid}", "").strip()
        rdesc = ss.get(f"rel_desc_{rid}", "").strip()
        if rname and rdesc:
            rels[rname] = _block(rdesc)
    if rels:
        data["relationships"] = rels

    # Secrets
    _CONDITIONS = ("trust_threshold", "keyword_trigger", "emotional_state", "explicit_ask", "never")
    secrets = []
    for sid in ss.get("sec_ids", []):
        content = ss.get(f"sec_content_{sid}", "").strip()
        if not content:
            continue
        condition = ss.get(f"sec_condition_{sid}", "trust_threshold")
        entry = {"content": _block(content), "reveal_condition": condition}
        if condition == "trust_threshold":
            entry["threshold"] = round(float(ss.get(f"sec_threshold_{sid}", 0.75)), 2)
        if condition in ("keyword_trigger", "emotional_state"):
            raw = ss.get(f"sec_triggers_{sid}", "")
            triggers = [t.strip() for t in raw.split(",") if t.strip()]
            if triggers:
                entry["triggers"] = triggers
        secrets.append(entry)
    if secrets:
        data["secrets"] = secrets

    # Dramatic register
    if fd := ss.get("fundamental_desire", "").strip():
        data["fundamental_desire"] = _block(fd)
    if lg := ss.get("lived_in_genre", "").strip():
        data["lived_in_genre"] = lg
    sub = ss.get("subtextuality", 5)
    if sub != 5:
        data["subtextuality"] = sub
    dig = ss.get("digressiveness", 5)
    if dig != 5:
        data["digressiveness"] = dig

    # Engine defaults
    data["memory_config"] = {
        "short_term_limit": 10, "long_term_limit": 50,
        "episodic_limit": 20, "consolidation_threshold": 5,
        "relevance_decay": 0.95,
    }

    if sp := ss.get("system_prefix", "").strip():
        data["system_prefix"] = _block(sp)

    return _yaml_dump(data)


def _save_character() -> str | None:
    """Write character YAML to disk. Returns filename on success, None on error."""
    name = st.session_state.get("name", "").strip()
    if not name:
        return None
    filename = name.lower().replace(" ", "_") + ".yaml"
    (CHARS_DIR / filename).write_text(_build_char_yaml(), encoding="utf-8")
    return filename


# ── Rubric helpers ────────────────────────────────────────────────────────────

def _list_rubrics() -> list[str]:
    return sorted(p.stem for p in RUBRICS_DIR.glob("*.yaml"))


def _load_rubric(path: Path) -> None:
    """Populate rubric session state from a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    st.session_state["rl_name"]        = str(data.get("name", "") or "")
    st.session_state["rl_version"]     = str(data.get("version", "1.0") or "1.0")
    st.session_state["rl_description"] = str(data.get("description", "") or "").strip()
    st.session_state["rl_weight"]      = float(data.get("weight", 1.0) or 1.0)
    st.session_state["rl_notes"]       = str(data.get("notes", "") or "").strip()

    criteria = data.get("criteria", [])
    ids = list(range(max(len(criteria), 1)))
    st.session_state["rl_crit_ids"]  = ids
    st.session_state["rl_crit_next"] = len(ids)
    for i in ids:
        c = criteria[i] if i < len(criteria) else {}
        st.session_state[f"rl_crit_id_{i}"]     = str(c.get("id", "") or "")
        st.session_state[f"rl_crit_label_{i}"]  = str(c.get("label", "") or "")
        st.session_state[f"rl_crit_prompt_{i}"] = str(c.get("prompt", "") or "").strip()

    scale = data.get("scale", {})
    for i in range(1, 6):
        st.session_state[f"rl_scale_{i}"] = str(scale.get(i, "") or "")


def _build_rubric_yaml() -> str:
    ss = st.session_state
    data: dict = {
        "name":        ss.get("rl_name", "").strip(),
        "version":     ss.get("rl_version", "1.0").strip() or "1.0",
        "description": _folded(ss.get("rl_description", "")),
        "weight":      float(ss.get("rl_weight", 1.0)),
    }

    criteria = []
    for cid in ss.get("rl_crit_ids", []):
        c_id     = ss.get(f"rl_crit_id_{cid}", "").strip()
        c_label  = ss.get(f"rl_crit_label_{cid}", "").strip()
        c_prompt = ss.get(f"rl_crit_prompt_{cid}", "").strip()
        if c_id:
            entry: dict = {"id": c_id}
            if c_label:
                entry["label"] = c_label
            if c_prompt:
                entry["prompt"] = _folded(c_prompt)
            criteria.append(entry)
    if criteria:
        data["criteria"] = criteria

    scale = {}
    for i in range(1, 6):
        anchor = ss.get(f"rl_scale_{i}", "").strip()
        if anchor:
            scale[i] = anchor
    if scale:
        data["scale"] = scale

    if notes := ss.get("rl_notes", "").strip():
        data["notes"] = _folded(notes)

    return _yaml_dump(data)


def _save_rubric() -> str | None:
    name = st.session_state.get("rl_name", "").strip()
    if not name:
        return None
    filename = name.lower().replace(" ", "_") + ".yaml"
    (RUBRICS_DIR / filename).write_text(_build_rubric_yaml(), encoding="utf-8")
    return filename


# ── Chat helpers ──────────────────────────────────────────────────────────────

def _chat_config() -> tuple:
    return (
        st.session_state.get("chat_char"),
        st.session_state.get("chat_mode"),
        st.session_state.get("chat_rag", False),
        st.session_state.get("chat_provider", "anthropic"),
    )


def _init_chat_engine() -> str | None:
    """Create orchestrator(s) based on current chat config. Returns error string or None."""
    char_name, mode, use_rag, provider = _chat_config()
    if not char_name:
        return "No character selected."

    char_path = CHARS_DIR / f"{char_name}.yaml"
    if not char_path.exists():
        return f"Character file not found: {char_path}"

    try:
        from core.orchestrator import create_orchestrator
        from core.comparison import BaselineComparison

        rag = None
        if use_rag:
            from core.rag_manager import RAGManager
            rag = RAGManager(knowledge_dir=str(KNOWLEDGE_DIR))
            if rag.document_count == 0:
                rag.ingest_directory()

        if mode == "compare":
            orc_a = create_orchestrator(char_path, provider=provider, mode="a")
            orc_b = create_orchestrator(char_path, provider=provider, mode="b", rag=rag)
            st.session_state["chat_comp"] = BaselineComparison(orc_a, orc_b)
            st.session_state["chat_orc"]  = None
        else:
            st.session_state["chat_orc"]  = create_orchestrator(
                char_path, provider=provider, mode="b", rag=rag
            )
            st.session_state["chat_comp"] = None

        st.session_state["chat_active_config"] = _chat_config()
        return None

    except Exception as exc:
        return str(exc)


# ── Page 1: Character Creator ─────────────────────────────────────────────────

def render_creator() -> None:
    st.title("Character Creator")
    st.caption(
        "Build or edit a character YAML file. "
        "Load an existing character to edit it, or fill out the form to create a new one."
    )

    # ── Load / Save bar ───────────────────────────────────────────────────────
    chars = _list_chars()
    with st.container(border=True):
        bar_l, bar_m, bar_r = st.columns([4, 1, 2])
        with bar_l:
            load_opts = ["— new character —"] + chars
            load_sel  = st.selectbox("Load existing character", load_opts,
                                     key="load_char_sel", label_visibility="collapsed")
        with bar_m:
            if st.button("Load", use_container_width=True):
                if load_sel != "— new character —":
                    _load_character(CHARS_DIR / f"{load_sel}.yaml")
                    st.rerun()
        with bar_r:
            if st.button("💾  Save Character", type="primary", use_container_width=True):
                filename = _save_character()
                if filename:
                    st.success(f"Saved → characters/{filename}")
                else:
                    st.error("Name is required before saving.")

    st.divider()

    # ── Identity ──────────────────────────────────────────────────────────────
    st.header("Identity")
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.text_input("Name *", key="name", placeholder="e.g. Reva Kasai")
        st.text_input(
            "Lived-in Genre", key="lived_in_genre",
            placeholder="e.g. neo-noir, screwball comedy, kitchen-sink realism",
            help="The emotional/tonal register the character inhabits — not the story's genre.",
        )
    with col_b:
        st.text_area(
            "Fundamental Desire", key="fundamental_desire", height=96,
            placeholder="The character's core want (McKee-style). What do they want more than anything?",
        )

    st.text_area("Description", key="description", height=110,
                 placeholder="Physical appearance, age, role, immediate situation.")

    # ── Character ─────────────────────────────────────────────────────────────
    st.header("Character")
    st.text_area("Personality", key="personality", height=130,
                 placeholder="Temperament, emotional patterns, how they handle conflict and intimacy.")
    st.text_area("Background", key="background", height=130,
                 placeholder="Origin, history, current situation — what shaped them.")

    # ── Voice ─────────────────────────────────────────────────────────────────
    st.header("Voice")
    st.text_area("Speaking Style", key="speaking_style", height=110,
                 placeholder="Vocabulary register, sentence rhythm, idioms, verbal tics, tone shifts.")

    st.subheader("Voice Samples")
    st.caption(
        "Free-form dialogue exchanges used as style anchors in the system prompt. "
        "Write them how the character actually sounds — not instructions, just reference."
    )

    for vs_id in list(st.session_state.vs_ids):
        with st.container(border=True):
            top_col, del_col = st.columns([11, 1])
            with top_col:
                st.text_input(
                    "Context (optional)", key=f"vs_ctx_{vs_id}",
                    placeholder="One line: 'Reva deflects a question about Ganymede'",
                )
                st.text_area(
                    "Exchange", key=f"vs_exc_{vs_id}", height=130,
                    placeholder="User: Hey, do you miss home?\nReva: Not much to miss.",
                )
            with del_col:
                st.write(""); st.write("")
                st.button("✕", key=f"vs_del_{vs_id}",
                          on_click=_remove, args=("vs_ids", vs_id))

    st.button("+ Add Voice Sample", key="add_vs",
              on_click=_add, args=("vs_ids", "vs_next"))

    # ── Traits ────────────────────────────────────────────────────────────────
    st.header("Traits")
    col_goals, col_quirks = st.columns(2)

    with col_goals:
        st.subheader("Goals")
        st.caption("What the character is actively working toward.")
        for gid in list(st.session_state.goal_ids):
            g_col, gd_col = st.columns([9, 1])
            with g_col:
                st.text_input("Goal", key=f"goal_{gid}", label_visibility="collapsed",
                              placeholder="e.g. Pay off the Debt Collector")
            with gd_col:
                st.button("✕", key=f"goal_del_{gid}",
                          on_click=_remove, args=("goal_ids", gid))
        st.button("+ Add Goal", key="add_goal", on_click=_add, args=("goal_ids", "goal_next"))

    with col_quirks:
        st.subheader("Quirks")
        st.caption("Observable behavioural mannerisms and habits.")
        for qid in list(st.session_state.quirk_ids):
            q_col, qd_col = st.columns([9, 1])
            with q_col:
                st.text_input("Quirk", key=f"quirk_{qid}", label_visibility="collapsed",
                              placeholder="e.g. Drums fingers when thinking")
            with qd_col:
                st.button("✕", key=f"quirk_del_{qid}",
                          on_click=_remove, args=("quirk_ids", qid))
        st.button("+ Add Quirk", key="add_quirk",
                  on_click=_add, args=("quirk_ids", "quirk_next"))

    # ── World ─────────────────────────────────────────────────────────────────
    st.header("World")
    st.subheader("Relationships")
    st.caption("Named people or groups the character knows.")

    for rid in list(st.session_state.rel_ids):
        with st.container(border=True):
            r_top, r_del = st.columns([11, 1])
            with r_top:
                r_l, r_r = st.columns([2, 5])
                with r_l:
                    st.text_input("Name", key=f"rel_name_{rid}",
                                  placeholder="e.g. Marcus Chen")
                with r_r:
                    st.text_area("Description", key=f"rel_desc_{rid}", height=72,
                                 placeholder="Station mechanic, one of the few people Reva trusts.")
            with r_del:
                st.write(""); st.write("")
                st.button("✕", key=f"rel_del_{rid}",
                          on_click=_remove, args=("rel_ids", rid))

    st.button("+ Add Relationship", key="add_rel",
              on_click=_add, args=("rel_ids", "rel_next"))

    # ── Secrets ───────────────────────────────────────────────────────────────
    st.header("Secrets")
    st.caption("Information the character guards. Reveal conditions control when it surfaces.")

    _CONDITIONS = ["trust_threshold", "keyword_trigger", "emotional_state", "explicit_ask", "never"]

    for sec_id in list(st.session_state.sec_ids):
        with st.container(border=True):
            s_top, s_del = st.columns([11, 1])
            with s_top:
                st.text_area("Secret content", key=f"sec_content_{sec_id}", height=80,
                             placeholder="What the character is hiding.")
                s_l, s_m, s_r = st.columns([2, 2, 3])
                with s_l:
                    condition = st.selectbox("Reveal condition", _CONDITIONS,
                                            key=f"sec_condition_{sec_id}")
                with s_m:
                    if condition == "trust_threshold":
                        st.slider("Trust threshold", 0.0, 1.0, 0.75, 0.05,
                                  key=f"sec_threshold_{sec_id}")
                with s_r:
                    if condition in ("keyword_trigger", "emotional_state"):
                        ph = ("artemis, missing ship" if condition == "keyword_trigger"
                              else "vulnerable, grieving")
                        st.text_input("Triggers (comma-separated)",
                                      key=f"sec_triggers_{sec_id}", placeholder=ph)
            with s_del:
                st.write(""); st.write("")
                st.button("✕", key=f"sec_del_{sec_id}",
                          on_click=_remove, args=("sec_ids", sec_id))

    st.button("+ Add Secret", key="add_sec",
              on_click=_add, args=("sec_ids", "sec_next"))

    # ── Dramatic Register ─────────────────────────────────────────────────────
    st.header("Dramatic Register")
    st.caption("Controls how the engine shapes voice and response structure in Version B.")

    dr_l, dr_r = st.columns(2)
    with dr_l:
        st.slider("Subtextuality", 1, 10, key="subtextuality",
                  help="1 = blunt and plain-spoken · 10 = almost never says what they mean")
        st.caption("**1–2** Direct · **3–4** Mostly plain · **5–6** Moderate · "
                   "**7–8** High subtext · **9–10** Near-total indirection")
    with dr_r:
        st.slider("Digressiveness", 1, 10, key="digressiveness",
                  help="1 = terse and on-point · 10 = constantly spiraling into tangents")
        st.caption("**1–2** Clipped · **3–4** Mostly direct · **5–6** Occasional asides · "
                   "**7–8** Digressive · **9–10** Spiraling tangents")

    # ── Engine settings ───────────────────────────────────────────────────────
    with st.expander("Engine / Prompt Settings (optional)"):
        st.text_area("System Prefix", key="system_prefix", height=80,
                     placeholder="Prepended to every Version B prompt.")
        st.caption("Memory config is written with sensible defaults. "
                   "Edit the YAML manually to adjust memory limits.")

    # ── Preview / Download ────────────────────────────────────────────────────
    st.divider()
    prev_col, dl_col = st.columns([2, 3])
    with prev_col:
        show_preview = st.button("Preview YAML", use_container_width=True)
    if show_preview or st.session_state.get("yaml_preview"):
        if show_preview:
            st.session_state["yaml_preview"] = _build_char_yaml()
        yaml_str: str = st.session_state["yaml_preview"]
        raw_name = st.session_state.get("name", "character").strip() or "character"
        with dl_col:
            st.download_button(
                "⬇ Download YAML", data=yaml_str,
                file_name=raw_name.lower().replace(" ", "_") + ".yaml",
                mime="text/yaml", use_container_width=True,
            )
        st.code(yaml_str, language="yaml")


# ── Page 2: Knowledge Base ────────────────────────────────────────────────────

def render_knowledge() -> None:
    st.title("Knowledge Base")
    st.caption(
        "Documents in `knowledge/` are ingested into ChromaDB and retrieved "
        "per-turn when RAG is enabled in Chat."
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    st.subheader("Upload Documents")
    uploaded = st.file_uploader(
        "Add files to the knowledge base",
        type=["txt", "md", "yaml", "yml"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        saved = []
        for f in uploaded:
            dest = KNOWLEDGE_DIR / f.name
            dest.write_bytes(f.read())
            saved.append(f.name)
        st.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")
        st.rerun()

    # ── Current documents ─────────────────────────────────────────────────────
    st.subheader("Current Documents")
    files = sorted(KNOWLEDGE_DIR.iterdir())
    docs  = [f for f in files if f.is_file() and f.suffix.lower() in (".txt", ".md", ".yaml", ".yml")]

    if not docs:
        st.info("No documents yet. Upload files above.")
    else:
        for doc in docs:
            col_name, col_size, col_del = st.columns([6, 2, 1])
            with col_name:
                st.write(f"📄 **{doc.name}**")
            with col_size:
                size = doc.stat().st_size
                st.write(f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB")
            with col_del:
                if st.button("✕", key=f"del_doc_{doc.name}"):
                    doc.unlink()
                    st.rerun()

    # ── Index controls ────────────────────────────────────────────────────────
    st.subheader("Index")
    idx_col, stat_col = st.columns([2, 3])
    with idx_col:
        if st.button("🔄  Rebuild Index", type="primary", use_container_width=True):
            chroma_dir = Path(".chroma")
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
            try:
                from core.rag_manager import RAGManager
                rag = RAGManager(knowledge_dir=str(KNOWLEDGE_DIR))
                total = rag.ingest_directory()
                st.success(f"Index rebuilt — {rag.document_count} chunks from {len(docs)} file(s).")
            except Exception as exc:
                st.error(f"Rebuild failed: {exc}")

    with stat_col:
        chroma_dir = Path(".chroma")
        if chroma_dir.exists():
            try:
                from core.rag_manager import RAGManager
                rag = RAGManager(knowledge_dir=str(KNOWLEDGE_DIR))
                st.metric("Indexed chunks", rag.document_count)
            except Exception:
                st.write("Index exists (unable to read chunk count).")
        else:
            st.info("No index built yet.")


# ── Page 3: Rubric Builder ────────────────────────────────────────────────────

def render_rubric() -> None:
    st.title("Rubric Builder")
    st.caption(
        "Evaluation rubrics are YAML files in `rubrics/`. "
        "They are used by `run_eval.py` to score conversations with LLM-as-judge metrics."
    )

    # ── Load / Save bar ───────────────────────────────────────────────────────
    rubrics = _list_rubrics()
    with st.container(border=True):
        bar_l, bar_m, bar_r = st.columns([4, 1, 2])
        with bar_l:
            opts = ["— new rubric —"] + rubrics
            sel  = st.selectbox("Load existing", opts, key="rl_select",
                                label_visibility="collapsed")
        with bar_m:
            if st.button("Load", key="rl_load_btn", use_container_width=True):
                if sel != "— new rubric —":
                    _load_rubric(RUBRICS_DIR / f"{sel}.yaml")
                    st.rerun()
        with bar_r:
            if st.button("💾  Save Rubric", type="primary", key="rl_save_btn",
                         use_container_width=True):
                filename = _save_rubric()
                if filename:
                    st.success(f"Saved → rubrics/{filename}")
                else:
                    st.error("Name is required.")

    st.divider()

    # ── Delete existing ───────────────────────────────────────────────────────
    if rubrics:
        with st.expander(f"Existing rubrics ({len(rubrics)})"):
            for rname in rubrics:
                col_n, col_d = st.columns([8, 1])
                with col_n:
                    st.write(f"📋 {rname}")
                with col_d:
                    if st.button("✕", key=f"del_rub_{rname}"):
                        (RUBRICS_DIR / f"{rname}.yaml").unlink()
                        st.rerun()

    # ── Metadata ──────────────────────────────────────────────────────────────
    st.header("Metadata")
    m_l, m_m, m_r = st.columns([4, 2, 1])
    with m_l:
        st.text_input("Name *  (used as filename)", key="rl_name",
                      placeholder="e.g. voice_consistency")
    with m_m:
        st.text_input("Version", key="rl_version", placeholder="1.0")
    with m_r:
        st.number_input("Weight", min_value=0.0, max_value=10.0,
                        step=0.1, key="rl_weight")

    st.text_area("Description", key="rl_description", height=100,
                 placeholder="What this rubric measures and how it should be applied.")

    # ── Criteria ──────────────────────────────────────────────────────────────
    st.header("Criteria")
    st.caption("Each criterion becomes a separate LLM-as-judge metric in DeepEval.")

    for cid in list(st.session_state.rl_crit_ids):
        with st.container(border=True):
            top_col, del_col = st.columns([11, 1])
            with top_col:
                c_l, c_r = st.columns([2, 4])
                with c_l:
                    st.text_input("ID", key=f"rl_crit_id_{cid}",
                                  placeholder="e.g. vocabulary")
                with c_r:
                    st.text_input("Label", key=f"rl_crit_label_{cid}",
                                  placeholder="e.g. Vocabulary & Idiom Consistency")
                st.text_area("Scoring prompt", key=f"rl_crit_prompt_{cid}", height=90,
                             placeholder="Does the character use vocabulary consistent with their speaking style?")
            with del_col:
                st.write(""); st.write("")
                st.button("✕", key=f"rl_crit_del_{cid}",
                          on_click=_remove, args=("rl_crit_ids", cid))

    st.button("+ Add Criterion", key="rl_add_crit",
              on_click=_add, args=("rl_crit_ids", "rl_crit_next"))

    # ── Scale anchors ─────────────────────────────────────────────────────────
    st.header("Scale Anchors  (1 → 5)")
    scale_cols = st.columns(5)
    _placeholders = [
        "No recognisable voice",
        "Occasional flashes",
        "Broadly recognisable, notable lapses",
        "Strong voice, minor deviations",
        "Immaculate consistency",
    ]
    for i, col in enumerate(scale_cols, 1):
        with col:
            st.text_input(f"**{i}**", key=f"rl_scale_{i}", label_visibility="visible",
                          placeholder=_placeholders[i - 1])

    # ── Notes ─────────────────────────────────────────────────────────────────
    st.text_area("Notes", key="rl_notes", height=80,
                 placeholder="Guidance for evaluators, known limitations, version history.")

    # ── Preview ───────────────────────────────────────────────────────────────
    st.divider()
    prev_col, dl_col = st.columns([2, 3])
    with prev_col:
        if st.button("Preview YAML", key="rl_preview_btn", use_container_width=True):
            st.session_state["rl_yaml_preview"] = _build_rubric_yaml()
    if st.session_state.get("rl_yaml_preview"):
        yaml_str = st.session_state["rl_yaml_preview"]
        rname = st.session_state.get("rl_name", "rubric").strip() or "rubric"
        with dl_col:
            st.download_button(
                "⬇ Download YAML", data=yaml_str,
                file_name=rname.lower().replace(" ", "_") + ".yaml",
                mime="text/yaml", use_container_width=True, key="rl_dl_btn",
            )
        st.code(yaml_str, language="yaml")


# ── Page 4: Transcript Ingester ──────────────────────────────────────────────

def render_ingester() -> None:
    import os
    import tempfile

    st.title("Transcript Ingester")
    st.caption(
        "Paste or upload a plain-text transcript and the engine will reverse-engineer "
        "a character YAML from it — name, personality, background, speaking style, "
        "secrets, voice samples, and more."
    )
    st.warning(
        "Fields marked `# inferred - verify` are the engine's best guess from the "
        "transcript. Review them in Character Creator before using in a demo."
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    input_mode = st.radio(
        "Input method", ["Paste text", "Upload .txt file"],
        horizontal=True, key="ing_input_mode",
    )

    raw_text: str | None = None

    if input_mode == "Paste text":
        pasted = st.text_area(
            "Transcript",
            key="ing_paste",
            height=300,
            placeholder=(
                "User: Hey, what's your name?\n"
                "Aria: Aria. And you are?\n"
                "User: Just passing through.\n"
                "Aria: Nobody just passes through here."
            ),
        )
        raw_text = pasted or None
    else:
        uploaded = st.file_uploader(
            "Upload transcript (.txt)", type=["txt"], key="ing_upload",
        )
        if uploaded:
            raw_text = uploaded.read().decode("utf-8")

    # ── Options ───────────────────────────────────────────────────────────────
    opt_l, opt_r = st.columns(2)
    with opt_l:
        ing_notes = st.text_input(
            "Notes (optional)", key="ing_notes",
            placeholder="e.g. Session 1, client demo 2024-01-15",
            help="Added as a provenance comment at the top of the generated YAML.",
        )
    with opt_r:
        ing_char_label = st.text_input(
            "Character label override (optional)", key="ing_char_label",
            placeholder="e.g. Aria",
            help=(
                "The speaker label for the character in the transcript. "
                "Only needed if auto-detection picks the wrong speaker."
            ),
        )

    # ── Generate ──────────────────────────────────────────────────────────────
    if st.button("Generate Character YAML", type="primary", key="ing_generate"):
        if not raw_text or not raw_text.strip():
            st.error("Paste a transcript or upload a file first.")
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        else:
            tmp_path: Path | None = None
            try:
                from ingest_transcript import (
                    parse_transcript,
                    format_transcript,
                    extract_character,
                    render_yaml,
                )

                # parse_transcript expects a file path — write to a temp file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(raw_text)
                    tmp_path = Path(f.name)

                user_label, char_label, turns = parse_transcript(
                    tmp_path, ing_char_label.strip() or None
                )
                st.info(
                    f"Detected **{len(turns)} turns** — "
                    f"user: `{user_label}` · character: `{char_label}`"
                )

                transcript_text = format_transcript(turns, user_label, char_label)

                with st.spinner(f"Sending to Claude…"):
                    try:
                        data = extract_character(
                            transcript_text, char_label, "claude-opus-4-6"
                        )
                    except SystemExit as exc:
                        st.error(f"Extraction failed: {exc}")
                        st.stop()

                yaml_output = render_yaml(
                    data, ing_notes.strip() or None, "pasted_transcript"
                )
                st.session_state["ingester_yaml"] = yaml_output
                st.session_state["ingester_char_name"] = (
                    data.get("character_name", {}).get("value", "character")
                )

            except ValueError as exc:
                st.error(f"Parse error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

    # ── Results ───────────────────────────────────────────────────────────────
    yaml_str: str | None = st.session_state.get("ingester_yaml")
    if yaml_str:
        char_name = st.session_state.get("ingester_char_name", "character")
        filename  = char_name.lower().replace(" ", "_") + ".yaml"

        st.divider()
        dl_col, save_col = st.columns(2)

        with dl_col:
            st.download_button(
                "⬇ Download YAML",
                data=yaml_str,
                file_name=filename,
                mime="text/yaml",
                use_container_width=True,
                key="ing_download",
            )
        with save_col:
            if st.button(
                "💾 Save to Characters Folder",
                use_container_width=True, key="ing_save",
            ):
                (CHARS_DIR / filename).write_text(yaml_str, encoding="utf-8")
                st.success(
                    f"Saved to `characters/{filename}` — "
                    "now available in Character Creator and Chat."
                )

        st.code(yaml_str, language="yaml")


# ── Page 5: Chat ──────────────────────────────────────────────────────────────

def render_chat() -> None:
    st.title("Chat")

    chars = _list_chars()
    if not chars:
        st.warning("No characters found. Create one in the Character Creator first.")
        return

    # ── Config bar ────────────────────────────────────────────────────────────
    with st.container(border=True):
        cfg_a, cfg_b, cfg_c, cfg_d = st.columns([3, 3, 1, 1])
        with cfg_a:
            st.selectbox("Character", chars, key="chat_char")
        with cfg_b:
            st.radio("Mode", ["Version B (dynamic)", "A vs B (compare)"],
                     key="chat_mode", horizontal=True)
        with cfg_c:
            st.toggle("RAG", key="chat_rag")
        with cfg_d:
            st.selectbox("Provider", ["anthropic", "openai"], key="chat_provider",
                         label_visibility="collapsed")

    # Detect config changes → reset session
    new_config = _chat_config()
    if st.session_state.get("chat_active_config") != new_config:
        st.session_state["chat_messages"]      = []
        st.session_state["chat_orc"]           = None
        st.session_state["chat_comp"]          = None
        st.session_state["chat_active_config"] = None   # force re-init on next send

    # ── Action bar ────────────────────────────────────────────────────────────
    act_a, act_b, act_c = st.columns([2, 2, 2])
    with act_a:
        show_state = st.button("📊  Show State", use_container_width=True)
    with act_b:
        if st.button("🔄  Reset Conversation", use_container_width=True):
            if st.session_state.get("chat_orc"):
                st.session_state["chat_orc"].reset()
            if st.session_state.get("chat_comp"):
                st.session_state["chat_comp"].reset()
            st.session_state["chat_messages"] = []
            st.rerun()
    with act_c:
        if st.session_state.get("chat_messages"):
            _export_conversation_log()

    # ── State panel ───────────────────────────────────────────────────────────
    if show_state:
        orc = st.session_state.get("chat_orc")
        comp = st.session_state.get("chat_comp")
        active_orc = orc or (comp.orchestrator_b if comp else None)
        if active_orc:
            summary = active_orc.get_state_summary()
            with st.expander("Conversation State", expanded=True):
                s_l, s_m, s_r = st.columns(3)
                with s_l:
                    st.metric("Turn", summary.get("turns", 0))
                    st.metric("Secrets revealed", summary.get("secrets_revealed", 0))
                with s_m:
                    mood = summary.get("emotional_state", {})
                    st.metric("Primary emotion", mood.get("primary", "—"))
                    st.metric("Intensity", f"{mood.get('intensity', 0):.2f}")
                with s_r:
                    rel = summary.get("relationship", {})
                    st.metric("Trust",      f"{rel.get('trust_level', 0):.2f}")
                    st.metric("Rapport",    f"{rel.get('rapport', 0):.2f}")
                topics = summary.get("active_topics", [])
                if topics:
                    st.write("**Active topics:**", ", ".join(topics))
        else:
            st.info("Start a conversation to see state.")

    # ── Message history ───────────────────────────────────────────────────────
    is_compare = st.session_state.get("chat_mode") == "A vs B (compare)"

    if is_compare and st.session_state["chat_messages"]:
        # Column headers — only shown once
        _, col_a_hdr, col_b_hdr = st.columns([2, 5, 5])
        col_a_hdr.markdown("**Version A — Static**")
        col_b_hdr.markdown("**Version B — Dynamic**")

    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg.get("is_compare"):
            with st.chat_message("assistant"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Version A**")
                    st.write(msg["content_a"])
                with col_b:
                    st.markdown("**Version B**")
                    st.write(msg["content_b"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Say something…"):
        # Lazy init
        if st.session_state.get("chat_active_config") != new_config:
            with st.spinner("Initialising character…"):
                err = _init_chat_engine()
            if err:
                st.error(f"Could not initialise: {err}")
                return

        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        orc  = st.session_state.get("chat_orc")
        comp = st.session_state.get("chat_comp")

        if comp:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    result = comp.chat(prompt)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Version A**")
                    st.write(result.response_a)
                with col_b:
                    st.markdown("**Version B**")
                    st.write(result.response_b)
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "is_compare": True,
                "content_a": result.response_a,
                "content_b": result.response_b,
            })
        elif orc:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    reply = orc.chat(prompt)
                st.write(reply)
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "is_compare": False,
                "content": reply,
            })
        else:
            st.error("Engine not initialised — try resending.")


def _export_conversation_log() -> None:
    """Render a download button for the conversation log."""
    turns = []
    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "user":
            turns.append({"role": "user", "content": msg["content"]})
        elif msg.get("is_compare"):
            turns.append({"role": "assistant_a", "content": msg["content_a"]})
            turns.append({"role": "assistant_b", "content": msg["content_b"]})
        else:
            turns.append({"role": "assistant", "content": msg["content"]})

    log = {
        "character": st.session_state.get("chat_char", "unknown"),
        "mode":      "compare" if st.session_state.get("chat_mode") == "A vs B (compare)" else "b",
        "timestamp": datetime.now().isoformat(),
        "turns":     turns,
    }
    log_yaml = yaml.dump(log, allow_unicode=True, sort_keys=False, default_flow_style=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "⬇  Save Log",
        data=log_yaml,
        file_name=f"conversation_{ts}.yaml",
        mime="text/yaml",
        use_container_width=True,
        key="chat_dl_log",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Persona Engine",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_state()

with st.sidebar:
    st.title("🎭 Persona Engine")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Character Creator", "Knowledge Base", "Rubric Builder", "Transcript Ingester", "Chat"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Persona Engine · [GitHub](https://github.com/msonnenschein-art/persona-engine)")

if page == "Character Creator":
    render_creator()
elif page == "Knowledge Base":
    render_knowledge()
elif page == "Rubric Builder":
    render_rubric()
elif page == "Transcript Ingester":
    render_ingester()
elif page == "Chat":
    render_chat()
