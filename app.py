"""
app.py — Persona Engine Character Creator

A Streamlit form for building character YAML files compatible with the
Persona Engine schema. Generates a downloadable YAML file ready to drop
into the characters/ directory.

Run with:
    streamlit run app.py
"""

import yaml
import streamlit as st


# ── YAML generation helpers ───────────────────────────────────────────────────

class _Literal(str):
    """Marker class so PyYAML emits this string as a literal block scalar (|)."""


def _literal_representer(dumper: yaml.Dumper, data: "_Literal") -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


class _CharacterDumper(yaml.Dumper):
    pass


_CharacterDumper.add_representer(_Literal, _literal_representer)


def _block(text: str) -> "_Literal | str":
    """Wrap multiline text as a literal block scalar; leave single lines plain."""
    text = text.strip()
    if not text:
        return text
    # Use block style if the text spans multiple lines or is long prose
    if "\n" in text or len(text) > 72:
        return _Literal(text + "\n")
    return text


def _dump(data: dict) -> str:
    return yaml.dump(
        data,
        Dumper=_CharacterDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=88,
    )


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        # Voice samples
        "vs_ids": [0],
        "vs_next": 1,
        # Goals
        "goal_ids": [0],
        "goal_next": 1,
        # Quirks
        "quirk_ids": [0],
        "quirk_next": 1,
        # Relationships
        "rel_ids": [0],
        "rel_next": 1,
        # Secrets
        "sec_ids": [0],
        "sec_next": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _add(id_key: str, next_key: str) -> None:
    st.session_state[id_key].append(st.session_state[next_key])
    st.session_state[next_key] += 1


def _remove(id_key: str, item_id: int) -> None:
    st.session_state[id_key] = [i for i in st.session_state[id_key] if i != item_id]


# ── YAML builder ──────────────────────────────────────────────────────────────

def _build_yaml() -> str:  # noqa: C901
    ss = st.session_state
    data: dict = {}

    # Identity
    name = ss.get("name", "").strip()
    data["name"] = name or "Unnamed Character"

    if desc := ss.get("description", "").strip():
        data["description"] = _block(desc)
    if pers := ss.get("personality", "").strip():
        data["personality"] = _block(pers)
    if bg := ss.get("background", "").strip():
        data["background"] = _block(bg)
    if style := ss.get("speaking_style", "").strip():
        data["speaking_style"] = _block(style)

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

    # Traits
    goals = [ss.get(f"goal_{gid}", "").strip() for gid in ss.get("goal_ids", [])]
    goals = [g for g in goals if g]
    if goals:
        data["goals"] = goals

    quirks = [ss.get(f"quirk_{qid}", "").strip() for qid in ss.get("quirk_ids", [])]
    quirks = [q for q in quirks if q]
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
    secrets = []
    for sid in ss.get("sec_ids", []):
        content = ss.get(f"sec_content_{sid}", "").strip()
        if not content:
            continue
        condition = ss.get(f"sec_condition_{sid}", "trust_threshold")
        entry = {"content": _block(content), "reveal_condition": condition}
        if condition == "trust_threshold":
            entry["threshold"] = round(ss.get(f"sec_threshold_{sid}", 0.75), 2)
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
        "short_term_limit": 10,
        "long_term_limit": 50,
        "episodic_limit": 20,
        "consolidation_threshold": 5,
        "relevance_decay": 0.95,
    }

    if sp := ss.get("system_prefix", "").strip():
        data["system_prefix"] = _block(sp)

    return _dump(data)


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Persona Engine — Character Creator",
    page_icon="🎭",
    layout="wide",
)

_init_state()

st.title("Character Creator")
st.caption(
    "Build a character YAML file ready to drop into the `characters/` directory. "
    "Only **Name** is required — fill in as much or as little as you have."
)

# ── Identity ──────────────────────────────────────────────────────────────────

st.header("Identity")

col_a, col_b = st.columns([2, 3])
with col_a:
    st.text_input("Name *", key="name", placeholder="e.g. Reva Kasai")
    st.text_input(
        "Lived-in Genre",
        key="lived_in_genre",
        placeholder="e.g. neo-noir, screwball comedy, kitchen-sink realism",
        help="The emotional/tonal register the character inhabits — their internal world, not the story's genre.",
    )

with col_b:
    st.text_area(
        "Fundamental Desire",
        key="fundamental_desire",
        height=96,
        placeholder="The character's core want or drive (McKee-style). What do they want more than anything?",
    )

st.text_area(
    "Description",
    key="description",
    height=110,
    placeholder="Physical appearance, age, role, immediate situation.",
)

# ── Character ─────────────────────────────────────────────────────────────────

st.header("Character")

st.text_area("Personality", key="personality", height=130,
             placeholder="Temperament, emotional patterns, how they handle conflict and intimacy.")
st.text_area("Background", key="background", height=130,
             placeholder="Origin, history, current situation — what shaped them.")

# ── Voice ─────────────────────────────────────────────────────────────────────

st.header("Voice")

st.text_area(
    "Speaking Style",
    key="speaking_style",
    height=110,
    placeholder="Vocabulary register, sentence rhythm, idioms, verbal tics, tone shifts.",
)

# ── Voice Samples ─────────────────────────────────────────────────────────────

st.subheader("Voice Samples")
st.caption(
    "Free-form dialogue exchanges used as style anchors in the system prompt. "
    "Write them how the character actually sounds — these are not instructions, just reference. "
    "Context is optional; Exchange should be multi-line dialogue."
)

for vs_id in list(st.session_state.vs_ids):
    with st.container(border=True):
        top_col, del_col = st.columns([11, 1])
        with top_col:
            st.text_input(
                "Context (optional)",
                key=f"vs_ctx_{vs_id}",
                placeholder="One line describing the situation, e.g. 'Reva deflects a question about Ganymede'",
            )
            st.text_area(
                "Exchange",
                key=f"vs_exc_{vs_id}",
                height=130,
                placeholder="User: Hey, do you miss home?\nReva: Not much to miss.",
            )
        with del_col:
            st.write("")  # vertical alignment nudge
            st.write("")
            st.button(
                "✕",
                key=f"vs_del_{vs_id}",
                on_click=_remove,
                args=("vs_ids", vs_id),
                help="Remove this sample",
            )

st.button(
    "+ Add Voice Sample",
    key="add_vs",
    on_click=_add,
    args=("vs_ids", "vs_next"),
)

# ── Traits ────────────────────────────────────────────────────────────────────

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
            st.button("✕", key=f"goal_del_{gid}", on_click=_remove,
                      args=("goal_ids", gid))
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
            st.button("✕", key=f"quirk_del_{qid}", on_click=_remove,
                      args=("quirk_ids", qid))
    st.button("+ Add Quirk", key="add_quirk", on_click=_add, args=("quirk_ids", "quirk_next"))

# ── World ─────────────────────────────────────────────────────────────────────

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
            st.write("")
            st.write("")
            st.button("✕", key=f"rel_del_{rid}", on_click=_remove,
                      args=("rel_ids", rid), help="Remove")

st.button("+ Add Relationship", key="add_rel", on_click=_add, args=("rel_ids", "rel_next"))

# ── Secrets ───────────────────────────────────────────────────────────────────

st.header("Secrets")
st.caption(
    "Information the character guards. Reveal conditions control when it surfaces "
    "during conversation."
)

_CONDITIONS = [
    "trust_threshold",
    "keyword_trigger",
    "emotional_state",
    "explicit_ask",
    "never",
]

for sec_id in list(st.session_state.sec_ids):
    with st.container(border=True):
        s_top, s_del = st.columns([11, 1])
        with s_top:
            st.text_area(
                "Secret content",
                key=f"sec_content_{sec_id}",
                height=80,
                placeholder="What the character is hiding.",
            )
            s_l, s_m, s_r = st.columns([2, 2, 3])
            with s_l:
                condition = st.selectbox(
                    "Reveal condition",
                    _CONDITIONS,
                    key=f"sec_condition_{sec_id}",
                )
            with s_m:
                if condition == "trust_threshold":
                    st.slider(
                        "Trust threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.75,
                        step=0.05,
                        key=f"sec_threshold_{sec_id}",
                    )
            with s_r:
                if condition in ("keyword_trigger", "emotional_state"):
                    st.text_input(
                        "Triggers (comma-separated)",
                        key=f"sec_triggers_{sec_id}",
                        placeholder="artemis, missing ship" if condition == "keyword_trigger" else "vulnerable, grieving",
                    )
        with s_del:
            st.write("")
            st.write("")
            st.button("✕", key=f"sec_del_{sec_id}", on_click=_remove,
                      args=("sec_ids", sec_id), help="Remove")

st.button("+ Add Secret", key="add_sec", on_click=_add, args=("sec_ids", "sec_next"))

# ── Dramatic Register ─────────────────────────────────────────────────────────

st.header("Dramatic Register")
st.caption("Controls how the engine shapes the character's voice and response structure.")

dr_l, dr_r = st.columns(2)

with dr_l:
    st.slider(
        "Subtextuality",
        min_value=1,
        max_value=10,
        value=5,
        key="subtextuality",
        help="1 = blunt and plain-spoken · 10 = almost never says what they mean",
    )
    st.caption(
        "**1–2** Direct, no subtext &nbsp;·&nbsp; "
        "**3–4** Mostly plain &nbsp;·&nbsp; "
        "**5–6** Moderate implication &nbsp;·&nbsp; "
        "**7–8** High subtext &nbsp;·&nbsp; "
        "**9–10** Near-total indirection"
    )

with dr_r:
    st.slider(
        "Digressiveness",
        min_value=1,
        max_value=10,
        value=5,
        key="digressiveness",
        help="1 = terse and on-point · 10 = constantly spiraling into tangents",
    )
    st.caption(
        "**1–2** Clipped, stays on topic &nbsp;·&nbsp; "
        "**3–4** Mostly direct &nbsp;·&nbsp; "
        "**5–6** Occasional asides &nbsp;·&nbsp; "
        "**7–8** Digressive &nbsp;·&nbsp; "
        "**9–10** Spiraling tangents"
    )

# ── Engine ────────────────────────────────────────────────────────────────────

with st.expander("Engine / Prompt Settings (optional)"):
    st.text_area(
        "System Prefix",
        key="system_prefix",
        height=80,
        placeholder="Prepended to every Version B prompt. e.g. 'You are roleplaying as X. Stay in character at all times.'",
    )
    st.caption(
        "Memory config is written with sensible defaults. "
        "Edit the YAML manually to adjust `short_term_limit`, `long_term_limit`, etc."
    )

# ── Generate ──────────────────────────────────────────────────────────────────

st.divider()

gen_col, dl_col = st.columns([2, 3])

with gen_col:
    generate = st.button("Generate YAML", type="primary", use_container_width=True)

if generate or st.session_state.get("yaml_output"):
    if generate:
        st.session_state["yaml_output"] = _build_yaml()

    yaml_str: str = st.session_state["yaml_output"]

    with dl_col:
        raw_name = st.session_state.get("name", "character").strip() or "character"
        filename = raw_name.lower().replace(" ", "_") + ".yaml"
        st.download_button(
            "⬇ Download YAML",
            data=yaml_str,
            file_name=filename,
            mime="text/yaml",
            use_container_width=True,
        )

    st.code(yaml_str, language="yaml")
