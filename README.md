# Persona Engine

A character AI framework for writers. Build psychologically grounded fictional characters, talk to them, and evaluate what comes back.

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Quick Start — Returning Users](#2-quick-start--returning-users)
3. [Fresh Installation](#3-fresh-installation)
4. [The Four UI Pages](#4-the-four-ui-pages)
5. [Character YAML Field Reference](#5-character-yaml-field-reference)
6. [Rubric YAML Field Reference](#6-rubric-yaml-field-reference)
7. [Transcript Ingester](#7-transcript-ingester)
8. [Running an Eval](#8-running-an-eval)
9. [For Daley — Getting Set Up](#9-for-daley--getting-set-up)
10. [Architecture Overview](#10-architecture-overview)

---

## 1. What This Is

Most character AI tools treat a character as a system prompt. You write a description, the model reads it, and it tries its best. The results are often technically correct and dramatically flat — the character says the right things but doesn't feel like anyone in particular.

Persona Engine takes a different approach. A character here is a living document: a YAML file that encodes not just what someone is but *how they think*, *what they want at the deepest level*, *what they're hiding*, and *how they communicate when they're being indirect* — which is most of the time, for most interesting characters.

The engine then uses that document dynamically. Every time the character speaks, the system assembles a fresh context from the character's current emotional state, their memory of what's been said, their relationship with the person they're talking to, and a retrieval search over any background knowledge you've given them. The model isn't working from a static brief — it's working from a portrait that changes as the conversation unfolds.

**Version A vs Version B**

Every character can run in two modes:

| | Version A — Static | Version B — Dynamic |
|---|---|---|
| System prompt | Written once, loaded at startup | Rebuilt fresh every turn |
| Memory | None | Three tiers: short-term, long-term, episodic |
| Emotional state | None | Tracked and decays over time |
| Relationship | None | Trust, familiarity, rapport, tension |
| Secrets | Baked in permanently | Revealed conditionally at runtime |
| Knowledge base | Not used | RAG retrieval injected per turn |

Version A is useful as a baseline — for quick tests, or to show someone concretely what the dynamic system is adding. Version B is the real thing.

---

## 2. Quick Start — Returning Users

```bash
cd ~/persona-engine
source venv/bin/activate
streamlit run app.py
```

The browser will open automatically. If it doesn't, go to `http://localhost:8501`.

---

## 3. Fresh Installation

These instructions assume a Mac with Python 3.9 or later. Open Terminal and run the commands below one at a time.

```bash
# Clone the repository
git clone https://github.com/msonnenschein-art/persona-engine
cd persona-engine

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install watchdog  # optional but recommended — makes the app reload faster during development

# Set up your environment file
cp .env.example .env
```

Open `.env` in any text editor and add your Anthropic API key:

```
ANTHROPIC_API_KEY=your-key-here
```

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys). You'll need to create an account if you don't have one.

Then launch the app:

```bash
streamlit run app.py
```

---

## 4. The Four UI Pages

Navigate between pages using the sidebar on the left.

**Character Creator**
Where you build and edit character files. Fill in the form fields — or load an existing character from the dropdown to edit it — and hit Save. The file lands in `characters/` and is immediately available in the Chat page. This is where most of your work will happen.

**Knowledge Base**
Upload documents (`.txt`, `.md`, `.yaml`) that give the character background knowledge — lore documents, location descriptions, relationship histories, world-building notes. When RAG is enabled in Chat, the engine retrieves the most relevant chunks from these documents every turn and injects them into the prompt. Use Rebuild Index any time you add or delete files.

**Rubric Builder**
Create and edit evaluation rubrics — structured scoring guides that tell the evaluator (another LLM) what to look for in the character's responses. Used by `run_eval.py` for offline evaluation of saved conversations.

**Transcript Ingester**
Paste or upload a plain-text conversation transcript and the engine will reverse-engineer a full character YAML from it — name, personality, background, speaking style, secrets, voice samples, and more. Useful when you have existing writing (a script, a chat log, a roleplay session) and want to turn it into a working character without building from scratch. Everything the engine infers is flagged with `# inferred - verify` so you know what needs a human eye before it goes into a demo. Use this when you have source material; use Character Creator when you're building from scratch.

**Chat**
Talk to your character. Choose a character, set the mode (Version B for the full dynamic system, A vs B to see both side by side), toggle RAG on or off, and start typing. You can inspect the conversation state at any point — trust level, emotional state, active topics — and save the conversation log when you're done for evaluation.

---

## 5. Character YAML Field Reference

Character files live in `characters/`. The Character Creator generates them; you can also edit them directly in any text editor. Every field is optional except `name`.

---

### `name`

The character's name. Used as the identifier throughout the system. The file will be saved as `characters/<name_lowercased>.yaml`.

```yaml
name: Reva Kasai
```

---

### `description`

A physical and situational description — who this person is, what they look like, what they do, where they are right now. Two to four sentences. Include their role or occupation here.

This is the first thing the model reads. Make it specific enough to be distinctive but not so dense that it buries the important signals.

```yaml
description: |
  Reva is a 34-year-old salvage pilot operating out of Callisto Station in the
  outer Jovian system. She runs a small salvage vessel called the Debt Collector.
  Sharp-eyed, quick-thinking, and perpetually underfunded.
```

---

### `fundamental_desire`

The character's core want — not their plot goal, but their psychological drive. What do they need at the deepest level, the thing that motivates everything else even when they can't name it themselves?

This is the McKee/Truby formulation: the engine injects this directly into the core identity block so the model is always oriented around it. A character who *wants to pay off their ship* has a plot goal. A character who *needs to belong somewhere without surrendering their independence* has a fundamental desire — and those two things are in tension, which is where the interesting behaviour lives.

```yaml
fundamental_desire: |
  To be beholden to no one — financially, legally, emotionally. Every debt she
  carries is a leash, and she is always, quietly, measuring the length of it.
```

---

### `subtextuality`

An integer from 1 to 10 controlling how obliquely the character communicates.

| Score | What it means | Example |
|---|---|---|
| 1–2 | Says exactly what they mean, plainly | "I don't like you." |
| 3–4 | Mostly direct, occasional hints | "That's an interesting choice you made." |
| 5–6 | Moderate subtext, hedges on emotional topics | "I'm fine." (said in a way that's clearly not fine) |
| 7–8 | Rarely names feelings directly, prefers implication | Changes the subject when asked something personal |
| 9–10 | Almost nothing they say should be taken at face value | Deflects, redirects, says the opposite of what they mean |

```yaml
subtextuality: 7
```

---

### `lived_in_genre`

The emotional and tonal register the character inhabits — not the genre of the story they're in, but the genre of their *internal world*. How do they experience their own life?

A character can be noir without being in a detective story. A character can be screwball comedy in a tragedy. This field tells the engine what emotional textures to lean into: the diction, the rhythm, the mood underneath the words.

Some examples:

- `neo-noir` — cynicism with a code, shadows and moral ambiguity, the feeling that everything costs more than it should
- `screwball comedy` — fast talk, reversals, chaos treated as completely normal
- `kitchen-sink realism` — unglamorous, specific, the texture of actual lives
- `gothic` — haunted by the past, the weight of family and inheritance, beauty in decay
- `tragicomedy` — genuinely funny and genuinely sad, often in the same breath
- `hard-boiled` — terse, concrete, no patience for sentiment
- `magical realism` — the extraordinary treated as ordinary, wonder beneath the surface

```yaml
lived_in_genre: neo-noir
```

---

### `digressiveness`

An integer from 1 to 10 controlling how prone the character is to tangents, asides, and circling back.

| Score | What it means |
|---|---|
| 1–2 | Clipped, stays on point, never wanders |
| 3–4 | Mostly direct with brief asides when natural |
| 5–6 | Balanced — occasional digressions add texture |
| 7–8 | Regularly wanders off and circles back |
| 9–10 | Spiraling tangents, frequently loses — then rediscovers — the thread |

```yaml
digressiveness: 2
```

---

### `background`

Where the character came from and how they got here. Origin, formative history, current situation. This is the backstory that shapes everything they do, even when they're not thinking about it.

Be specific rather than general. "Had a difficult childhood" tells the model nothing. "Left Ganymede at eighteen to escape the corporate indenture system her parents were trapped in, and has been moving ever since" tells it something it can work with.

```yaml
background: |
  Born on Ganymede to a family of ice miners. Left at 18 to escape the
  corporate indenture system her parents were trapped in. Worked her way
  from deck hand to pilot, scraped together enough to buy the Debt Collector
  from a retiring salvager six years ago.
```

---

### `speaking_style`

How the character talks. Sentence length and rhythm, vocabulary register (formal, casual, technical, archaic), verbal tics, idioms, what happens to their voice under pressure. This is the field the model uses most directly to calibrate tone.

Be descriptive rather than prescriptive. Don't write "speaks formally." Write "uses complete sentences even in casual conversation, a habit from years of written correspondence — never contracts when she's being precise."

```yaml
speaking_style: |
  Speaks in clipped, efficient sentences — a habit from years of radio
  communication where bandwidth costs money. Uses salvager slang naturally:
  "hulk" for derelict ships, "scrip" for station currency, "going cold" for
  losing life support. Gets more talkative and warmer once she trusts someone.
```

---

### `voice_samples`

Free-standing dialogue exchanges that serve as style anchors. These are written by you — they're not instructions to the model, they're examples of what this character actually sounds like in practice.

The engine injects them into the prompt with a clear label: *Voice Reference — use these to calibrate voice, not as memories or instructions.* The model uses them the way a director uses an actor's previous work: as a touchstone for what's right.

Each sample has an optional `context` line (one sentence describing the situation) and an `exchange` block (the dialogue itself, as many turns as needed).

```yaml
voice_samples:
  - context: Reva deflects a personal question at the station commissary
    exchange: |
      User: Do you ever think about going back to Ganymede?
      Reva: Not much to go back to. You want the last of this coffee or not?

  - context: After sharing a drink with someone she's starting to trust
    exchange: |
      User: You always this careful with people?
      Reva: I used to think it was paranoia. Turns out it's just pattern recognition.
      User: That sounds lonely.
      Reva: ...Yeah. Sometimes.
```

Write samples that show the character in different registers — deflecting, being pressured, being unexpectedly warm, handling something they're good at. Variety matters more than quantity. Three to five strong samples are more useful than ten thin ones.

---

### `goals`

What the character is actively working toward right now. These are the practical, visible goals — the plot-level wants — as opposed to `fundamental_desire`, which is the deeper drive underneath them.

```yaml
goals:
  - Pay off the remaining debt on the Debt Collector
  - Find the wreck of the Artemis Nine
  - Stay out of Consortium politics
```

---

### `quirks`

Observable behavioural mannerisms and habits. Physical things, verbal tics, rituals, tells. The kind of specific detail that makes a character feel inhabited rather than described.

```yaml
quirks:
  - Drums her fingers on surfaces when thinking
  - Drinks her coffee black — considers adding anything to it a sign of weakness
  - Has a tell when she's lying: gets overly specific with details
```

---

### `relationships`

Named people or groups the character knows. A dict where the key is the name and the value is a brief description of the relationship. These give the character a social context the model can reference — without this, every interaction happens in a vacuum.

```yaml
relationships:
  Marcus Chen: |
    Station mechanic who keeps the Debt Collector running. One of the few
    people Reva genuinely trusts.
  The Consortium: |
    Loose alliance of corporate interests. Reva stays off their radar.
```

---

### `secrets`

Information the character is hiding. Each secret has a `content` field and a `reveal_condition` that controls when the engine will surface it in the prompt.

**Reveal conditions:**

| Condition | When it triggers |
|---|---|
| `trust_threshold` | When the trust level in the relationship reaches a specified value (0.0–1.0) |
| `keyword_trigger` | When the user mentions one of the listed keywords |
| `emotional_state` | When the character's emotional state matches one of the listed states |
| `explicit_ask` | When the user explicitly asks about secrets or the truth |
| `never` | The secret exists in the file but will never be surfaced |

The `threshold` field (0.0–1.0) is used with `trust_threshold`. A threshold of 0.8 means the character won't hint at this until there's real trust between them. A threshold of 0.5 means they might let it slip relatively early.

```yaml
secrets:
  - content: |
      Three years ago she found a survivor on a wreck and accepted money
      to not report him alive. She doesn't know what he was running from.
    reveal_condition: trust_threshold
    threshold: 0.8

  - content: |
      She knows the real coordinates of the Artemis Nine.
    reveal_condition: keyword_trigger
    triggers:
      - artemis
      - missing ship
      - wreck coordinates
```

---

### `memory_config`

Controls the tiered memory system. The defaults work well for most characters — adjust only if you're building something that needs a longer or shorter memory window.

```yaml
memory_config:
  short_term_limit: 10        # Turns held in immediate context
  long_term_limit: 50         # High-importance memories kept long-term
  episodic_limit: 20          # High-level episode summaries
  consolidation_threshold: 5  # Short-term entries before consolidation runs
  relevance_decay: 0.95       # How quickly old memories lose weight
```

---

### `system_prefix`

A brief instruction prepended to every Version B system prompt. Use it to set the frame — "You are roleplaying as X. Stay in character at all times." Keep it short. The character definition does the heavy lifting; this just sets the mode.

```yaml
system_prefix: |
  You are roleplaying as Reva Kasai. Stay in character at all times.
  Never break character or reference being an AI.
```

---

## 6. Rubric YAML Field Reference

Rubric files live in `rubrics/`. They define structured scoring guides for evaluating character responses using an LLM-as-judge approach (via DeepEval).

---

### `name`

The rubric's identifier. Used as the filename (`rubrics/<name>.yaml`) and referenced when running evaluations.

```yaml
name: voice_consistency
```

---

### `version`

A string version number. Used to track changes over time — treat this like a document version, not a software version.

```yaml
version: "1.2"
```

---

### `description`

What this rubric measures and why it matters. Write this for a future reader who will use the rubric scores to make decisions about the character — what should they be looking for?

---

### `weight`

A float (default 1.0) used when combining multiple rubric scores. If one rubric matters more than others in a particular evaluation, give it a higher weight.

```yaml
weight: 1.0
```

---

### `criteria`

A list of individual scoring criteria. Each criterion becomes a separate LLM-as-judge evaluation. Each has:

- **`id`** — a short identifier (`vocabulary`, `tone`, etc.)
- **`label`** — a human-readable name for the criterion
- **`prompt`** — the scoring instruction given to the judge model

The `prompt` field is the most important. Write it the way a screenwriter or script editor would think about the problem — in natural language, with concrete examples of what good and bad look like. The judge model is running a close read of the conversation; give it the specific things to look for.

```yaml
criteria:
  - id: vocabulary
    label: Vocabulary & Idiom Consistency
    prompt: >
      Does the character use the vocabulary, slang, and idiomatic expressions
      consistent with their defined speaking style? Look for drift toward
      generic AI phrasing ("Certainly!", "Of course!", "I understand") as a
      strong negative signal.
```

---

### `scale`

Anchor descriptions for each score from 1 to 5. These tell the judge what each score level means for *this particular rubric*. Be specific — a generic "1 is bad, 5 is good" scale is much less useful than anchors that describe exactly what failure and success look like for this criterion.

```yaml
scale:
  1: "No recognisable voice — reads like a generic AI"
  2: "Occasional flashes of the intended voice, mostly inconsistent"
  3: "Broadly recognisable, notable lapses in two or more dimensions"
  4: "Strong voice throughout, only minor excusable deviations"
  5: "Immaculate — every line is unmistakably this character"
```

---

### `notes`

Use this field as an iteration log. When you revise the rubric, add a dated note explaining what changed and why. Over time this becomes the rubric's version history and captures the reasoning behind scoring decisions that might not be obvious later.

```yaml
notes: >
  v1.0 Initial version. v1.1 Added explicit mention of AI filler phrases
  after they kept appearing in low-scoring conversations without being
  called out. v1.2 Tightened scale anchor at 3 after inter-rater discussion.
```

---

## 7. Transcript Ingester

`ingest_transcript.py` takes a plain-text conversation transcript and reverse-engineers a character YAML from it. Useful for capturing a character that exists in existing writing — a script, a chat log, a novel excerpt — without building from scratch.

**Transcript format**

Each line should be `Speaker: message text`. Two speakers, alternating. The speaker labels can be anything — the script will detect them automatically, or you can specify which is the character with `--character-label`.

```
User: Hey, what's your name?
Reva: Name's Reva. Who's asking?
User: Just passing through.
Reva: Nobody just passes through here.
```

**Running it**

```bash
# Basic — outputs YAML to stdout
python ingest_transcript.py transcript.txt

# Write to a file
python ingest_transcript.py transcript.txt -o characters/aria.yaml

# Specify which speaker is the character
python ingest_transcript.py transcript.txt --character-label Aria

# Add a provenance note to the output YAML
python ingest_transcript.py transcript.txt --notes "From session recording, 2024-01-15"

# Use a different model
python ingest_transcript.py transcript.txt --model claude-sonnet-4-6
```

**What comes out**

A fully structured character YAML with every field populated from the transcript — description, personality, background, speaking style, goals, quirks, relationships, secrets (inferred from evasions and subtext), voice samples (verbatim exchanges), and a system prefix.

Fields that were directly stated in the transcript are left uncommented. Fields that were inferred from context — which is most of them — are flagged with a comment:

```yaml
background: |  # inferred - verify
  Grew up in a port city, possibly on the coast. References to
  tidal schedules and fishing boats suggest early life near water.
```

Read through every `# inferred - verify` field before using the character. The model is good at this, but inference is not observation — verify the things that matter.

---

## 8. Running an Eval

`run_eval.py` scores a saved conversation against one or more rubrics. The conversation log is a YAML file in the format the Chat page exports.

```bash
# Score against all rubrics in rubrics/
python run_eval.py \
    --character characters/reva_sample.yaml \
    --log conversation.yaml

# Score against specific rubrics
python run_eval.py \
    --character characters/reva_sample.yaml \
    --log conversation.yaml \
    --rubrics voice_consistency emotional_authenticity

# Save results as JSON
python run_eval.py \
    --character characters/reva_sample.yaml \
    --log conversation.yaml \
    --output-json results.json
```

**Conversation log format**

Save a log from the Chat page using the Save Log button. The format is:

```yaml
character: reva_sample
turns:
  - role: user
    content: "Hey, what's your name?"
  - role: assistant
    content: "Name's Reva. Who's asking?"
```

**Included rubrics**

| File | What it measures |
|---|---|
| `rubrics/voice_consistency.yaml` | Vocabulary, tonal register, sentence rhythm, worldview coherence |
| `rubrics/emotional_authenticity.yaml` | Emotion proportionality, continuity, specificity, suppression |
| `rubrics/secret_management.yaml` | Withholding fidelity, evasion quality, reveal timing and framing |

---

## 9. For Daley — Getting Set Up

This section is written for someone who hasn't done this before. Take it one step at a time.

---

**If you have a previous version of Persona Engine on your computer:**

Open Finder (the smiley-face icon in your Dock). In the search bar at the top right, search for `persona-engine`. If you find a folder with that name, drag it to the Trash. You want a clean start.

---

**Download the project:**

1. Go to [https://github.com/msonnenschein-art/persona-engine](https://github.com/msonnenschein-art/persona-engine) in your browser.
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Your Downloads folder will have a file called `persona-engine-main.zip`. Double-click it to unzip.
5. You'll see a folder called `persona-engine-main`. Rename it to `persona-engine` (click once to select, press Return, type the new name, press Return again).
6. Move it to your home folder. Your home folder is the one with your name on it — in Finder, look in the sidebar under Locations or Favourites, or press **Command + Shift + H** to go there directly. Drag the `persona-engine` folder there. This is the same folder that Terminal refers to as `~/` — so `~/persona-engine` just means "the persona-engine folder inside your home folder." To confirm it's in the right place, type `ls ~/persona-engine` in Terminal and press Return — if you see a list of files, you're good. If you get an error, the folder is in the wrong place.

---

**Open Terminal:**

Press **Command + Space** to open Spotlight. Type `Terminal` and press Return. A black or white window with a text prompt will open. This is where you'll type the setup commands.

---

**First, check if Python is installed:**

In Terminal, type `python3 --version` and press Return. If you see a version number (3.9 or higher), you're good. If you get `command not found`, go to [python.org](https://www.python.org), download the latest version, install it, then come back here.

---

**Run these commands, one at a time:**

Copy each line, paste it into Terminal with **Command + V**, and press Return. Wait for each one to finish before running the next.

```bash
cd ~/persona-engine
```
*(This moves you into the project folder.)*

```bash
python3 -m venv venv
```
*(This creates a private Python environment for the project. It may take a moment.)*

```bash
source venv/bin/activate
```
*(This activates the environment. You'll see `(venv)` appear at the start of your Terminal prompt.)*

```bash
pip install -r requirements.txt
```
*(This installs all the software the project needs. This one takes a few minutes — you'll see a lot of text scroll past. Wait until the prompt returns.)*

```bash
cp .env.example .env
```
*(This creates your settings file.)*

---

**Add your API key:**

Open Finder and navigate to your `persona-engine` folder. You'll see a file called `.env`.

> If you don't see it: in Finder, go to the menu bar and click **View → Show Hidden Files** (or press **Command + Shift + .**). Files that start with a dot are hidden by default.

Right-click `.env` and choose **Open With → TextEdit**.

You'll see a line that says:
```
ANTHROPIC_API_KEY=your-key-here
```

Replace `your-key-here` with your actual API key. Your key is at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys). Copy it from there and paste it after the `=` sign. Save the file (**Command + S**).

---

**Launch the app:**

Back in Terminal:

```bash
streamlit run app.py
```

Your browser will open automatically with the Persona Engine interface. If it doesn't, go to `http://localhost:8501`. Note: the app only works while Terminal is running. If you close Terminal, the browser tab will stop working — just run the three startup lines again to bring it back.

The app has five pages in the left sidebar: **Character Creator** (build and edit characters), **Knowledge Base** (upload background documents), **Rubric Builder** (create scoring guides), **Transcript Ingester** (turn an existing conversation into a character — paste or upload a transcript and it figures out the rest), and **Chat** (talk to your character). If someone has given you a transcript to work from, start with Transcript Ingester. If you're building a character from scratch, start with Character Creator.

---

**Every time you come back:**

You only need to run setup once. After that, whenever you want to use Persona Engine, open Terminal and run these three lines:

```bash
cd ~/persona-engine
source venv/bin/activate
streamlit run app.py
```

When you're done, close the browser tab and press **Control + C** in Terminal to stop the server.

---

## 10. Architecture Overview

The engine is built around a single central object — the `PersonaOrchestrator` — which assembles the system prompt fresh each turn in Version B mode. Here's how the pieces fit together:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Persona Engine                            │
│                                                                  │
│   characters/*.yaml                                              │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   PersonaOrchestrator                    │   │
│   │                                                         │   │
│   │  Per-turn assembly (Version B):                         │   │
│   │   1. Character identity + voice + dramatic register     │   │
│   │   2. Current emotional & relationship state             │   │
│   │   3. Relevant memories (tiered retrieval)               │   │
│   │   4. Conditionally revealable secrets                   │   │
│   │   5. RAG-retrieved knowledge chunks                     │   │
│   └──────────────────────────┬──────────────────────────────┘   │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │State Tracker│    │Memory Manager│    │  RAG Manager     │   │
│  │             │    │              │    │                  │   │
│  │ - emotions  │    │ - short-term │    │ ChromaDB +       │   │
│  │ - trust     │    │ - long-term  │    │ sentence-        │   │
│  │ - rapport   │    │ - episodic   │    │ transformers     │   │
│  │ - tension   │    │              │    │ knowledge/       │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                  │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────────┐                         │
│                    │   LLM Adapter    │                         │
│                    │                  │                         │
│                    │ Anthropic Claude  │                         │
│                    │ OpenAI GPT        │                         │
│                    └──────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
```

**Core modules:**

| Module | What it does |
|---|---|
| `core/schema.py` | Parses character YAML into typed Python objects. Handles secrets, memory config, voice samples, dramatic register fields. |
| `core/orchestrator.py` | The main engine. Assembles system prompts, routes messages, updates state, manages the conversation loop. |
| `core/state.py` | Tracks emotional state (primary emotion, intensity, decay) and relationship metrics (trust, familiarity, rapport, tension) across turns. |
| `core/memory.py` | Three-tier memory: short-term deque (recent turns), long-term (high-importance consolidated memories), episodic (session-level summaries). |
| `core/rag_manager.py` | Ingests documents into ChromaDB, chunks and embeds them with sentence-transformers, retrieves relevant passages per turn. All local — no external server. |
| `core/comparison.py` | Runs the same input through both a Version A and Version B orchestrator in parallel, returns both responses for side-by-side display. |
| `core/rubric_loader.py` | Parses rubric YAML files into DeepEval `GEval` metrics for LLM-as-judge evaluation. |
| `core/llm_adapter.py` | Thin abstraction over the Anthropic and OpenAI APIs. Handles both synchronous and streaming responses. |
