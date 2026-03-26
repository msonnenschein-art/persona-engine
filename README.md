# Persona Engine

A character AI framework for dynamic, stateful conversations with richly defined fictional characters.  Supports a static baseline (Version A) and a full dynamic context-assembly system (Version B) with tiered memory, emotional state tracking, relationship metrics, a RAG knowledge base, and YAML-driven evaluation rubrics.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add your API key(s)

# Interactive chat — Version B (dynamic)
python cli.py -c reva_sample

# Static baseline — Version A
python cli.py -c reva_sample -m a

# Single message
python cli.py -c reva_sample --message "What's your deal?"

# With RAG knowledge base enabled
python cli.py -c reva_sample --rag

# Side-by-side comparison (A vs B)
python cli.py -c reva_sample --compare

# Use OpenAI instead of Anthropic
python cli.py -c reva_sample -p openai
```

---

## Architecture

### Two operating modes

| | Version A | Version B |
|---|---|---|
| System prompt | Static, loaded once from YAML | Rebuilt every turn |
| Memory | None | Three-tier (short-term, long-term, episodic) |
| Emotional state | None | Tracked, decays over time |
| Relationship | None | Trust, familiarity, rapport, tension |
| Secrets | Baked into prompt | Conditionally revealed at runtime |
| Knowledge base | Not used | RAG retrieval injected per turn |

### Core modules

```
core/
  schema.py          Character definition and secret system
  state.py           Emotional state + relationship metrics
  memory.py          Three-tier memory (short-term / long-term / episodic)
  rag_manager.py     ChromaDB-backed cold memory tier (RAG)
  orchestrator.py    Main orchestration engine
  comparison.py      Baseline comparison mode (A vs B)
  rubric_loader.py   YAML rubric → DeepEval metric conversion
  llm_adapter.py     Model-agnostic LLM abstraction (Anthropic, OpenAI)
```

---

## Characters

Characters are defined as YAML files under `characters/`.

```yaml
name: Reva Kasai
description: ...
personality: ...
speaking_style: ...
secrets:
  - content: "I owe Dax a debt I can't repay."
    reveal_condition: TRUST_THRESHOLD
    threshold: 0.6
version_a_prompt: |      # full static prompt for Version A
  You are Reva Kasai...
system_prefix: |          # prepended to the Version B dynamic prompt
  You are roleplaying as Reva Kasai...
```

Reveal conditions: `NEVER` · `TRUST_THRESHOLD` · `EXPLICIT_ASK` · `EMOTIONAL_STATE` · `KEYWORD_TRIGGER`

---

## RAG — Knowledge Base

Drop `.txt`, `.md`, or `.yaml` files into `knowledge/`.  Enable with `--rag`:

```bash
python cli.py -c reva_sample --rag
```

The RAG manager uses **ChromaDB** with local persistent storage (`.chroma/`) and sentence-transformer embeddings — no external server required.  Documents are re-indexed automatically when the collection is empty.

Programmatic usage:

```python
from core.rag_manager import RAGManager
from core.orchestrator import create_orchestrator

rag = RAGManager(knowledge_dir="knowledge")
rag.ingest_directory()           # or ingest_file("knowledge/lore.md")

orchestrator = create_orchestrator("characters/reva_sample.yaml", rag=rag)
```

---

## Baseline Comparison Mode

Run the same message through Version A and Version B simultaneously and see both responses side by side — useful for client demos.

```bash
python cli.py -c reva_sample --compare
```

Programmatic usage:

```python
from core.comparison import BaselineComparison

comparison = BaselineComparison.from_character_path("characters/reva_sample.yaml")
result = comparison.chat("What happened on Callisto?")
print(result.format())
```

---

## Evaluation — Rubric System

### Rubric format

Rubrics live in `rubrics/` as YAML files.  No code changes are needed to add or swap rubrics.

```yaml
name: voice_consistency
version: "1.0"
description: >
  Evaluates how consistently the character maintains their voice...
weight: 1.0
criteria:
  - id: vocabulary
    label: Vocabulary & Idiom Consistency
    prompt: >
      Does the character use vocabulary consistent with their speaking style?
scale:
  1: "No recognisable voice"
  2: "Occasional flashes of the intended voice"
  3: "Broadly recognisable with notable lapses"
  4: "Strong voice with minor deviations"
  5: "Immaculate consistency"
notes: "Initial version."
```

### Included rubrics

| File | What it measures |
|---|---|
| `rubrics/voice_consistency.yaml` | Vocabulary, tone, sentence rhythm, worldview coherence |
| `rubrics/emotional_authenticity.yaml` | Proportionality, continuity, specificity, suppression |
| `rubrics/secret_management.yaml` | Withholding fidelity, evasion quality, reveal timing & framing |

### Running evaluations

```bash
# Score all rubrics
python run_eval.py \
    --character characters/reva_sample.yaml \
    --log conversation.yaml

# Score specific rubrics
python run_eval.py \
    --character characters/reva_sample.yaml \
    --log conversation.yaml \
    --rubrics voice_consistency emotional_authenticity

# Save JSON output
python run_eval.py \
    --character characters/reva_sample.yaml \
    --log conversation.yaml \
    --output-json results.json
```

### Conversation log format

```yaml
character: reva_sample
turns:
  - role: user
    content: "Hey, what's your name?"
  - role: assistant
    content: "Name's Reva. Who's asking?"
```

### Programmatic usage

```python
from core.rubric_loader import load_rubrics_from_dir
from deepeval.test_case import LLMTestCase

rubrics = load_rubrics_from_dir("rubrics", names=["voice_consistency"])
metric = rubrics[0].to_deepeval_metric()

test_case = LLMTestCase(
    input="full conversation transcript",
    actual_output="last assistant response",
    context=["character description and traits"],
)
metric.measure(test_case)
print(metric.score, metric.reason)
```

---

## CLI reference

```
python cli.py -c CHARACTER [options]

  -c, --character       Character file name or path (required)
  -m, --mode            a (static) or b (dynamic, default)
  -p, --provider        anthropic (default) or openai
  --model               Override model name
  --message             Single message (non-interactive)
  --stream              Stream responses
  --compare             Baseline comparison mode (A vs B side by side)
  --rag                 Enable RAG knowledge base
  --knowledge-dir       Knowledge base directory (default: knowledge/)
  --characters-dir      Characters directory (default: characters/)
  --max-tokens          Response token limit (default: 1024)
  --list                List available characters

In-session commands:
  /quit /exit           End session
  /state                Show current state (Version B only)
  /reset                Reset conversation
  /save <file>          Save session to JSON
  /load <file>          Load session from JSON
  /mode <a|b>           Switch mode
```

---

## Environment variables

```
ANTHROPIC_API_KEY       Required for Anthropic models
OPENAI_API_KEY          Required for OpenAI models
PERSONA_DEFAULT_PROVIDER  Optional default provider
PERSONA_DEFAULT_MODEL     Optional default model override
```
