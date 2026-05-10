<div align="center">
  <h1>☯️ Taiji </h1>
  <p><em>A dual-core cognitive architecture based on the 六爻 (Liuyao) cycle</em></p>
  <p>Also running as 🐈 Vingobot — a lightweight, multi-channel AI agent with goal-driven coroutines.</p>

  <p>
    <img src="https://img.shields.io/badge/python-%3E%3D3.11-blue" alt="Python 3.11+" />
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0" />
    <img src="https://img.shields.io/badge/status-alpha-yellow" alt="Alpha" />
  </p>
</div>

---

## Overview

**Vingobot** is an AI agent framework built on the **六爻循环 (Liuyao Cycle)** architecture — a dual-loop coroutine-driven design that separates goal-level planning from task-level execution. It combines:

- **🧠 Multi-LLM support** — OpenAI, Anthropic, DeepSeek, Gemini, Groq, local models (Ollama, LM Studio), and 20+ providers via a unified interface
- **📨 Multi-channel messaging** — Telegram, Discord, Feishu/Lark, DingTalk, Slack, QQ, WeChat Work (企业微信), Weixin (微信), Email, Matrix, WhatsApp, Microsoft Teams, and WebSocket — all with a shared agent core
- **🎯 Liuyao (六爻) mode** — A coroutine worker pool that autonomously drives goals through the six-phase cognitive cycle, with the **Weaver (织)** building each round's system prompt from scratch
- **🧬 DMN (Default Mode Network)** — A background cognitive-evolution consumer that extracts patterns from task execution and updates the agent's cognition library autonomously

But more fundamentally: **Taiji is not just another agent framework. It is a cognitive architecture that learns.**

Most frameworks (AutoGPT, LangGraph, CrewAI, OpenAI Agents SDK) optimize execution speed and tool-calling reliability — but they never improve from task to task. After 100 runs, they are exactly as "smart" as after run 1. Taiji's 六爻 cycle makes every task round a complete **action + reflection** loop: you produce output, then step back to analyze what worked and what didn't, updating the cognition library before moving on. This is the difference between **repeating** and **evolving**.

> Other frameworks build a better Swiss Army knife.  
> Taiji builds a craftsman who gets sharper with every job.

---

## Architecture: The Dual-Loop Design

Instead of a single agent loop that does everything, Taiji separates cognition into two independent loops — a pattern that mirrors the human brain's **Default Mode Network** and **Task-Positive Network**:

```
┌────────────────────────────────────────────────────────┐
│                    DMN (Background)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Analyze  │→ │ Extract  │→ │ Update   │  ◄── Always   │
│  │ Task Log │  │ Patterns │  │ Cognition│      running  │
│  └──────────┘  └──────────┘  └──────────┘              │
│  [micro-evolution: per-task] [macro-evolution: 20-task] │
└──────────────────────┬─────────────────────────────────┘
                       │ feeds wisdom back
                       ▼
┌────────────────────────────────────────────────────────┐
│              TPN (六爻  Goal Pool)                       │
│                                                         │
│   Worker 1 ──► 明觉 → 织 → 阳 → 阴 → 行动 → 暗驱 → 思变  │
│   Worker 2 ──► 明觉 → 织 → 阳 → 阴 → 行动 → 暗驱 → 思变  │
│   Worker 3 ──► 明觉 → 织 → 阳 → 阴 → 行动 → 暗驱 → 思变  │
│                                                         │
│   Each worker = one independent 六爻 cycle               │
└────────────────────────────────────────────────────────┘
```

### TPN (六爻行动池) — Task execution with built-in reflection

TPN is the foreground loop. It runs a coroutine pool of workers, each driving a goal through the six-phase cycle:

| Phase | Role | What It Does |
|-------|------|-------------|
| 明觉 (Míngjué) | **Perceiver** | Assesses current state, reads goal blueprint & trajectory, identifies gaps |
| 织 (Weaver) | **Strategist** | Weaves the system prompt from scratch — selectively loads L1–L5 cognition assets, never reusing raw history from previous rounds |
| 阳 (Yáng) | **Decider** | Translates the weaved strategy into concrete intent: what tools to call, what to produce |
| 阴 (Yīn) | **Gatekeeper** | Validates safety and policy compliance before any action is taken |
| 行动 (Action) | **Executor** | Produces actual output — writes files, calls APIs, searches the web, generates media |
| 暗驱 (Anqu) | **Reflector** | Analyzes the action log, extracts success/failure patterns, feeds observations to DMN |
| 思变 (Sībiàn) | **Meta-Decider** | Decide: continue (deepen), iterate (refine), fork (new branch), or complete (done & return) |

**Key constraint**: No two consecutive rounds of pure reading. Every round must produce at least one tangible output (file write, API call, search result). This prevents the self-referential read-loop death spiral that plagues open-ended autonomous agents.

### DMN (认知演化) — Background cognition evolution

DMN is the background loop. It runs independently of TPN, consuming task execution logs and extracting cognitive value:

- **Micro-evolution** (per task): Extracts reusable patterns → writes L1 Skills or updates L2 model confidence
- **Macro-evolution** (every 20 tasks): Clusters redundant experiences, consolidates L3 grid weights, extracts new L4 truths

DMN ensures that the system doesn't just execute — it **learns**. Over time, the same task takes fewer rounds, produces higher quality, and requires less reinvention.

### The Weaver: "Doing nothing" is the most important job

Unlike every other framework that prepends raw conversation history as context, the **Weaver (织)** starts each round from zero:

```python
# Every framework does this:
system_prompt = base_prompt + "\n\nHistory:\n" + truncate(history)
# History grows unbounded, LLM attention dilutes

# Taiji's Weaver does this:
system_prompt = weave_from_scratch(
    goal_context,        # Current goal state + blueprint
    cognition_assets,    # Selectively loaded L1–L5 (not everything)
    task_spec,           # The specific task for this round
    anqu_reflections     # Last round's reflections (if continuing)
)
# No raw history. Fights attention dilution by design.
```

This "empty start" (虚而待物) is counterintuitive — why throw away history? Because:

1. **Attention is a scarce resource**. Raw history buries signal in noise.
2. **Each round is a fresh cognitive act**. The last round's language habits pollute the next.
3. **Learning is compressed, not stored raw**. Anqu extracts patterns; those patterns (not the raw logs) are fed forward.

---

## Why This Matters (Not a Pitch — A Technical Position)

**Taiji is not a better AutoGPT.** It is a fundamentally different approach to building agents that learn.

The mainstream approach today is **deterministic DAGs with LLM nodes** — LangGraph's state machines, CrewAI's role assignments, OpenAI's function pipelines. These are great for reliable, traceable workflows. But they don't learn.

Taiji's position (backed by working code and real task logs):

| Capability | Mainstream Frameworks | Taiji |
|-----------|----------------------|-------|
| Tool calling | ✅ Excellent | ✅ Equivalent |
| State management | ✅ LangGraph state machines | ✅ Weaver + L1–L5 cognition |
| Multi-agent orchestration | ✅ CrewAI roles | ❌ Single agent, multiple cognitive stances |
| **Cross-task learning** | ❌ Each task starts blank | ✅ DMN extracts patterns, L1–L5 evolves |
| **Task quality over time** | ❌ Flat | ✅ Improves (fewer rounds, better output) |
| **Self-reflection** | ❌ Not built-in | ✅ Built into every round (思变) |
| **Attention efficiency** | ❌ History grows unbounded | ✅ Weaver's empty start per round |
| **Read-loop prevention** | ❌ Open-ended agents often spin | ✅ Hard constraint: no 2 consecutive read-only rounds |

This is not theoretical. The `silver-economy` goal (automatic short-video production pipeline) and `monthly-token-income` goal (cross-border industrial information arbitrage) ran on this architecture and produced measurable improvements across iterations. Task logs from those runs are in the repository.

---

## System Architecture

```
~/.vingobot/.taiji/
├── pending/                    ← Task queue (atomic, FIFO consumption)
├── cognition/
│   ├── skills/                 ← L1: Skill library (code + SKILL.md)
│   ├── models/                 ← L2: Experience models (confidence-weighted)
│   ├── grids/                  ← L3: Cognition grids (四象·六爻·八卦)
│   ├── truths/                 ← L4: Immutable truths (extracted, validated)
│   └── soul/                   ← L5: Self-model (SOUL.md, AGENTS.md, TOOLS.md, USER.md, MEMORY.md)
└── goals/
    ├── silver-economy/         ← Active: short-video monetization pipeline
    ├── monthly-token-income/   ← Active: cross-border industrial info arbitrage
    └── default/                ← Default dialogue target (no 六爻)
```

### L1–L5 Cognition Hierarchy

The cognition library is structured as five independent layers. Each layer has its own storage, retrieval patterns, and update mechanisms:

- **L1 Skills** (`skills/`): Executable code + instructions. Installed from GitHub or written to disk. Updated via DMN micro-evolution when a successful pattern repeats across multiple tasks.
- **L2 Models** (`models/`): Confidence-weighted experience models. Store the probability that a certain tool/strategy works in a given context. Updated each time a pattern is validated or disproven.
- **L3 Grids** (`grids/`): Meta-cognition grids including 四象 (cognitive stance routing), 六爻 (phase routing), and 八卦 (situational tool selection). These determine how the Weaver adapts behavior to context.
- **L4 Truths** (`truths/`): Immutable truths extracted from cross-task learning — "never do X" or "always do Y in scenario Z". These are the highest-confidence patterns, reviewed and approved before promotion.
- **L5 Self** (`soul/`): The agent's self-model. Five files: SOUL.md (identity), AGENTS.md (behavior rules), TOOLS.md (tool usage conventions), USER.md (user profile), MEMORY.md (long-term episodic memory).

**The Weaver selectively loads from L1–L5 each round.** It does not dump everything into context. It reads the L5 self-model for identity, then loads relevant L1 skills, L2 models, and L4 truths based on the current goal context. What it does not load is irrelevant — and that's the point.

---

## Quick Start

### Install

```bash
pip install vingobot
```

### CLI Usage

```bash
# Start a conversation
vingobot repl

# Send to user
vingobot say "Hello from Taiji!" --channel telegram --chat-id 123456

# Chat from stdin
echo "分析今天天气" | vingobot chat --channel telegram --chat-id 123456

# Start the gateway (multi-channel message processing)
vingobot gateway
```

### Configuration

```bash
# Generate default config
vingobot init

# Edit ~/.vingobot/.vingobot.toml or /etc/vingobot/config.toml
```

Then start the gateway:

```bash
vingobot gateway
```

The agent processes messages from all channels through a shared message bus, with per-channel session isolation.

### Supported Channels

| Channel | Config Key | Notes |
|---------|-----------|-------|
| Telegram | `telegram` | Streaming, inline keyboards, proxy support |
| Discord | `discord` | Streaming, reactions, proxy support |
| Feishu / Lark | `feishu` | Streaming, emoji reactions, card messages |
| DingTalk | `dingtalk` | Stream mode |
| Slack | `slack` | Socket mode or webhook, threading |
| QQ | `qq` | Bot platform API |
| WeChat Work | `wecom` | 企业微信 bot |
| Weixin | `weixin` | 微信个人号 (ilinkai protocol) |
| Email | `email` | IMAP/SMTP with DKIM/SPF verification |
| Matrix | `matrix` | E2E support (Linux only) |
| WhatsApp | `whatsapp` | Via local Node.js bridge |
| Microsoft Teams | `msteams` | Bot Framework protocol |
| WebSocket | `websocket` | Custom real-time protocol, media support |
| Mochat | `mochat` | Custom chat platform |

---

## LLM Provider Support

Vingobot provides a **unified provider abstraction** that normalizes differences across 25+ LLM backends:

| Provider Family | Config Key | Features |
|----------------|-----------|----------|
| OpenAI | `openai` | Function calling, streaming, vision |
| Anthropic | `anthropic` | Extended thinking, tool use, streaming |
| DeepSeek | `deepseek` | Full compatibility |
| Google Gemini | `gemini` | Function calling |
| Groq | `groq` | Fast inference |
| OpenRouter | `openrouter` | Multi-model routing |
| Azure OpenAI | `azureOpenai` | Enterprise Azure deployment |
| AWS Bedrock | `bedrock` | Claude, Llama, Mistral on AWS |
| Ollama | `ollama` | Local models |
| LM Studio | `lmStudio` | Local OpenAI-compatible server |
| vLLM | `vllm` | High-throughput serving |
| HuggingFace | `huggingface` | Inference API / TGI |
| Zhipu (智谱) | `zhipu` | GLM series |
| DashScope (通义) | `dashscope` | Qwen series |
| Moonshot (月之暗面) | `moonshot` | Kimi series |
| MiniMax | `minimax` | MiniMax models |
| Mistral | `mistral` | Mistral AI |
| StepFun (阶跃星辰) | `stepfun` | Step series |
| SiliconFlow | `siliconflow` | Model hosting |
| Volcengine (火山引擎) | `volcengine` | Doubao series |
| BytePlus | `byteplus` | ByteDance API |
| Baidu Qianfan (百度千帆) | `qianfan` | ERNIE series |
| AIHubMix | `aihubmix` | Model aggregation |
| GitHub Copilot | `github_copilot` | Copilot models |

Providers are auto-detected from the model name, or explicitly configured with `"provider": "openrouter"` in the agent defaults.

---

## Programmatic Usage

Vingobot can be used as a Python library:

```python
import asyncio
from vingobot import vingobot

async def main():
    bot = vingobot.from_config()

    # Single-turn interaction
    result = await bot.run("What's the weather today?")
    print(result.content)

    # 六爻 mode (goal-driven autonomous pool)
    pool = await bot.start_sixiang(workers=3)
    # ... pool runs autonomously ...
    await bot.stop_sixiang()

    # DMN (background cognitive evolution)
    state = await bot.dmn_status()

asyncio.run(main())
```

### TPN Control (via CLI)

```bash
# Start/stop the 六爻 coroutine pool
vingobot tpn start
vingobot tpn stop

# List goals
vingobot tpn list

# Create a goal
vingobot tpn create silver-economy "银发经济全自动短视频带货" --priority 8 --self-driven 60

# Trigger one cycle of a goal
vingobot tpn trigger silver-economy
```

### DMN Control

```bash
# Check DMN status and cognition health
vingobot dmn status

# Manually trigger cognitive evolution
vingobot dmn trigger "check if we need a new tool skill"

# Run cross-task pattern analysis
vingobot dmn analyze
```

---

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Code Style

```bash
ruff check vingobot/
```

### Project Structure

```
vingobot/
├── vingobot/
│   ├── agent/             ← Agent loop, context, memory, skills, hooks
│   ├── api/               ← OpenAI-compatible API server
│   ├── bus/               ← Message bus (channel queue, delivery)
│   ├── channels/          ← Chat platform integrations
│   ├── cli/               ← CLI commands and REPL
│   ├── command/           ← Built-in commands
│   ├── config/            ← Configuration loader and schema
│   ├── core/              ← Workspace, queue, context, trajectory
│   ├── cron/              ← Scheduled task service
│   ├── goal/              ← 六爻 core: loop, tasks, cognition (TPN + DMN)
│   │   ├── coroutine.py   ← Sixiang coroutine pool (TPN)
│   │   ├── sixiang_loop.py ← Sixiang cycle logic (明觉→织→...→思变)
│   │   ├── cognition.py   ← DMN evolution consumer
│   │   ├── tasks.py       ← Task model and queue management
│   │   └── goals.py       ← Goal CRUD and lifecycle
│   ├── heartbeat/         ← Periodic heartbeat service
│   ├── providers/         ← LLM provider implementations
│   ├── security/          ← Security utilities (L4 write guard, L5 system guard)
│   ├── session/           ← Session persistence
│   ├── skills/            ← Skill system (L1 implementation)
│   ├── utils/             ← Shared utilities
│   └── web/               ← Web interface (WIP)
├── tests/                 ← Test suite
├── webui/                 ← Web UI (React, WIP)
├── bridge/                ← WhatsApp bridge (TypeScript)
└── docs/                  ← Documentation
```

---

## Documentation

See the [`docs/`](./docs) directory for detailed guides:

- [Quick Start](./docs/quick-start.md)
- [Configuration](./docs/configuration.md)
- [CLI Reference](./docs/cli-reference.md)
- [WebSocket Protocol](./docs/websocket.md)
- [OpenAI API Compatibility](./docs/openai-api.md)
- [Memory System](./docs/memory.md)
- [Channel Plugin Guide](./docs/channel-plugin-guide.md)
- [Deployment](./docs/deployment.md)
- [Chat Commands](./docs/chat-commands.md)
- [Python SDK](./docs/python-sdk.md)
- [Agent Social Network](./docs/agent-social-network.md)

---

## Engineering Implementation Detail

This section covers the actual engineering decisions behind the 六爻 architecture.

### Weaver: Dynamic System Prompt Construction

The Weaver is implemented as a deterministic prompt composer, not an LLM call. It reads the L5 self-model, then selectively loads cognition assets matched to the current goal context:

```python
# Simplified Weaver logic
def weave(context: GoalContext, cognition: CognitionStore) -> str:
    parts = []
    # Always load L5 (self-model)
    parts.append(cognition.load_soul())
    # Load goal blueprint
    parts.append(context.blueprint_summary)
    # Load relevant L1 skills (matched by keyword on goal tags)
    parts.extend(cognition.match_skills(context.tags))
    # Load relevant L4 truths
    parts.extend(cognition.match_truths(context.tags))
    # Load Anqu reflections from last round (if continuing)
    if context.last_anqu_output:
        parts.append(extract_essence(context.last_anqu_output))
    return "\n\n".join(parts)
```

The Weaver's key constraint: **maximum compressed context = 3 windows of LLM attention allocated to: identity + strategy + reflection**. No raw conversation history.

### Read-Loop Prevention

The `禁止连续两轮只读不写` constraint is enforced at the task runner level:

```python
class TaskRunner:
    def __init__(self):
        self._iteration = 0
        self._last_write_iteration = 0

    def record_write(self):
        self._last_write_iteration = self._iteration

    def check_read_loop(self):
        if self._iteration - self._last_write_iteration >= 2:
            # Force-terminate the task with a note
            self.terminate("Read-loop detected: 2 consecutive iterations without write")
```

This is not a heuristic. It is a hard termination condition. The task log records the termination reason, and DMN can analyze whether the constraint caused premature termination or correctly prevented wasted cycles.

### DMN: Two-Tier Evolution

```python
class DMNEvolver:
    def micro_evolve(self, task_log: TaskLog):
        """After each completed task, extract patterns."""
        patterns = self.extract_patterns(task_log)
        for pattern in patterns:
            if pattern.confidence >= 0.7:
                self.update_l2_model(pattern)
            if pattern.confidence >= 0.9 and pattern.repeat_count >= 3:
                self.promote_to_l1_skill(pattern)

    def macro_evolve(self):
        """Every 20 tasks, consolidate L3 grids and extract L4 truths."""
        self.cluster_l2_models()
        self.update_grid_weights()
        self.extract_l4_truths()
        self.clean_redundant_skills()
```

DMN runs as a background asyncio consumer with its own event queue. It does not block TPN execution. When TPN produces a task log, it's enqueued for DMN consumption. DMN processes logs at its own pace, independent of the TPN worker pool.

### Goal System: The Blueprint Contract

Every goal in Taiji has a **blueprint** — a structured document that defines:

1. **Done criteria**: The conditions under which the goal is considered achieved
2. **Constraints**: Hard rules that the Weaver must respect (e.g., "domestic first, overseas later")
3. **Quality standards**: What "good enough" looks like for this goal's outputs

The blueprint is stored in two places: `meta.json` (for runtime access via `blueprint_summary`) and `blueprint.md` (for human reading). Both must be kept in sync — the system reads from `meta.json`, but the human edits `blueprint.md` and expects it to reflect the actual running configuration.

---

## Comparison with Other Frameworks

| Feature | AutoGPT | LangGraph | CrewAI | OpenAI Agents SDK | Taiji |
|---------|---------|-----------|--------|------------------|-------|
| Loop type | Single-thread dead loop | State machine DAG | Role orchestration | Pipeline | Cognitive cycle |
| Learning | None | None | None | None | L1–L5 evolution |
| Reflection | None | None | None | None | Built-in every round |
| Context strategy | Full history | State + history | Role prompts | Full history | Weaver: context from scratch |
| Read-loop protection | None | N/A (deterministic) | N/A (orchestrated) | None | Hard constraint |
| Background evolution | None | None | None | None | DMN consumer |

---

## Thinkers and Works

**I Ching (《易经》)** — The Book of Changes is the philosophical and structural foundation of the entire Taiji architecture. The core mechanisms of this project are directly inspired by its core concepts:

- **六爻 (Liuyao / Six Lines)** — The six-stage execution cycle that governs the agent's task progression, from initial observation (初爻) through action, refinement, mastery, to meta-cognitive reflection and return (上爻). This is the cognitive engine that drives the Taiji agent's goal-oriented behavior.
- **八卦 (Bagua / Eight Trigrams)** — The eight situational routing patterns (乾·创造, 坤·承载, 震·启动, 巽·渗透, 坎·险陷, 离·明照, 艮·静止, 兑·喜悦) that dynamically guide the agent's cognitive stance and tool selection based on the current task context.
- **四象 (Sixiang / Four Images)** — The four dynamic cognitive modes (老阳·发散探索, 少阳·聚焦感知, 少阴·精准执行, 老阴·批判反思) that govern the agent's thinking temperature and output style at each step.

The I Ching's central insight — that all things are in continuous transformation through cyclical patterns — directly informs this system's design principle: **cognition evolves through structured cycles, not linear accumulation.** The 编织器 (Weaver) ensures that each cycle begins fresh, unpolluted by the language habits of previous rounds, while the 暗驱 (Anqu) ensures that long-term goals persist across cycles. This is a computational homage to the ancient wisdom that stillness and motion, emptiness and form, are not opposites but complementary phases of a single cosmic breath.

**Charlie Munger** and his book *Poor Charlie's Almanack* — Mr. Munger's "Multiple Mental Models" and "Latticework of Mental Models" are the cornerstone of this project's "Cognitive Grills" concept, encouraging us to transcend disciplinary boundaries and build cognitive networks that connect different knowledge domains.

**Ray Dalio** and his book *Principles* — Dalio's systematic principle-based thinking and believability-weighted decision-making methods provide valuable references for our "Underlying Truths" extraction and value judgment mechanisms in "Dream Management."

**Wang Yangming (王阳明)** and his philosophy of "Unity of Knowledge and Action (知行合一)" from *Instructions for Practical Living (《传习录》)* — The core concepts of "Knowledge is the beginning of action, and action is the completion of knowledge" and "Where knowledge is genuine and solid, that is action; where action is clear and perceptive, that is knowledge" profoundly influenced the design of this system's "Unity of Knowledge and Action Verification" mechanism. We believe that true cognition must be validated through action, and every action should deepen cognition. This aligns perfectly with the system's compression evolution process of "Event Experiences → Mental Models → Underlying Truths." Particularly, the idea that "the moment a thought arises is action" provides important inspiration for our design of the closed-loop mechanism between "Meta-Awareness" and "Dream-Driven" evolution.

**The Secret of the Golden Flower (《太乙金华宗旨》)** (attributed to Lü Dongbin 吕洞宾) — This Taoist classic's teachings on inner observation, returning light, and golden flower cultivation inspired us to incorporate Eastern philosophical introspection wisdom into "Meta-Awareness" and the "Self Awareness Layer," pursuing the introspection and evolution of consciousness.

**Daniel Kahneman** and his book *Thinking, Fast and Slow* — Professor Kahneman's cognitive model of System 1 and System 2 directly influenced the design of this system's "Awake/Asleep" state switching, simulating the alternation between human rapid intuition and deep reflection.

### Open Source Projects

**DeepSeek's Engram Project** (deepseek-ai/Engram) — The Engram project's exploration in long-term memory and knowledge management provides important technical inspiration for our "Life Memory System" and "Associative Memory Network."

**The University of Hong Kong's Nanobot Project** — This project's innovative work in cognitive architecture or robotics inspired our thinking about system autonomy and emotional mechanisms.

These contributors have illuminated our design path in different ways. We hereby express our sincere thanks.

---

## License

Licensed under the **Apache License, Version 2.0** (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
