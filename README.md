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
- **🎯 Liuyao (六爻) mode** — A coroutine pool for goal-driven autonomous task execution, with a cognition hierarchy (skills → models → grids → truths → soul)
- **🔧 Tool-agnostic execution** — Native function calling, MCP servers, command execution, web search, file I/O, and a plugin tool system
- **💭 Dream (consolidation)** — Automatic memory consolidation that summarizes and compresses long-running conversations
- **🔌 Extensible** — Plugin architecture for channels, tools, provider backends, and lifecycle hooks

---

## Features

| Area | Capabilities |
|------|-------------|
| **LLM Providers** | OpenAI, Anthropic, DeepSeek, Gemini, Groq, Ollama, LM Studio, vLLM, OpenRouter, Azure OpenAI, Bedrock, HuggingFace, Zhipu (智谱), DashScope (通义千问), Moonshot (月之暗面), MiniMax, Mistral, StepFun (阶跃星辰), SiliconFlow, Volcengine (火山引擎), BytePlus, Baidu Qianfan (百度千帆), AIHubMix, GitHub Copilot, and more |
| **Chat Channels** | Telegram, Discord, Feishu/Lark, DingTalk, Slack, QQ, WeChat Work (企业微信), Weixin (微信), Email (IMAP/SMTP), Matrix, WhatsApp, Microsoft Teams, WebSocket, Mochat |
| **Agent Core** | Session management, tool-use loops, context window optimization, streaming responses, auto-compaction, memory consolidation, skill loading, MCP integration |
| **六爻 (Liuyao)** | Coroutine worker pool, goal-driven task decomposition, cognition hierarchy (L1 skills → L2 models → L3 grids → L4 truths → L5 soul), inner task loop (Weaver → Yang → Yin → Executor), outer decision loop (Mingjue → Anqu) |
| **Tools** | Web search (DuckDuckGo, custom), web fetch (Jina Reader), file I/O, command execution, code execution, MCP servers, cron scheduling, TPN (pool control), DMN (decision management), custom user tools |
| **Extras** | Dream consolidation, heartbeat service, OpenAI-compatible API server, WebSocket gateway, rich CLI (prompt_toolkit), programmatic Python SDK |

---

## Architecture: The 六爻 (Liuyao) Cycle

Vingobot's core innovation is the **六爻循环** — a dual-loop architecture inspired by the six lines (六爻) of the I Ching:

The core of Vingobot is a dual-loop architecture inspired by the I Ching's six lines (六爻). The **Outer Loop** handles goal-level planning and decision, while the **Inner Loop** executes tasks through a sequence of specialized cognitive agents.

```
┌─────────────────────────────────────────────────────┐
│                  外循环 (Outer Loop)                   │
│                                                       │
│  目标启动 → 初爻·明觉 (Mingjue) → 任务队列 → 决策·暗驱     │
│                                     ↑        │       │
│                                     └────────┘       │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │           内循环 (Inner Loop)                 │     │
│  │    二爻·编织器 → 三爻·阳 → 四爻·阴 → 五爻·执行器   │     │
│  │    (Weaver)   (Yang)   (Yin)  (Executor)    │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

- **初爻·明觉 (Mingjue)** — Translates goals into concrete tasks, applies八卦 routing for tool selection
- **二爻·编织器 (Weaver)** — Builds context with three-layer system prompts, dynamic tool definitions, temperature tuning, and infinite-loop detection
- **三爻·阳 (Yang)** — Receives the model's reasoning and native function calling output; detects task completion signals
- **四爻·阴 (Yin)** — Rule-based approval gate: read-only operations pass automatically, write operations trigger safety checks, dangerous commands are blocked
- **五爻·执行器 (Executor)** — Dual-layer safety enforcement: prefers ToolRegistry execution, falls back to built-in implementations
- **上爻·暗驱 (Anqu)** — Goal-level decision loop: continue next task, goal complete, goal failed, or rework

---

## Quick Start

### Installation

```bash
# Install from source (latest features)
pip install -e .

# Or install from PyPI
pip install vingobot

# Or with uv
uv tool install vingobot
```

### Configuration

Set up your LLM provider in `~/.vingobot/config.json`:

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "provider": "openrouter"
    }
  }
}
```

Or run the interactive setup wizard:

```bash
vingobot onboard --wizard
```

### Start Chatting

```bash
vingobot agent
```

---

## CLI Usage

| Command | Description |
|---------|-------------|
| `vingobot agent` | Interactive CLI chat mode |
| `vingobot onboard` | Initialize or reconfigure |
| `vingobot gateway` | Start gateway (channel + API server) |
| `vingobot status` | Show runtime status |
| `vingobot cron` | Manage scheduled tasks |
| `vingobot liuyao goal list` | List all liuyao goals |
| `vingobot liuyao queue list` | View task queue |
| `vingobot channels login <name>` | Login to a chat channel |
| `vingobot channels list` | List configured channels |

---

## Multi-Channel Messaging

Vingobot can run as a **unified bot** across multiple platforms simultaneously. Configure channels in `~/.vingobot/config.json`:

```json
{
  "channels": {
    "telegram": { "enabled": true, "token": "..." },
    "discord": { "enabled": true, "token": "..." },
    "feishu": { "enabled": true, "appId": "...", "appSecret": "..." }
  }
}
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

## Liuyao Mode (Goal-Driven Autonomous Execution)

The **六爻 (Liuyao)** mode enables goal-driven autonomous task execution through a coroutine worker pool:

```python
from vingobot import vingobot

bot = vingobot.from_config()
pool = await bot.start_sixiang(workers=3)
# ... pool runs autonomously ...
await bot.stop_sixiang()
```

### Cognition Hierarchy

The 六爻 system maintains a five-layer cognition workspace:

```
~/.vingobot/.taiji/
├── pending/              ← Task queue (atomic consumption)
├── cognition/
│   ├── skills/           ← L1 Skill library
│   ├── models/           ← L2 Experience models
│   ├── grids/            ← L3 Cognition grids (including meta-grids: 四象, 六爻, 八卦)
│   ├── truths/           ← L4 Immutable truths
│   └── soul/             ← L5 Self-model (identity, narrative, temporal memory)
└── goals/
    └── default/          ← Default dialogue target
```

L5 Soul (灵魂) stores the agent's self-model, including identity foundation, self-cognition, social others model, and temporal narrative.

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

    # Liuyao mode (goal-driven)
    pool = await bot.start_sixiang(workers=3)
    # ... let it run ...
    await bot.stop_sixiang()

asyncio.run(main())
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
│   ├── goal/              ← 六爻 core: loop, tasks, cognition
│   ├── heartbeat/         ← Periodic heartbeat service
│   ├── providers/         ← LLM provider implementations
│   ├── security/          ← Security utilities
│   ├── session/           ← Session persistence
│   ├── skills/            ← Skill system
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
### Thinkers and Works

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
