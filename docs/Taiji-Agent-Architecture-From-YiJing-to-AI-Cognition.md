# ☯ Taiji Agent Architecture: From the I Ching to AI Cognition

> An open-source experiment weaving ancient Chinese philosophy into AI Agent architecture

---

## The Problem: Why Do All Agent Frameworks Look the Same?

If you've used AutoGPT, CrewAI, LangGraph, or OpenAI Agents SDK, you've noticed the pattern:

1. You write a fixed system prompt
2. You register a fixed set of tools
3. The Agent loops through the same execution cycle every time

**The blind spot**: they all assume a *fixed cognitive structure*.

- Fixed system prompt — same behavior guidelines every execution
- Fixed toolset — same tools regardless of the task
- Fixed knowledge base — static documents that never evolve

This is like giving someone a permanent personality and expecting them to handle every situation perfectly.

**Taiji** is different. It treats an AI Agent not as a tool with fixed functionality, but as a **continuously evolving cognitive being** — every round, its system prompt, toolset, and cognitive strategy are dynamically rewoven.

---

## Chapter 1: Emptiness (空) — The First Principle

### 空 ≠ Nothing, 空 = Undetermined Possibility

In Taiji architecture, "Emptiness" (空) is the highest principle. It's close to **structural indeterminacy** — the loom's threads can be changed every round.

**Three dimensions of Emptiness:**

| Dimension | Traditional Approach | Taiji's Emptiness |
|-----------|-------------------|-------------------|
| **System Prompt** | Fixed, never changes | **Dynamically woven** every round by Weaver from the Bagua grid |
| **Toolset** | Hardcoded at startup | **Dynamically discovered** from Bagua trigrams each round |
| **Knowledge Base** | Static encyclopedia | **Continuously evolves** through L1-L5 layers, always open to new experience |

---

## Chapter 2: Weaver (编织器) — The Core Orchestrator

If Taiji is a person, Weaver is the **weaver at the loom**. Every round it does four things:

1. **Weave cognitive strategy** — Read L3 grid (Bagua), determine the cognitive posture (trigram), decide the mode (create/execute/launch/analyze/etc.)
2. **Weave tools** — Discover associated L1 skills and L2 models from the Bagua grid, dynamically bind available tools
3. **Weave execution guidance** — Inject trigram workflow steps and philosophy into the system prompt
4. **Weave cross-round memory** — Incorporate previous round results into the new context, forming a continuous cognitive flow

### Simplified Weaver Logic

```
function weaver_weave(round_context):
    trigram = read_trigram_from_grid(round_context.task_type)
    yao = determine_yao(trigram, round_context.round_number)
    tools = discover_tools(trigram.gua, round_context.goal)
    system_prompt = compose_system_prompt(trigram, yao, tools)
    memory = load_memory(round_context.goal_id)
    full_context = weave_memory(system_prompt, memory)
    return full_context, tools
```

---

## Chapter 3: The Six-Yao Cycle (六爻循环)

Taiji's frontend (TPN — Task Positive Network) consists of six Yao (爻), forming a complete cognitive loop:

```
Mingjue (Perception) → Weaver (Weaving) → Yang (Intention)
  → Yin (Approval) → Executor (Execution) → Anqu (Reflection)
  → Sixiang (Decision)
```

- **Mingjue** — Perceive the current state and available resources
- **Weaver** — Weave scattered information into executable strategy
- **Yang** — Propose execution intent (what to do)
- **Yin** — Approver role, checks intent with safety rules and ethics (dual-layer: hardcoded + LLM semantic)
- **Executor** — Execute approved actions (call tools, write files, run commands)
- **Anqu** — Reflect on results, record lessons, extract patterns → triggers DMN cognitive evolution
- **Sixiang** — Decide next step: continue, switch strategy, or complete

### DMN (Background Cognitive Evolution)

Anqu analyzes task logs → extracts patterns → triggers evolution → updates L1~L4 cognitive libraries → Weaver perceives grid changes → behavior adjusts

---

## Chapter 4: The Bagua Cognitive Grid (八卦认知网格)

Taiji defines eight cognitive postures mapped to the Bagua:

| Trigram | Posture | Element | Lifecycle | Use Case |
|---------|---------|---------|-----------|----------|
| ☰ **Qian** | Create/Pioneer | Metal | Birth | Brainstorming, architecture design |
| ☷ **Kun** | Execute/Carry | Earth | Growth | Steady execution, coding, testing |
| ☳ **Zhen** | Launch/Inspire | Wood | Nurture | Quick prototype, bootstrap |
| ☴ **Xun** | Penetrate/Analyze | Wood | Transform | Code review, performance analysis |
| ☵ **Kan** | Breakthrough | Water | Harvest | Solve tough problems, fix bugs |
| ☲ **Li** | Illuminate/Evaluate | Fire | Store | Quality check, retrospective |
| ☶ **Gen** | Focus/Stabilize | Earth | Settle | Stop divergence, stabilize |
| ☱ **Dui** | Express/Output | Metal | Recover | Documentation, reports, release |

The eight postures interact through the **Five Elements (五行) system** — generating (生) and restraining (克) relationships that Weaver uses to predict workflow conflicts and synergies.

---

## Chapter 5: L1-L5 Cognitive Library

| Level | Content | Description |
|-------|---------|-------------|
| **L5** | SOUL/AGENTS/TOOLS/USER/MEMORY | Self-awareness, identity, long-term memory |
| **L4** | truths/ | Immutable principles from experience |
| **L3** | grids/ | Bagua thinking grids with dynamic weights |
| **L2** | models/ | Trusted workflow patterns |
| **L1** | skills/ | Installable tools and capabilities |

---

## Chapter 6: Taiji vs Mainstream Frameworks

### vs OpenClaw (257K+ GitHub Stars)

| Dimension | OpenClaw | Taiji |
|-----------|----------|-------|
| Philosophy | 24/7 life assistant, 1000+ skills marketplace | Cognitive evolution engine |
| Learning | ❌ No native learning layer | ✅ Yin-Yang cycle for continuous self-iteration |
| Security | ❌ CVSS 8.8 RCE, prompt injection | ✅ Dual-layer approval (hardcoded + LLM) |
| Cognitive arch | ❌ Toolchain orchestrator only | ✅ TPN+DMN dual-mode, L1-L5 knowledge layers |

### vs Hermes Agent (Nous Research)

| Dimension | Hermes | Taiji |
|-----------|--------|-------|
| Skill learning | "What worked" (experience memory) | "Why this grid fits" (structural understanding) |
| Modes | Single execution channel | Dual mode (DMN + TPN) |
| Skill origin | Post-hoc extraction | Architectural "weave as you run" |
| Philosophy | Learn from past | Weave from nothing (start from emptiness) |

### One-liners

- **OpenClaw** = Swiss Army knife (many tools, never gets better)
- **Hermes** = Diligent learner (accumulates experience, relies on history)
- **Taiji** = Meditating monk (starts from emptiness, weaves cognition anew each round)

---

## Quick Start

```bash
git clone https://github.com/wenge6090-cell/Taiji.git
cd Taiji
pip install -e .
taiji run --goal "Analyze this project's code structure"
```

**Zero cost**: Pure Python CLI, no paid API dependencies.

---

## Current Status

- ✅ Six-Yao cognitive cycle implemented
- ✅ Bagua cognitive grid with Five Elements interactions
- ✅ Weaver dynamic weaving engine
- ✅ L1-L5 cognitive library layers
- ✅ Dual-layer safety approval
- 🚧 English documentation & community outreach
- 🚧 More usage examples and tutorials

---

**Repository**: https://github.com/wenge6090-cell/Taiji
**License**: Open source

*Ideal for: AI/ML researchers exploring cognitive architectures, developers interested in non-conventional Agent design, teams wanting cognitive flexibility + self-evolution in their Agents.*
