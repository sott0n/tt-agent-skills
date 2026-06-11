# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tt-claude is a centralized repository for Claude Code configuration files for Tenstorrent Software projects. It provides skills, settings, and CLAUDE.md files that are symlinked into target project directories.

## Supported Projects

- **tt-metal** - Tenstorrent Metal library (low-level programming model + TT-NN neural network library)
- **tt-forge** - DNN compiler for Tenstorrent hardware (tt-xla/tt-onnx-fe frontend → tt-mlir optimization → tt-metal runtime)

## Setup

```bash
./setup.sh tt-metal   # Setup for tt-metal project
./setup.sh tt-forge   # Setup for tt-forge repos (tt-forge-models, tt-xla, tt-onnx-fe, tt-mlir)
./setup.sh common     # Link common skills globally only (no per-project linking)
```

### Hybrid linking model

The setup uses a hybrid layout so common skills work everywhere while project-specific configs stay scoped to their repos:

- **Common skills → global** (`~/.claude/skills/`): `common/skills/*` are linked once into the user-level skills directory, so they are available in any directory regardless of project. Linked automatically on every `setup.sh` run.
- **Project-specific skills → per-repo** (`<repo>/.claude/skills/`): only the matching project's skills are linked, keeping the skill list focused on the current context.
- **CLAUDE.md / settings.json → per-repo** (`<repo>/.claude/`): these are project-specific and cannot be global.

The setup script:
1. Links `common/skills` globally to `~/.claude/skills/`
2. Registers the DeepWiki MCP server globally at user scope (idempotent; via `claude mcp add -s user`)
3. Searches for target repositories under `$HOME` (max depth 2)
4. If not found, offers to clone from GitHub
5. Creates symlinks in each repository's `.claude/` directory (project skills, CLAUDE.md, settings.json)
6. Removes any stale per-repo common skill symlinks left by older setups (common is now global)

### DeepWiki MCP (global)

Like common skills, the [DeepWiki](https://deepwiki.com) MCP server is registered once at
**user scope** (`~/.claude.json`) so it works in any directory. It is a single server
(`https://mcp.deepwiki.com/mcp`, no auth) queried per-repo via a `repoName` parameter
(e.g. `tenstorrent/tt-metal`) — there is no per-repo registration. The list of relevant
Tenstorrent repos and usage guidance lives in the `querying-tt-deepwiki` common skill.

Note: `tt-forge` links configs to all related repositories: tt-forge-models, tt-xla, tt-onnx-fe, tt-mlir

## Repository Structure

```
tt-agent-skills/
├── setup.sh              # Setup script for linking configs
├── .claude/
│   ├── agents/           # Custom subagent definitions
│   └── settings.local.json
├── common/
│   └── skills/           # Skills shared across all projects
│       ├── using-mgrep/  # Semantic search via mgrep CLI
│       ├── recovering-tt-hardware/ # HW reset + firmware reflash recovery
│       ├── analyzing-tt-profiles/  # Front-end-agnostic profile analysis (CSV/NoC JSON)
│       └── querying-tt-deepwiki/   # Query repo docs via DeepWiki MCP
├── tt-metal/
│   ├── CLAUDE.md         # Project-specific Claude instructions
│   └── skills/
│       ├── porting-models-to-ttnn/      # 7-step model bringup workflow
│       ├── optimizing-ttnn-models/      # Performance optimization workflow
│       ├── profiling-tt-metal/          # TTNN profile *capture* (tracy build, python -m tracy, memory SQLite)
│       ├── tt-metal-perf-case-studies/  # End-to-end perf optimization case studies
│       └── serving-ttnn-with-vllm/      # Serve a TTNN model through vLLM
└── tt-forge/
    ├── CLAUDE.md         # Project-specific Claude instructions
    └── skills/
        ├── tt-forge-bringup/   # Bring up new models
        ├── tt-forge-debug/     # Debug compilation/execution errors
        ├── tt-forge-test/      # Run tests and validate accuracy
        ├── tt-forge-perf/      # Measure model performance
        └── tt-forge-optimize/  # Implement performance optimizations
```

## Creating Skills

Skills are stored in `<project>/skills/<skill-name>/` with a required `SKILL.md` file.

### SKILL.md Format

```markdown
---
name: skill-name-in-gerund-form
description: "Third person description. Use when [context]."
---

# Skill Title

Instructions for Claude...
```

### Skill Best Practices

- **Name**: lowercase, hyphens, gerund form (e.g., `porting-models-to-ttnn`)
- **Description**: third person, includes "Use when..." context, max 1024 chars
- **Body**: max 500 lines; split into separate files if larger
- **References**: keep one level deep from SKILL.md
- **TOC**: add table of contents for files >100 lines

### Using skill-reviewer Agent

After creating or modifying SKILL.md files, use the `skill-reviewer` subagent to validate against Anthropic's best practices:

```
Review the skill at tt-metal/skills/my-skill/SKILL.md
```

## Common Skills

Skills in `common/skills/` are shared across all projects and are linked globally to `~/.claude/skills/`, so they are available in any directory.

| Skill | Description |
|-------|-------------|
| `using-mgrep` | Semantic code search via mgrep CLI (natural language queries) |
| `recovering-tt-hardware` | Recover wedged TT hardware: tt-smi reset → tt-flash firmware reflash fallback |
| `analyzing-tt-profiles` | Front-end-agnostic profile *analysis*: `ops_perf_results*.csv` columns, `tt-perf-report` CLI, Python recipes, NoC JSON, pitfalls. Same CSV from `python -m tracy` (TTNN) or `ttrt perf` (tt-forge) |
| `querying-tt-deepwiki` | Query Tenstorrent repo docs via the DeepWiki MCP server (`read_wiki_structure` / `read_wiki_contents` / `ask_question`, `repoName: tenstorrent/<repo>`). Lists the 10 target repos and when to use DeepWiki vs. reading source |

## tt-metal Skills

| Skill | Description |
|-------|-------------|
| `porting-models-to-ttnn` | 7-step workflow for converting PyTorch models to TTNN |
| `optimizing-ttnn-models` | Performance optimization (data formats, sharding, Metal Trace, multi-device). LLM-specific refs: `llm-decoder-optimization.md` (matmul/precision tuning), `tensor-parallel-llm.md` (TP/EP), `metal-trace-debugging.md` (program-cache warmup), `datatype-sweep.md` (accuracy-gated precision) |
| `profiling-tt-metal` | TTNN profile *capture*: Tracy build setup, `python -m tracy` (Performance/NoC), and TTNN Memory Reports (`full_graph_capture` → SQLite + SQL recipes). Analysis lives in `analyzing-tt-profiles` |
| `tt-metal-perf-case-studies` | Worked end-to-end perf optimization case studies (profile → bottleneck → fix → verify) |
| `serving-ttnn-with-vllm` | Serve a working TTNN model through vLLM: thin `generator_vllm.py` adapter, plugin registration, `run_vllm_server` readiness runner, async-decode / on-device-sampling serving optimization |

## tt-forge Skills

| Skill | Description |
|-------|-------------|
| `tt-forge-bringup` | Bring up new models on tt-forge |
| `tt-forge-debug` | Debug compilation/execution errors across the tt-xla, tt-mlir, tt-metal stack |
| `tt-forge-test` | Run tests and validate PCC/atol accuracy |
| `tt-forge-perf` | Measure model performance (latency, throughput, bottlenecks) |
| `tt-forge-optimize` | Implement performance optimizations in tt-mlir |
