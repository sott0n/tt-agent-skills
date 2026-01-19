# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tt-claude is a centralized repository for Claude Code configuration files for Tenstorrent Software projects. It provides skills, settings, and CLAUDE.md files that are symlinked into target project directories.

## Supported Projects

- **tt-metal** - Tenstorrent Metal library (low-level programming model + TT-NN neural network library)
- **tt-mlir** - Tenstorrent MLIR compiler
- **tt-xla** - Tenstorrent XLA integration

## Setup

```bash
./setup.sh tt-metal   # Setup for tt-metal project
./setup.sh tt-mlir    # Setup for tt-mlir project
./setup.sh tt-xla     # Setup for tt-xla project
```

The setup script:
1. Searches for the project under `$HOME` (max depth 2)
2. If not found, offers to clone from GitHub
3. Creates symlinks in the project's `.claude/` directory (skills, CLAUDE.md, settings.json)

## Repository Structure

```
tt-claude/
├── setup.sh              # Setup script for linking configs
├── .claude/
│   ├── agents/           # Custom subagent definitions
│   └── settings.local.json
├── common/
│   └── skills/           # Skills shared across all projects
│       ├── using-github/ # GitHub operations via gh CLI
│       └── using-mgrep/  # Semantic search via mgrep CLI
├── tt-metal/
│   ├── CLAUDE.md         # Project-specific Claude instructions
│   └── skills/
│       ├── porting-models-to-ttnn/    # 7-step model bringup workflow
│       └── optimizing-ttnn-models/    # Performance optimization workflow
├── tt-mlir/              # (configs to be added)
└── tt-xla/               # (configs to be added)
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

Skills in `common/skills/` are shared across all projects.

| Skill | Description |
|-------|-------------|
| `using-github` | GitHub operations via gh CLI (PRs, commits, issues, git blame) |
| `using-mgrep` | Semantic code search via mgrep CLI (natural language queries) |

## tt-metal Skills

| Skill | Description |
|-------|-------------|
| `porting-models-to-ttnn` | 7-step workflow for converting PyTorch models to TTNN |
| `optimizing-ttnn-models` | Performance optimization (data formats, sharding, Metal Trace, multi-device) |
