---
name: using-codex
description: "Delegates tasks to OpenAI Codex CLI. Use when user wants to offload code review, implementation, or complex coding tasks to Codex agent."
---

# Using Codex CLI

Codex CLI is OpenAI's coding agent that runs in your terminal. Use it to delegate code review, implementation, or complex coding tasks when you want a second AI perspective or parallel work.

## Table of Contents

- [Prerequisites Check](#prerequisites-check-run-first)
- [Delegating Tasks to Codex](#delegating-tasks-to-codex)
- [Workflow: Claude Code + Codex Collaboration](#workflow-claude-code--codex-collaboration)
- [Command Reference](#command-reference)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Prerequisites Check (Run First)

### Step 1: Check Installation

```bash
which codex
```

### Step 2: Install if Missing

**macOS (Homebrew):**
```bash
brew install --cask codex
```

**npm (Cross-platform):**
```bash
npm install -g @openai/codex
```

### Step 3: Check Authentication

```bash
codex --version
```

If not authenticated, user will be prompted to sign in with ChatGPT account or API key on first run.

---

## Delegating Tasks to Codex

### Code Review

Request Codex to review specific files or changes:

```bash
# Review a specific file
codex "Review the code in src/module.py for bugs, security issues, and best practices"

# Review recent changes
codex "Review the changes in the last commit and suggest improvements"

# Review a PR diff
git diff main...HEAD > /tmp/changes.diff && codex "Review this diff for issues: $(cat /tmp/changes.diff)"
```

### Implementation Tasks

Delegate implementation work to Codex:

```bash
# Implement a feature
codex "Implement a function that validates email addresses in utils/validators.py"

# Fix a bug
codex "Fix the null pointer exception in src/parser.rs line 42"

# Refactor code
codex "Refactor the UserService class to use dependency injection"
```

### Full Auto Mode

For trusted, well-defined tasks, use full auto mode:

```bash
codex --full-auto "Add unit tests for the Calculator class"
```

**Warning:** Full auto mode executes without confirmation. Use only for safe, reversible operations.

---

## Workflow: Claude Code + Codex Collaboration

### Pattern 1: Parallel Work

1. Claude Code works on feature A
2. Delegate feature B to Codex:
   ```bash
   codex "Implement feature B while I work on feature A"
   ```
3. Review Codex's changes when complete

### Pattern 2: Second Opinion Review

1. After implementing a feature in Claude Code
2. Request Codex review:
   ```bash
   codex "Review my implementation in src/feature.py - check for edge cases and improvements"
   ```
3. Incorporate feedback

### Pattern 3: Complex Refactoring

1. Plan the refactoring in Claude Code
2. Execute with Codex:
   ```bash
   codex "Refactor according to this plan: [paste plan]"
   ```
3. Verify results

---

## Command Reference

| Command | Description |
|---------|-------------|
| `codex` | Start interactive TUI session |
| `codex "prompt"` | Execute single task |
| `codex --full-auto "prompt"` | Execute without confirmations |
| `codex --model <model>` | Use specific model (default: gpt-5.3-codex) |
| `codex resume` | Resume previous session |
| `codex --help` | Show all options |

## Configuration

Config file location: `~/.config/codex/config.toml`

```toml
# Example config
model = "gpt-5.3-codex"
```

---

## Best Practices

1. **Be specific** - Provide clear, detailed prompts with file paths and context
2. **Review changes** - Always review Codex's output before committing
3. **Use interactive mode** - Prefer interactive TUI for complex tasks
4. **Isolate work** - Use separate branches when delegating large changes
5. **Verify tests** - Run tests after Codex makes changes

## Troubleshooting

### Authentication Issues

```bash
# Re-authenticate
codex logout
codex  # Will prompt for login
```

### Session Recovery

If Codex session was interrupted:

```bash
codex resume
```

### Rate Limits

Codex is included with ChatGPT Plus/Pro/Business/Enterprise plans. Check your plan limits if encountering rate errors.
