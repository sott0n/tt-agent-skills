---
name: using-mgrep
description: "Sets up and uses mgrep for semantic code search. Use when user wants to install mgrep plugin or perform natural language search across codebase."
---

# Using mgrep

mgrep is a semantic search CLI tool by Mixedbread that enables natural language search across codebases. Unlike traditional `grep`, mgrep understands intent and works with code, images, PDFs, and text files.

## Installation

### Step 1: Install mgrep CLI

```bash
npm install -g @mixedbread/mgrep
```

Alternatively, use `pnpm` or `bun`:

```bash
pnpm add -g @mixedbread/mgrep
# or
bun add -g @mixedbread/mgrep
```

### Step 2: Authenticate with Mixedbread

```bash
mgrep login
```

A browser window opens for authentication. For CI/CD environments, set `MXBAI_API_KEY` environment variable instead.

### Step 3: Install Claude Code Plugin

```bash
mgrep install-claude-code
```

This adds the mgrep plugin to Claude Code's marketplace.

### Step 4: Restart Claude Code

Restart your Claude Code session to load the new plugin.

---

## Project Setup

For each project you want to search, index it with:

```bash
cd /path/to/project
mgrep watch
```

This performs initial sync and keeps the index updated as files change. It respects `.gitignore` patterns.

---

## Search Commands

### Basic Search

```bash
mgrep "your natural language query"
```

### Common Options

| Option | Description |
|--------|-------------|
| `-m <count>` | Limit number of results (e.g., `-m 25`) |
| `-a, --answer` | Generate AI summary from results |
| `-w, --web` | Include web search alongside local files |
| `--store <name>` | Use named store to isolate workspaces |

### Examples

```bash
# Find authentication setup
mgrep "where do we set up auth?"

# Find schema definitions with limit
mgrep -m 25 "store schema"

# Get AI-generated answer
mgrep -a "how does error handling work?"

# Search with web results included
mgrep -w "best practices for caching"
```

---

## Configuration

Configure via CLI flags, environment variables, or config files.

### Config File Locations

- **Local:** `.mgreprc.yaml` in project root
- **Global:** `~/.config/mgrep/config.yaml`

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `max-file-size` | 1MB | Maximum file size to upload |
| `max-file-count` | 1000 | Maximum files per directory |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MXBAI_API_KEY` | API key for CI/CD (skip interactive login) |
| `MGREP_MAX_COUNT` | Default result limit |
| `MGREP_CONTENT` | Default content mode |
| `MGREP_RERANK` | Enable/disable reranking |

---

## Troubleshooting

### Login Issues

If login doesn't persist, re-run:

```bash
mgrep login
```

### Plugin Not Loading

1. Verify plugin is installed: `mgrep install-claude-code`
2. Restart Claude Code session
3. Check Claude Code settings for mgrep in mcpServers

### Index Not Updating

Ensure `mgrep watch` is running in the project directory, or re-sync:

```bash
mgrep sync
```
