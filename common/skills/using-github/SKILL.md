---
name: using-github
description: "Operates GitHub via gh CLI. Use when user wants to view PRs, commits, issues, or perform git blame."
---

# Using GitHub

GitHub operations are performed using the `gh` CLI tool.

## Prerequisites Check (Run First)

Before any GitHub operation, verify gh CLI is installed and authenticated.

### Step 1: Check Installation

```bash
which gh
```

### Step 2: Install if Missing

Detect OS and install accordingly:

```bash
# Check OS
uname -s
```

**macOS (Homebrew):**
```bash
brew install gh
```

**Ubuntu/Debian:**
```bash
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update \
  && sudo apt install gh -y
```

### Step 3: Check Authentication

```bash
gh auth status
```

### Step 4: Authenticate if Needed

If not authenticated, prompt user to run:

```bash
gh auth login
```

This is interactive - let the user complete it manually.

---

## Common Operations

### Pull Requests

| Command | Description |
|---------|-------------|
| `gh pr list` | List open PRs |
| `gh pr view <number>` | View PR details |
| `gh pr diff <number>` | Show PR diff |
| `gh pr checks <number>` | View CI status |
| `gh pr comments <number>` | View PR comments |
| `gh api repos/{owner}/{repo}/pulls/{number}/comments` | Get PR review comments via API |

### Commits

| Command | Description |
|---------|-------------|
| `git log --oneline -n 20` | Recent commits |
| `gh api repos/{owner}/{repo}/commits/{sha}` | Commit details via API |
| `gh browse <sha>` | Open commit in browser |

### Git Blame

| Command | Description |
|---------|-------------|
| `git blame <file>` | Show line-by-line authorship |
| `git blame -L 10,20 <file>` | Blame specific line range |
| `git blame -L :functionName <file>` | Blame specific function |
| `git log --follow -p <file>` | Full file history with diffs |

### Issues

| Command | Description |
|---------|-------------|
| `gh issue list` | List open issues |
| `gh issue view <number>` | View issue details |
| `gh issue create` | Create new issue |

### Repository Info

| Command | Description |
|---------|-------------|
| `gh repo view` | View repo details |
| `gh api repos/{owner}/{repo}` | Repo info via API |
