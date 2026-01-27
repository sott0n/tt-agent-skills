# tt-agent-skills

Centralized repository for Agent configuration files for Tenstorrent Software projects.

## Supported Projects

- **tt-metal** - Tenstorrent Metal library

## Structure

```
tt-claude/
├── setup.sh              # Setup script
├── tt-metal/
│   ├── skills/           # Claude Code skills
│   ├── CLAUDE.md         # Project-specific instructions (optional)
│   └── settings.json     # Settings file (optional)
```

## Setup

```bash
git clone <repository-url> ~/workspace/tt-claude
cd ~/workspace/tt-claude

./setup.sh tt-metal
```

The setup script will:
1. Search for the project under `$HOME` (max depth 2)
2. If not found, offer to clone it from GitHub
3. Create symlinks in the project's `.claude/` directory

Example output:
```
[tt-metal]
  Found: /Users/you/workspace/tt-metal
  [DONE] skills: Linked -> /Users/you/tt-claude/tt-metal/skills
```

## Skills

### tt-metal

| Skill | Description |
|-------|-------------|
| `porting-models-to-ttnn` | Workflow for converting PyTorch models to TTNN |
| `optimizing-ttnn-models` | Performance optimization for TTNN models |

## Adding New Skills

1. Create a skill directory for the target project:
   ```bash
   mkdir -p tt-metal/skills/new-skill
   ```

2. Create `SKILL.md` (required):
   ```markdown
   ---
   name: new-skill
   description: Skill description here
   ---

   # Skill Name

   Instructions for Claude...
   ```

3. Re-run `./setup.sh` to update links

## Reference

- [Claude Code Skills Documentation](https://docs.anthropic.com/en/docs/claude-code/skills)
