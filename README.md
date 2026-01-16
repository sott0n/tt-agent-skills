# tt-claude

Centralized repository for Claude Code configuration files for Tenstorrent Software projects.

## Supported Projects

- **tt-metal** - Tenstorrent Metal library
- **tt-mlir** - Tenstorrent MLIR compiler
- **tt-xla** - Tenstorrent XLA integration

## Structure

```
tt-claude/
├── setup.sh              # Setup script
├── tt-metal/
│   ├── skills/           # Claude Code skills
│   ├── CLAUDE.md         # Project-specific instructions (optional)
│   └── settings.json     # Settings file (optional)
├── tt-mlir/
│   └── ...
└── tt-xla/
    └── ...
```

## Setup

```bash
git clone <repository-url> ~/workspace/tt-claude
cd ~/workspace/tt-claude
./setup.sh
```

The setup script creates symlinks in each project's `.claude/` directory:

```
~/workspace/tt-metal/.claude/skills -> ~/workspace/tt-claude/tt-metal/skills
~/workspace/tt-mlir/.claude/skills  -> ~/workspace/tt-claude/tt-mlir/skills
~/workspace/tt-xla/.claude/skills   -> ~/workspace/tt-claude/tt-xla/skills
```

## Skills

### tt-metal

| Skill | Description |
|-------|-------------|
| `ttnn-model-bringup` | Workflow for converting PyTorch models to TTNN |
| `ttnn-model-optimization` | Performance optimization for TTNN models |

## Customizing Project Paths

The default project paths assume `$HOME/workspace/`. If your projects are located elsewhere, edit the `PROJECTS` array in `setup.sh`:

```bash
PROJECTS=(
    "tt-metal:/path/to/your/tt-metal"
    "tt-mlir:/path/to/your/tt-mlir"
    "tt-xla:/path/to/your/tt-xla"
)
```

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
