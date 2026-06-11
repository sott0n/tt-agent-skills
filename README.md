# tt-agent-skills

Centralized repository for Agent configuration files for Tenstorrent Software projects.

## Supported Projects

- **tt-metal** - Tenstorrent Metal library
- **tt-forge** - Compiler for Tenstorrent hardware

## Structure

```
tt-agent-skills/
├── setup.sh              # Setup script
├── tt-metal/
│   ├── skills/           # Claude Code skills
│   ├── CLAUDE.md         # Project-specific instructions (optional)
│   └── settings.json     # Settings file (optional)
├── tt-forge/
│   ├── skills/           # Claude Code skills
│   └── CLAUDE.md         # Project-specific instructions
```

## Setup

```bash
git clone git@github.com:sott0n/tt-agent-skills.git
cd tt-agent-skills

./setup.sh tt-metal
./setup.sh tt-forge
```

The setup script will:
1. Link common skills globally to `~/.claude/skills/`
2. Register the DeepWiki MCP server globally (user scope), so repo docs are queryable in any directory
3. Search for the project under `$HOME` (max depth 2)
4. If not found, offer to clone it from GitHub
5. Create symlinks in the project's `.claude/` directory

Example output:
```
[tt-metal]
  Found: /home/you/workspace/tt-metal
  [DONE] skills: Linked -> /home/you/tt-claude/tt-metal/skills
```

## Skills

### common (global)

| Skill | Description |
|-------|-------------|
| `using-mgrep` | Semantic code search via mgrep CLI |
| `recovering-tt-hardware` | Recover wedged TT hardware (tt-smi reset → tt-flash reflash) |
| `analyzing-tt-profiles` | Front-end-agnostic profile analysis (CSV / NoC JSON) |
| `querying-tt-deepwiki` | Query Tenstorrent repo docs via the DeepWiki MCP server |

### tt-metal

| Skill | Description |
|-------|-------------|
| `porting-models-to-ttnn` | Workflow for converting PyTorch models to TTNN |
| `optimizing-ttnn-models` | Performance optimization for TTNN models |
| `profiling-tt-metal` | Capture TTNN/TT-Metal profiles (Tracy, NoC, Memory Reports) |
| `tt-metal-perf-case-studies` | Worked end-to-end perf optimization case studies |
| `serving-ttnn-with-vllm` | Serve a TTNN model through vLLM (adapter, plugin, readiness runner) |

### tt-forge

| Skill | Description |
|-------|-------------|
| `tt-forge-bringup` | Bring up new models on tt-forge |
| `tt-forge-debug` | Debug compilation/execution errors |
| `tt-forge-test` | Run tests and validate accuracy |
| `tt-forge-perf` | Measure model performance |
| `tt-forge-optimize` | Implement performance optimizations |

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
