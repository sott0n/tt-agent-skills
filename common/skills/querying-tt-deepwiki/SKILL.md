---
name: querying-tt-deepwiki
description: "Queries Tenstorrent (TT) repository documentation via the DeepWiki MCP server (read_wiki_structure / read_wiki_contents / ask_question). Use when you need architectural overviews, API/usage explanations, or 'how does X work' answers about a TT repo (tt-metal, tt-mlir, tt-umd, tt-kmd, tt-isa-documentation, etc.) and reading the source directly would be slow or you lack a local checkout."
---

# Querying TT DeepWiki

[DeepWiki](https://deepwiki.com) auto-generates wiki-style documentation for public
GitHub repositories. The **DeepWiki MCP server** exposes that knowledge as tools so you
can ask grounded questions about a repo without cloning or grepping it.

The server is registered globally (user scope) by `setup.sh`:

```bash
claude mcp add -s user -t http deepwiki https://mcp.deepwiki.com/mcp
```

It is a **single server**, not one entry per repo. You pick the repo at query time via
the `repoName` parameter (`owner/repo`). No authentication is required (public repos).

## Tools

| Tool | Use for | Key params |
|------|---------|------------|
| `read_wiki_structure` | List the documentation topics/sections available for a repo | `repoName` |
| `read_wiki_contents` | Read the generated wiki pages (architecture, subsystems) | `repoName` |
| `ask_question` | Get a context-grounded answer to a specific natural-language question | `repoName`, `question` |

Typical flow: `read_wiki_structure` to see what exists → `ask_question` for a targeted
answer, or `read_wiki_contents` to read a section in full.

## Target Tenstorrent repositories

Pass these as `repoName` (all under the `tenstorrent` org):

| `repoName` | What it covers |
|------------|----------------|
| `tenstorrent/tt-metal` | Low-level programming model + TT-NN neural network library |
| `tenstorrent/tt-mlir` | MLIR-based compiler middle layer for tt-forge |
| `tenstorrent/tt-xla` | JAX/XLA → tt-mlir frontend |
| `tenstorrent/tt-onnx-fe` | ONNX → tt-mlir frontend |
| `tenstorrent/tt-forge-models` | Model zoo / bringup models for tt-forge |
| `tenstorrent/tt-inference-server` | Model serving / inference server |
| `tenstorrent/tt-umd` | User-mode driver (host ↔ device) |
| `tenstorrent/tt-kmd` | Kernel-mode driver |
| `tenstorrent/tt-system-firmware` | Device system firmware |
| `tenstorrent/tt-isa-documentation` | Tensix ISA / hardware architecture reference |

## When to use vs. read source directly

- **Use DeepWiki** for high-level "how does X work / where does Y live / what is the
  architecture of Z" questions, or when you have no local checkout of that repo.
- **Read the source** (local files, `grep`, `mgrep`) when you need exact current code,
  line numbers, or to make edits — DeepWiki reflects the indexed snapshot, which may lag
  the latest commits, so verify specifics against the actual tree before relying on them.

## Example

> "How does tt-metal dispatch work?"

1. `ask_question` with `repoName: "tenstorrent/tt-metal"`, `question: "How does the
   command/dispatch path work from host to device?"`
2. If you need more depth, `read_wiki_structure` then `read_wiki_contents` for the
   dispatch/runtime section.
