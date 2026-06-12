#!/bin/bash
# Setup script for tt-claude
# Creates symlinks from each project's .claude directory to tt-claude configs
#
# Usage:
#   ./setup.sh <project>          Setup specific project (tt-metal, tt-forge)
#   ./setup.sh <project> <path>   Link <project> config to an explicit repo path
#                                 (skips auto-discovery; handy for worktrees/forks
#                                  with a non-standard directory name)
#   ./setup.sh common             Link common skills globally to ~/.claude/skills only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Available projects (plus the special "common" target)
PROJECTS="tt-metal tt-forge common"

# tt-forge links to multiple repositories
TT_FORGE_REPOS="tt-forge-models tt-xla tt-onnx-fe tt-mlir"

# Common skills are linked globally so they work in any directory
GLOBAL_SKILLS_DIR="$HOME/.claude/skills"

# DeepWiki MCP server (https://deepwiki.com) — queried per-repo via repoName
# (e.g. tenstorrent/tt-metal). Registered once at user scope so it is available
# in any directory, matching the "common skills -> global" philosophy.
DEEPWIKI_MCP_NAME="deepwiki"
DEEPWIKI_MCP_URL="https://mcp.deepwiki.com/mcp"

# Find project directory under HOME (max depth 2)
find_project() {
    local name=$1
    local found=""

    # Search HOME with max depth 2, excluding this repo
    found=$(find "$HOME" -maxdepth 2 -type d -name "$name" -not -path "$SCRIPT_DIR/*" 2>/dev/null | head -n 1)

    echo "$found"
}

# Link individual skills from a source directory
link_skills() {
    local source_skills_dir=$1
    local target_skills_dir=$2
    local label=$3

    if [[ ! -d "$source_skills_dir" ]]; then
        return
    fi

    for skill_dir in "$source_skills_dir"/*/; do
        if [[ -d "$skill_dir" ]]; then
            local skill_name
            skill_name=$(basename "$skill_dir")
            local target="$target_skills_dir/$skill_name"

            if [[ -L "$target" ]]; then
                echo "  [OK]   $label/$skill_name"
            elif [[ -d "$target" ]]; then
                echo "  [WARN] $label/$skill_name: Directory exists (not a symlink)"
            else
                ln -s "$skill_dir" "$target"
                echo "  [DONE] $label/$skill_name"
            fi
        fi
    done
}

# Link common skills globally to ~/.claude/skills
link_common_global() {
    local common_dir="$SCRIPT_DIR/common"

    echo "[common -> global]"
    mkdir -p "$GLOBAL_SKILLS_DIR"
    link_skills "$common_dir/skills" "$GLOBAL_SKILLS_DIR" "skills(common)"
    echo ""
    echo "  Linked to: $GLOBAL_SKILLS_DIR"
    echo ""
}

# Register the DeepWiki MCP server globally (user scope), idempotently.
register_deepwiki_mcp() {
    echo "[deepwiki MCP -> global]"

    if ! command -v claude >/dev/null 2>&1; then
        echo "  [SKIP] 'claude' CLI not on PATH; register manually with:"
        echo "         claude mcp add -s user -t http $DEEPWIKI_MCP_NAME $DEEPWIKI_MCP_URL"
        echo ""
        return
    fi

    if claude mcp get "$DEEPWIKI_MCP_NAME" >/dev/null 2>&1; then
        echo "  [OK]   $DEEPWIKI_MCP_NAME already registered"
    elif claude mcp add -s user -t http "$DEEPWIKI_MCP_NAME" "$DEEPWIKI_MCP_URL" >/dev/null 2>&1; then
        echo "  [DONE] $DEEPWIKI_MCP_NAME -> $DEEPWIKI_MCP_URL (user scope)"
    else
        echo "  [WARN] failed to register $DEEPWIKI_MCP_NAME; add manually with:"
        echo "         claude mcp add -s user -t http $DEEPWIKI_MCP_NAME $DEEPWIKI_MCP_URL"
    fi
    echo ""
}

# Remove common skill symlinks left in a project's .claude/skills by old setups
# (common skills now live in the global ~/.claude/skills instead).
cleanup_project_common() {
    local skills_dir=$1
    local common_skills_dir="$SCRIPT_DIR/common/skills"

    [[ -d "$common_skills_dir" ]] || return

    for skill_dir in "$common_skills_dir"/*/; do
        [[ -d "$skill_dir" ]] || continue
        local skill_name
        skill_name=$(basename "$skill_dir")
        local stale="$skills_dir/$skill_name"
        if [[ -L "$stale" ]]; then
            rm "$stale"
            echo "  [CLEAN] skills/$skill_name (now global)"
        fi
    done
}

# Link project configs
link_project() {
    local name=$1
    local project_path=$2
    local source_dir="$SCRIPT_DIR/$name"

    local claude_dir="$project_path/.claude"
    mkdir -p "$claude_dir"

    # Create skills directory (not a symlink, to hold both project and common skills)
    local skills_dir="$claude_dir/skills"
    if [[ -L "$skills_dir" ]]; then
        echo "  [WARN] skills: Is a symlink from old setup. Removing to use new structure."
        rm "$skills_dir"
    fi
    mkdir -p "$skills_dir"

    # Link project-specific skills (if project dir exists)
    if [[ -d "$source_dir" ]]; then
        link_skills "$source_dir/skills" "$skills_dir" "skills"
    fi

    # Common skills are linked globally (see link_common_global), not per-project.
    # Remove any stale per-project common links left by older setups.
    cleanup_project_common "$skills_dir"

    # Link CLAUDE.md if exists
    if [[ -f "$source_dir/CLAUDE.md" ]]; then
        local target="$claude_dir/CLAUDE.md"
        if [[ -L "$target" ]]; then
            echo "  [OK]   CLAUDE.md"
        elif [[ -f "$target" ]]; then
            echo "  [WARN] CLAUDE.md: File exists (not a symlink)"
        else
            ln -s "$source_dir/CLAUDE.md" "$target"
            echo "  [DONE] CLAUDE.md"
        fi
    fi

    # Link settings.json if exists
    if [[ -f "$source_dir/settings.json" ]]; then
        local target="$claude_dir/settings.json"
        if [[ -L "$target" ]]; then
            echo "  [OK]   settings.json"
        elif [[ -f "$target" ]]; then
            echo "  [WARN] settings.json: File exists (not a symlink)"
        else
            ln -s "$source_dir/settings.json" "$target"
            echo "  [DONE] settings.json"
        fi
    fi

    # Show summary
    echo ""
    echo "  Linked to: $claude_dir"
}

# Setup a single repository
setup_repo() {
    local config_name=$1    # Config directory name (tt-metal or tt-forge)
    local repo_name=$2      # Target repository name
    local explicit_path=$3  # Optional: explicit repo path (skips auto-discovery)

    echo "[$repo_name]"

    # Resolve project path: explicit path overrides auto-discovery
    local project_path
    if [[ -n "$explicit_path" ]]; then
        if [[ ! -d "$explicit_path" ]]; then
            echo "  [SKIP] Path not found: $explicit_path"
            return
        fi
        project_path="$explicit_path"
    else
        project_path=$(find_project "$repo_name")
        if [[ -z "$project_path" ]]; then
            echo "  [SKIP] Not found under \$HOME (depth 2)"
            return
        fi
    fi

    echo "  Found: $project_path"
    link_project "$config_name" "$project_path"
}

# Setup a project (may link to multiple repos)
setup_project() {
    local name=$1
    local explicit_path=$2

    if [[ "$name" == "common" ]]; then
        # Global common skills only; no per-project linking
        return
    fi

    # Explicit path overrides auto-discovery; link this config to that single dir
    # (works for any repo name, e.g. a worktree like tt-metal-perf-vadv2).
    if [[ -n "$explicit_path" ]]; then
        setup_repo "$name" "$(basename "$explicit_path")" "$explicit_path"
        return
    fi

    if [[ "$name" == "tt-forge" ]]; then
        # tt-forge links to multiple repositories
        for repo in $TT_FORGE_REPOS; do
            setup_repo "$name" "$repo"
            echo ""
        done
    else
        setup_repo "$name" "$name"
    fi
}

show_help() {
    echo "tt-claude Setup"
    echo ""
    echo "Usage: ./setup.sh <project> [path]"
    echo ""
    echo "Available projects:"
    echo "  - tt-metal"
    echo "  - tt-forge (links to: $TT_FORGE_REPOS)"
    echo "  - common   (link common skills globally to $GLOBAL_SKILLS_DIR only)"
    echo ""
    echo "Optional [path]: link <project> config to an explicit repo directory,"
    echo "skipping auto-discovery (useful for worktrees / non-standard names)."
    echo ""
    echo "Common skills are always linked globally to $GLOBAL_SKILLS_DIR."
    echo "The DeepWiki MCP server is always registered globally (user scope)."
    echo "Project-specific skills, CLAUDE.md and settings.json are linked per repo."
    echo ""
    echo "Example:"
    echo "  ./setup.sh tt-metal"
    echo "  ./setup.sh tt-metal /path/to/repo"
    echo "  ./setup.sh tt-forge"
    echo "  ./setup.sh common"
}

# Check if project is valid
is_valid_project() {
    local name=$1
    for p in $PROJECTS; do
        if [[ "$p" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

# Main
if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

if ! is_valid_project "$1"; then
    echo "Error: Unknown project '$1'"
    echo ""
    show_help
    exit 1
fi

echo "tt-claude Setup"
echo "==============="
echo ""

# Common skills are always linked globally (works in any directory)
link_common_global

# DeepWiki MCP is always registered globally (works in any directory)
register_deepwiki_mcp

setup_project "$1" "$2"

echo ""
echo "Setup complete."
