#!/bin/bash
# Setup script for tt-claude
# Creates symlinks from each project's .claude directory to tt-claude configs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project configurations: PROJECT_NAME:PROJECT_PATH
# Add or modify projects here
PROJECTS=(
    "tt-metal:$HOME/workspace/tt-metal"
    "tt-mlir:$HOME/workspace/tt-mlir"
    "tt-xla:$HOME/workspace/tt-xla"
)

link_project() {
    local name=$1
    local project_path=$2
    local source_dir="$SCRIPT_DIR/$name"

    if [[ ! -d "$source_dir" ]]; then
        echo "  [SKIP] $name: No config in tt-claude ($source_dir)"
        return
    fi

    if [[ ! -d "$project_path" ]]; then
        echo "  [SKIP] $name: Project not found ($project_path)"
        return
    fi

    local claude_dir="$project_path/.claude"

    # Create .claude directory if it doesn't exist
    mkdir -p "$claude_dir"

    # Link skills directory
    if [[ -d "$source_dir/skills" ]]; then
        local target="$claude_dir/skills"
        if [[ -L "$target" ]]; then
            echo "  [OK]   $name/skills: Already linked"
        elif [[ -d "$target" ]]; then
            echo "  [WARN] $name/skills: Directory exists (not a symlink)"
            echo "         Remove $target manually if you want to link"
        else
            ln -s "$source_dir/skills" "$target"
            echo "  [DONE] $name/skills: Linked"
        fi
    fi

    # Link CLAUDE.md if exists
    if [[ -f "$source_dir/CLAUDE.md" ]]; then
        local target="$claude_dir/CLAUDE.md"
        if [[ -L "$target" ]]; then
            echo "  [OK]   $name/CLAUDE.md: Already linked"
        elif [[ -f "$target" ]]; then
            echo "  [WARN] $name/CLAUDE.md: File exists (not a symlink)"
        else
            ln -s "$source_dir/CLAUDE.md" "$target"
            echo "  [DONE] $name/CLAUDE.md: Linked"
        fi
    fi

    # Link settings.json if exists
    if [[ -f "$source_dir/settings.json" ]]; then
        local target="$claude_dir/settings.json"
        if [[ -L "$target" ]]; then
            echo "  [OK]   $name/settings.json: Already linked"
        elif [[ -f "$target" ]]; then
            echo "  [WARN] $name/settings.json: File exists (not a symlink)"
        else
            ln -s "$source_dir/settings.json" "$target"
            echo "  [DONE] $name/settings.json: Linked"
        fi
    fi
}

echo "tt-claude Setup"
echo "==============="
echo "Source: $SCRIPT_DIR"
echo ""

for project in "${PROJECTS[@]}"; do
    IFS=':' read -r name path <<< "$project"
    echo "[$name]"
    link_project "$name" "$path"
    echo ""
done

echo "Setup complete."
