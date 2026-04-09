#!/bin/bash
# Setup script for tt-claude
# Creates symlinks from each project's .claude directory to tt-claude configs
#
# Usage:
#   ./setup.sh <project>    Setup specific project (tt-metal, tt-forge)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Available projects
PROJECTS="tt-metal tt-forge"

# tt-forge links to multiple repositories
TT_FORGE_REPOS="tt-forge-models tt-xla tt-onnx-fe tt-mlir"

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

# Link project configs
link_project() {
    local name=$1
    local project_path=$2
    local source_dir="$SCRIPT_DIR/$name"
    local common_dir="$SCRIPT_DIR/common"

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

    # Link common skills
    link_skills "$common_dir/skills" "$skills_dir" "skills(common)"

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
    local config_name=$1  # Config directory name (tt-metal or tt-forge)
    local repo_name=$2    # Target repository name

    echo "[$repo_name]"

    # Find project
    local project_path
    project_path=$(find_project "$repo_name")

    if [[ -z "$project_path" ]]; then
        echo "  [SKIP] Not found under \$HOME (depth 2)"
        return
    fi

    echo "  Found: $project_path"
    link_project "$config_name" "$project_path"
}

# Setup a project (may link to multiple repos)
setup_project() {
    local name=$1

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
    echo "Usage: ./setup.sh <project>"
    echo ""
    echo "Available projects:"
    echo "  - tt-metal"
    echo "  - tt-forge (links to: $TT_FORGE_REPOS)"
    echo ""
    echo "Example:"
    echo "  ./setup.sh tt-metal"
    echo "  ./setup.sh tt-forge"
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

setup_project "$1"

echo ""
echo "Setup complete."
