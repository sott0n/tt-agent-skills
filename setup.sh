#!/bin/bash
# Setup script for tt-claude
# Creates symlinks from each project's .claude directory to tt-claude configs
#
# Usage:
#   ./setup.sh <project>    Setup specific project (tt-metal, tt-mlir, tt-xla)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Available projects
PROJECTS="tt-metal tt-mlir tt-xla"

# Get git repository URL for a project
get_repo_url() {
    local name=$1
    case "$name" in
        tt-metal) echo "https://github.com/tenstorrent/tt-metal.git" ;;
        tt-mlir)  echo "https://github.com/tenstorrent/tt-mlir.git" ;;
        tt-xla)   echo "https://github.com/tenstorrent/tt-xla.git" ;;
        *)        echo "" ;;
    esac
}

# Find project directory under HOME (max depth 2)
find_project() {
    local name=$1
    local found=""

    # Search HOME with max depth 2
    found=$(find "$HOME" -maxdepth 2 -type d -name "$name" 2>/dev/null | head -n 1)

    echo "$found"
}

# Clone project if not found
clone_project() {
    local name=$1
    local repo_url
    repo_url=$(get_repo_url "$name")
    local clone_dir="$HOME/$name"

    if [[ -z "$repo_url" ]]; then
        echo "  [ERROR] Unknown project: $name"
        return 1
    fi

    echo "  Project not found. Clone to $clone_dir? [y/N]"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        git clone "$repo_url" "$clone_dir"
        echo "$clone_dir"
    else
        echo ""
    fi
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
                echo "  [OK]   $label/$skill_name: Already linked"
            elif [[ -d "$target" ]]; then
                echo "  [WARN] $label/$skill_name: Directory exists (not a symlink)"
            else
                ln -s "$skill_dir" "$target"
                echo "  [DONE] $label/$skill_name: Linked"
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

    if [[ ! -d "$source_dir" ]]; then
        echo "  [SKIP] No config in tt-claude for $name"
        return
    fi

    local claude_dir="$project_path/.claude"
    mkdir -p "$claude_dir"

    # Create skills directory (not a symlink, to hold both project and common skills)
    local skills_dir="$claude_dir/skills"
    if [[ -L "$skills_dir" ]]; then
        echo "  [WARN] skills: Is a symlink from old setup. Removing to use new structure."
        rm "$skills_dir"
    fi
    mkdir -p "$skills_dir"

    # Link project-specific skills
    link_skills "$source_dir/skills" "$skills_dir" "skills"

    # Link common skills
    link_skills "$common_dir/skills" "$skills_dir" "skills(common)"

    # Link CLAUDE.md if exists
    if [[ -f "$source_dir/CLAUDE.md" ]]; then
        local target="$claude_dir/CLAUDE.md"
        if [[ -L "$target" ]]; then
            echo "  [OK]   CLAUDE.md: Already linked"
        elif [[ -f "$target" ]]; then
            echo "  [WARN] CLAUDE.md: File exists (not a symlink)"
        else
            ln -s "$source_dir/CLAUDE.md" "$target"
            echo "  [DONE] CLAUDE.md: Linked"
        fi
    fi

    # Link settings.json if exists
    if [[ -f "$source_dir/settings.json" ]]; then
        local target="$claude_dir/settings.json"
        if [[ -L "$target" ]]; then
            echo "  [OK]   settings.json: Already linked"
        elif [[ -f "$target" ]]; then
            echo "  [WARN] settings.json: File exists (not a symlink)"
        else
            ln -s "$source_dir/settings.json" "$target"
            echo "  [DONE] settings.json: Linked"
        fi
    fi
}

# Setup a single project
setup_project() {
    local name=$1

    echo "[$name]"

    # Find project
    local project_path
    project_path=$(find_project "$name")

    if [[ -z "$project_path" ]]; then
        echo "  Project not found under \$HOME (depth 2)"
        project_path=$(clone_project "$name")
        if [[ -z "$project_path" ]]; then
            echo "  [SKIP] Skipping $name"
            return
        fi
    else
        echo "  Found: $project_path"
    fi

    link_project "$name" "$project_path"
}

show_help() {
    echo "tt-claude Setup"
    echo ""
    echo "Usage: ./setup.sh <project>"
    echo ""
    echo "Available projects:"
    for name in $PROJECTS; do
        echo "  - $name"
    done
    echo ""
    echo "Example:"
    echo "  ./setup.sh tt-metal"
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
