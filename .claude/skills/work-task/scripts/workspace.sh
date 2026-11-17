#!/usr/bin/env bash
# workspace.sh - print the workspace: the main checkout of sound-scout-plugin, found from the
# current directory, a folder inside it, or one of a task's trees under .work-task/ (a worktree's
# common git dir is the main checkout's). Nothing else counts: a clone of another repo is an error.
#
# usage: workspace.sh
# output: the main checkout's absolute path
# exit:   0 found, 1 not inside sound-scout-plugin

set -u

common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || {
  echo "not inside a git repository: $PWD" >&2
  exit 1
}
ws=$(dirname "$common")
url=$(git -C "$ws" remote get-url origin 2>/dev/null || true)
case "$url" in
  *sound-scout-plugin*) echo "$ws" ;;
  *) echo "the repository at $ws is not sound-scout-plugin (origin: ${url:-none})" >&2; exit 1 ;;
esac
