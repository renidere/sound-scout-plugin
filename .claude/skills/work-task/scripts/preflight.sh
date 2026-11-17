#!/usr/bin/env bash
# preflight.sh - read-only checks before working a task; changes nothing.
#
# It looks only for what a person has to fix — no workspace, no GitHub access, no board, no uv or
# ffmpeg — because every task would fail the same way. The main checkout's working tree doesn't
# matter: every task works in a worktree of its own, cut from origin/main (worktree.sh).
#
# usage: preflight.sh [--ws <workspace dir>]   (default: what workspace.sh finds)
# output: lines `STOP: <reason>`, or `ok` if nothing was found
# exit:   0 nothing found, 1 at least one STOP, 2 bad usage

set -u

stop=0
problem() { echo "STOP: $*"; stop=1; }

ws=
while [ $# -gt 0 ]; do
  case "$1" in
    --ws) ws="${2:-}"; shift 2 || { echo "usage: --ws needs a directory" >&2; exit 2; } ;;
    *) echo "usage: preflight.sh [--ws <dir>] — unknown argument '$1'" >&2; exit 2 ;;
  esac
done

if [ -z "$ws" ] && ! ws=$(bash "$(dirname "$0")/workspace.sh" 2>/dev/null); then
  problem "not inside sound-scout-plugin — start Claude Code in its checkout"
fi

if [ -n "$ws" ]; then
  git -C "$ws" remote get-url origin >/dev/null 2>&1 || problem "no origin remote in $ws"
  git -C "$ws" fetch -q origin main 2>/dev/null || problem "git fetch origin main failed in $ws — network or GitHub access"
fi

for tool in gh uv ffmpeg ffprobe; do
  command -v "$tool" >/dev/null 2>&1 || problem "$tool is not installed or not on PATH"
done

if command -v gh >/dev/null 2>&1; then
  if ! gh auth status >/dev/null 2>&1; then
    problem "gh auth status fails — GitHub login is needed (gh auth login)"
  elif ! gh project view 1 --owner renidere >/dev/null 2>&1; then
    problem "can't read the Sound Scout board — gh needs the project scope (gh auth refresh -s project)"
  fi
fi

[ "$stop" = 1 ] && exit 1
echo ok
