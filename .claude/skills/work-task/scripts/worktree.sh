#!/usr/bin/env bash
# worktree.sh - the tree a task works in: a git worktree on a branch of its own cut from
# origin/main, under <workspace>/.work-task/<key>/ (gitignored). A person's work in the main
# checkout stays as it is: it gets a fetch, git's worktree bookkeeping and the task's branches, and
# nothing else — no checkout, no stash, no reset. So several tasks can run side by side.
#
# <key> names the task: the issue number, e.g. 42. The branch is task/<key>.
#
# usage:
#   worktree.sh [--ws <dir>] add <key>
#       after fetching origin/main:
#         `moved <path>`        a folder in the tree's place that is no worktree (a remove that
#                               failed halfway), set aside there
#         `saved <branch>`      uncommitted changes an earlier attempt left in the tree,
#                               committed onto that leftover/ branch
#         `ready <path>`        clean, on origin/main, the main checkout's gitignored local
#                               config copied in
#         `ahead <path> <n>: <sha subject;...>`
#                               the branch holds an earlier attempt's commits that reached no
#                               remote; the tree is on them, clean
#         `FAIL <why>`
#   worktree.sh [--ws <dir>] remove <key>
#       the tree goes, its uncommitted changes saved first, as in add; the branch is deleted once
#       each of its commits is on a remote, and kept otherwise:
#         `removed` | `kept task/<key>: <n> unpushed` | `saved <branch>` | `moved <path>`
#         | `FAIL <why>` | `none` (no tree for <key>)
#   worktree.sh [--ws <dir>] list
#       TSV per key that has a tree: key  issue-url
#   (--ws default: what workspace.sh finds)
# exit: 0 ok, 1 a FAIL line, 2 bad usage, 3 (add) `ahead`

set -u

ISSUES=https://github.com/renidere/sound-scout-plugin/issues

usage() { sed -n '10,30p' "$0" >&2; exit 2; }

ws=
if [ "${1:-}" = --ws ]; then
  ws="${2:-}"
  shift 2 || usage
fi
cmd="${1:-}"
[ -n "$cmd" ] || usage
shift
[ -n "$ws" ] || ws=$(bash "$(dirname "$0")/workspace.sh") || exit 1
root="$ws/.work-task"

failed=0
fail() { echo "FAIL $*"; failed=1; }
stamp() { date +%Y%m%d-%H%M%S; }

# Several tasks fetch the same origin/main ref at once; one that loses the ref lock just tries again.
fetch_main() {
  for _ in 1 2 3; do
    git -C "$1" fetch -q origin main && return 0
    sleep 5
  done
  return 1
}

# A folder with a .git that git itself accepts — not one a half-finished remove left behind.
live_tree() { [ -e "$1/.git" ] && git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1; }

# set_aside <tree> <key> - a folder that is no worktree blocks `git worktree add` for good; it may
# still hold work, so it is moved, never deleted.
set_aside() {
  local dest="$root/.stale/$2-$(stamp)"
  mkdir -p "$root/.stale" && mv "$1" "$dest" || return 1
  echo "moved $dest"
}

# A rebase, merge or cherry-pick that an earlier attempt left half done; aborting loses nothing.
abort_ops() {
  for op in rebase merge cherry-pick; do git -C "$1" "$op" --abort >/dev/null 2>&1; done
}

# save_leftover <tree> <key> - commits whatever is uncommitted in the tree, untracked files too,
# onto a branch leftover/<key>-<time>, then cleans the tree. A private index and commit-tree keep
# it off the shared stash (which is the person's own `git stash list` in the main checkout) and
# away from the repo's commit hooks. Fails only when there was something to save and it couldn't be.
save_leftover() {
  [ -n "$(git -C "$1" status --porcelain)" ] || return 0
  local idx branch tree commit
  idx=$(git -C "$1" rev-parse --path-format=absolute --git-path index.work-task) || return 1
  branch="leftover/$2-$(stamp)"
  rm -f "$idx"
  if GIT_INDEX_FILE="$idx" git -C "$1" read-tree HEAD &&
     GIT_INDEX_FILE="$idx" git -C "$1" add -A &&
     tree=$(GIT_INDEX_FILE="$idx" git -C "$1" write-tree) &&
     commit=$(git -C "$1" commit-tree "$tree" -p HEAD \
       -m "work-task $2: left uncommitted in the tree, $(date '+%Y-%m-%d %H:%M')") &&
     git -C "$1" branch "$branch" "$commit"; then
    rm -f "$idx"
  else
    rm -f "$idx"
    return 1
  fi
  git -C "$1" reset -q --hard && git -C "$1" clean -fdq || return 1
  echo "saved $branch"
}

# The operator's local config and env files are gitignored, so a new worktree lacks them. Values
# are copied, never printed. The virtualenv is not: uv recreates it in the tree.
copy_local() {
  git -C "$1" ls-files --others --ignored --exclude-standard --directory --no-empty-directory |
    grep -E '(^|/)(\.env(\.[^/]*)?|sound-scout\.toml)$' | grep -v '\.example$' |
    while IFS= read -r f; do
      mkdir -p "$2/$(dirname "$f")" && cp -p "$1/$f" "$2/$f"
    done
}

key_ok() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "not a task key (an issue number): $1" >&2; exit 2; }; }

case "$cmd" in
  add)
    key="${1:-}"
    [ -n "$key" ] && [ $# -eq 1 ] || usage
    key_ok "$key"
    base=origin/main
    branch="task/$key"
    tree="$root/$key"
    fetch_main "$ws" || { fail "git fetch origin main failed three times"; exit 1; }
    if [ -e "$tree" ] && ! live_tree "$tree"; then
      set_aside "$tree" "$key" || { fail "$tree is no worktree and could not be moved aside"; exit 1; }
    fi
    # A tree whose folder is gone still blocks its branch until git forgets it.
    git -C "$ws" worktree prune
    if live_tree "$tree"; then
      abort_ops "$tree"
      save_leftover "$tree" "$key" || { fail "could not save the changes in $tree"; exit 1; }
      if [ "$(git -C "$tree" branch --show-current)" != "$branch" ]; then
        git -C "$tree" switch -q "$branch" 2>/dev/null ||
          git -C "$tree" switch -q --no-track -c "$branch" "$base" ||
          { fail "could not switch $tree to $branch"; exit 1; }
      fi
    elif git -C "$ws" rev-parse --verify -q "refs/heads/$branch" >/dev/null; then
      mkdir -p "$root"
      git -C "$ws" worktree add -q "$tree" "$branch" ||
        { fail "could not add a worktree at $tree for $branch"; exit 1; }
    else
      mkdir -p "$root"
      git -C "$ws" worktree add -q --no-track -b "$branch" "$tree" "$base" ||
        { fail "could not add a worktree at $tree"; exit 1; }
    fi
    copy_local "$ws" "$tree"
    n=$(git -C "$tree" rev-list --count HEAD --not --remotes)
    if [ "$n" = 0 ]; then
      git -C "$tree" reset -q --hard "$base"
      echo "ready $tree"
      exit 0
    fi
    echo "ahead $tree $n: $(git -C "$tree" log --format='%h %s' HEAD --not --remotes | head -5 | paste -sd ';' -)"
    exit 3
    ;;

  remove)
    key="${1:-}"
    [ -n "$key" ] || usage
    key_ok "$key"
    branch="task/$key"
    tree="$root/$key"
    [ -e "$tree" ] || { echo none; exit 0; }
    if ! live_tree "$tree"; then
      set_aside "$tree" "$key" || fail "$tree is no worktree and could not be moved aside"
      [ "$failed" = 1 ] && exit 1
      exit 0
    fi
    abort_ops "$tree"
    save_leftover "$tree" "$key" || { fail "could not save the changes in $tree; the tree stays"; exit 1; }
    git -C "$ws" worktree remove --force "$tree" || { fail "could not remove $tree"; exit 1; }
    if git -C "$ws" rev-parse --verify -q "refs/heads/$branch" >/dev/null; then
      n=$(git -C "$ws" rev-list --count "$branch" --not --remotes)
      if [ "$n" = 0 ]; then
        git -C "$ws" branch -q -D "$branch"
        echo removed
      else
        echo "kept $branch: $n unpushed"
      fi
    else
      echo removed
    fi
    exit 0
    ;;

  list)
    [ -d "$root" ] || exit 0
    for dir in "$root"/*/; do
      [ -d "$dir" ] || continue
      key=$(basename "$dir")
      [[ "$key" =~ ^[0-9]+$ ]] || continue
      live_tree "$dir" || continue
      printf '%s\t%s/%s\n' "$key" "$ISSUES" "$key"
    done
    ;;

  *) usage ;;
esac
