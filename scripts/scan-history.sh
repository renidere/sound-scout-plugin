#!/usr/bin/env bash
# scan-history.sh - looks through the whole history of a revision for what must never have reached
# the public repository: media files, env files, operator config, recordings and runs, task trees,
# anything shaped like a key, and any `/home/<user>/` path except the placeholders tests use.
#
# The rules are those of .claude/skills/work-task/scripts/scan-staged.sh (tests/test_scan_history.py
# catches a drift), applied to every commit instead of the staged diff, plus more media extensions
# and the `/home/<user>/` rule. A file removed later is still a finding: it stays in the history.
# Matched values are never printed, only where they are.
#
# usage: scan-history.sh [<rev>]   (default HEAD)
# output: `ok` and exit 0, or one `BLOCK: <commit 12> <file>[:<line>] <why>` per hit and exit 1

set -u

# Placeholder user names tests and docs may put in a `/home/<user>/` path.
home_allowlist='someone|user|username|runner|example'

rev="${1:-HEAD}"
fail=0
block() { echo "BLOCK: $*"; fail=1; }

if ! git rev-parse --verify --quiet "$rev^{commit}" >/dev/null; then
  echo "scan-history.sh: not a commit: $rev" >&2
  exit 2
fi

shopt -s nocasematch
commit=''
# `--no-renames`: a file renamed onto a forbidden name is an added path too.
while IFS= read -r -d '' path; do
  path="${path#$'\n'}"
  if [[ "$path" =~ ^commit\ ([0-9a-f]{40})$ ]]; then
    commit="${BASH_REMATCH[1]:0:12}"
    continue
  fi
  [ -n "$path" ] || continue
  case "$path" in
    .env|.env.*|*/.env|*/.env.*)
      case "$path" in *.example) ;; *) block "$commit $path is an env file" ;; esac ;;
    sound-scout.toml|*/sound-scout.toml)
      block "$commit $path is the operator's local config" ;;
    data/*|runs/*|local/*|.work-task/*|.revmux/*|.venv/*|*.pem|*.p12|*.key)
      block "$commit $path should not be committed" ;;
    *.wav|*.mp3|*.ogg|*.oga|*.opus|*.m4a|*.aac|*.flac|*.webm|*.mp4|*.mov|*.aif|*.aiff|*.wma|*.mkv|*.avi|*.3gp|*.amr)
      block "$commit $path is a media file" ;;
  esac
done < <(git -c core.quotePath=false log "$rev" --format='commit %H' --name-only \
  --diff-filter=A --no-renames -z)
shopt -u nocasematch

# The key patterns of scan-staged.sh, the same line: tests/test_scan_history.py compares them.
patterns='AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|hf_[A-Za-z0-9]{30,}|(password|secret|api[_-]?key|token)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"'[:space:]]{12,}|/home/[a-z0-9_-]+/\.hermes/'

# Added lines of every commit as `<commit 12> <file>:<line>\t<text>`. awk only numbers the lines and
# grep matches them, as in scan-staged.sh (mawk reads `{16}` as plain text).
tab=$'\t'
added=$(mktemp)
trap 'rm -f "$added"' EXIT
git -c core.quotePath=false log "$rev" --format='commit %H' -p -U0 --no-color |
  LC_ALL=C awk '
    /^commit [0-9a-f]+$/ { commit = substr($2, 1, 12); header = 0; next }
    /^diff --git / { header = 1; next }
    header && /^\+\+\+ / { file = substr($0, 5); sub(/\t$/, "", file)
                           if (file ~ /^".*"$/) file = substr(file, 2, length(file) - 2)
                           sub(/^b\//, "", file); next }
    /^@@/ { header = 0; split($3, a, ","); line = substr(a[1], 2) + 0; next }
    !header && /^\+/ { print commit " " file ":" line "\t" substr($0, 2); line++ }
  ' >"$added"

LC_ALL=C grep -aiE "^[^$tab]*$tab.*($patterns)" "$added" | cut -f1 |
  sed 's/^/BLOCK: /; s/$/ looks like a secret or a private local path/' | grep . && fail=1

# Any `/home/<user>/` left once the placeholders are taken out.
LC_ALL=C sed -E "s#/home/($home_allowlist)/##g" "$added" |
  LC_ALL=C grep -aE "^[^$tab]*$tab.*/home/[A-Za-z0-9._-]+/" | cut -f1 |
  sed 's/^/BLOCK: /; s/$/ holds a \/home\/<user>\/ path/' | grep . && fail=1

[ "$fail" = 0 ] && echo ok
exit "$fail"
