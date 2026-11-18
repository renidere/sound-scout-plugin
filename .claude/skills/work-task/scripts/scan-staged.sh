#!/usr/bin/env bash
# scan-staged.sh - last look at what is about to be committed, for files and lines that must never
# reach the public repository: env files, operator config, recordings and runs, review archives,
# task trees, and anything shaped like a key.
#
# The repository is public: a secret or a recording caught here costs nothing, one caught after
# the push means rotating the key or rewriting history. Matched values are never printed, only
# where they are.
#
# usage: scan-staged.sh <repo dir>
# output: `ok` and exit 0, or one `BLOCK: <file>[:<line>] <why>` per hit and exit 1

set -u

repo="${1:?usage: scan-staged.sh <repo dir>}"
fail=0
block() { echo "BLOCK: $*"; fail=1; }

while IFS= read -r path; do
  case "$path" in
    .env|.env.*|*/.env|*/.env.*)
      case "$path" in *.example) ;; *) block "$path is an env file" ;; esac ;;
    sound-scout.toml|*/sound-scout.toml)
      block "$path is the operator's local config (sound-scout.example.toml is the one to commit)" ;;
    docs/spec/*)
      block "$path is the owner's requirements package: agents never change docs/spec/" ;;
    data/*|runs/*|local/*|.work-task/*|.revmux/*|.venv/*|htmlcov/*|coverage/*|*.pem|*.p12|*.key)
      block "$path should not be committed" ;;
    *.wav|*.WAV|*.mp3|*.ogg|*.oga|*.opus|*.m4a|*.aac|*.flac|*.webm|*.mp4|*.mov)
      block "$path is a media file — tests synthesize their fixtures, recordings stay out of git" ;;
  esac
done < <(git -C "$repo" -c core.quotePath=false diff --cached --name-only --diff-filter=ACMR)

# Added lines only, with file and line number, so a hit can be found without printing the value.
# awk only numbers the lines and grep matches them: the awk of Ubuntu 22.04 (mawk 20200120) and of
# older macOS reads `{16}` as plain text, so an awk match would silently miss most of these.
patterns='AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|hf_[A-Za-z0-9]{30,}|(password|secret|api[_-]?key|token)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"'[:space:]]{12,}|/home/[a-z0-9_-]+/\.hermes/'
# Bytes, not characters: under a UTF-8 locale grep skips a line with a byte that isn't valid UTF-8
# (Latin-1 text, a minified asset), and the secret on it with it.
# A `+++ ` line is the path header only between `diff --git` and the first hunk: an added line
# that starts with `++ ` reads the same inside one. The header can end in a tab (a name with a
# space) or be quoted (a `"` or `\` in the name).
tab=$'\t'
git -C "$repo" -c core.quotePath=false diff --cached -U0 --no-color | LC_ALL=C awk '
  /^diff --git / { header = 1; next }
  header && /^\+\+\+ / { file = substr($0, 5); sub(/\t$/, "", file)
                         if (file ~ /^".*"$/) file = substr(file, 2, length(file) - 2)
                         sub(/^b\//, "", file); next }
  /^@@/ { header = 0; split($3, a, ","); line = substr(a[1], 2) + 0; next }
  !header && /^\+/ { print file ":" line "\t" substr($0, 2); line++ }
' | LC_ALL=C grep -aiE "^[^$tab]*$tab.*($patterns)" | cut -f1 | sed 's/^/BLOCK: /; s/$/ looks like a secret or a private local path/' |
  grep . && fail=1

[ "$fail" = 0 ] && echo ok
exit "$fail"
