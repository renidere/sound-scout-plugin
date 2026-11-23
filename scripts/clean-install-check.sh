#!/usr/bin/env bash
# clean-install-check.sh - installs the committed HEAD from scratch and checks it works (T10):
# clone into a temp dir, `uv sync --locked`, the test suite, `doctor --json` and `analyze` of a
# synthetic WAV with `backend = none`. Uncommitted changes are not part of the clone, by design.
#
# Neither the operator's config nor the development recording takes part. The ASR model may be
# cached here or not: `asr` is accepted as `completed` or `unavailable`, `audio_review` must be
# `unavailable` (scripts/clean_install_expect.py has the rules).
#
# usage: clean-install-check.sh [pytest args…] — the args go to the pytest step as they are
# output: `clean-install-check: <commit>`, one `<step> | ok|fail | <seconds> | <detail>` line per
# step, then `result: ok` (exit 0) or `result: fail (<step>)` (exit 1). The full output of a
# failed step goes to stderr, with the temp dir shown as `<tmp>`.

set -euo pipefail

root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
repo="$tmp/repo"
offline=(env -u SOUND_SCOUT_CONFIG -u SOUND_SCOUT_DEV_AUDIO HF_HUB_OFFLINE=1 SOUND_SCOUT_NO_NETWORK=1)

# step <name> <detail command> -- <command…>: runs the command with its output in <tmp>/<name>.log,
# prints the summary line, and stops the script on a failure.
step() {
  local name="$1" detail_cmd="$2" started status=0 detail
  shift 3
  started=$SECONDS
  "$@" >"$tmp/$name.log" 2>&1 || status=$?
  detail="$(eval "$detail_cmd" 2>/dev/null || true)"
  if [ "$status" -ne 0 ]; then
    echo "$name | fail | $((SECONDS - started)) | ${detail:-exit $status}"
    echo "--- $name output ---" >&2
    sed "s|$tmp|<tmp>|g" "$tmp/$name.log" >&2
    echo "result: fail ($name)"
    exit 1
  fi
  echo "$name | ok | $((SECONDS - started)) | $detail"
}

echo "clean-install-check: $(git -C "$root" rev-parse --short=12 HEAD)"
step clone "git -C '$repo' log -1 --format=%cs" -- git clone --quiet "$root" "$repo"

cd "$repo"
step "uv sync" "uv --version" -- uv sync --locked --all-groups
step pytest "grep -E '[0-9]+ (passed|failed|error)' '$tmp/pytest.log' | tail -1 | sed -E 's/^=+ *| *=+\$//g'" -- \
  "${offline[@]}" uv run pytest "$@"

mkdir "$tmp/in" "$tmp/data"
cat >"$tmp/sound-scout.toml" <<EOF
allowed_roots = ["$tmp/in"]
data_root = "$tmp/data"
backend = "none"
EOF
offline+=(SOUND_SCOUT_CONFIG="$tmp/sound-scout.toml")
expect=(uv run python -I scripts/clean_install_expect.py)

doctor() { "${offline[@]}" uv run sound-scout doctor --json >"$tmp/doctor.json"; }
step doctor "${expect[*]} --detail doctor '$tmp/doctor.json'" -- doctor

analyze_code=0
analyze() {
  ffmpeg -nostdin -loglevel error -f lavfi -i sine=frequency=220:sample_rate=48000:duration=5 \
    -c:a pcm_s16le "$tmp/in/synthetic.wav" || return 1
  "${offline[@]}" uv run sound-scout analyze "$tmp/in/synthetic.wav" --json \
    >"$tmp/analyze.json" || analyze_code=$?
  # a refusal prints nothing on stdout; a result of any status is for `expect` to judge
  [ -s "$tmp/analyze.json" ]
}
step analyze "echo \"exit=\$analyze_code \$(${expect[*]} --detail analyze '$tmp/analyze.json')\"" -- \
  analyze

check() {
  "${expect[@]}" "$tmp/doctor.json" "$tmp/analyze.json" "$analyze_code"
}
step expect "tr '\n' ';' <'$tmp/expect.log' | sed 's/;\$//'" -- check
echo "result: ok"
