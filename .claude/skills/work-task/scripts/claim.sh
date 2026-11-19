#!/usr/bin/env bash
# claim.sh - a lock on one issue, so two agents (two machines, or two sessions on one) never work
# the same task at once.
#
# The lock is a comment on the issue. GitHub numbers comments in the order it stores them, so when
# two agents claim at nearly the same moment, both see both comments once they settle and agree on
# the same winner: the earliest live claim. The loser deletes its own comment and leaves. Labels,
# assignees and card status can't do this: setting them twice succeeds twice, and two agents often
# share one GitHub account, so the claim carries its own id instead of relying on the login.
#
# A claim nobody has refreshed for CLAIM_STALE_MINUTES (default 360) counts as abandoned — its
# agent crashed or lost its session — and the next claimant removes it. The holder refreshes its
# claim with `heartbeat` between steps; a heartbeat that finds the claim gone means another agent
# took the task over, and the holder must stop.
#
# The repository is public, so only comments by the repository's owner or a collaborator count as
# claims: anyone else's comment, whatever it says, is ignored.
#
# usage:
#   claim.sh acquire <issue-url>              -> `acquired <claim-id>` (exit 0)
#                                                 `taken <claim-id> <minutes since last seen>` (exit 3)
#                                                 `orphan <claim-id> <comment-id|->` (exit 5): failed,
#                                                 and this attempt's claim may be left on the issue
#   claim.sh heartbeat <issue-url> <claim-id> -> `ok` (exit 0) | `lost` (exit 3)
#   claim.sh release <issue-url> <claim-id>   -> `released` (exit 0)
# any other exit code is an error (network, auth): the claim state is unknown, retry.

set -euo pipefail

STALE=${CLAIM_STALE_MINUTES:-360}
SETTLE=${CLAIM_SETTLE_SECONDS:-8}
RETRY=${CLAIM_RETRY_SECONDS:-5}
ORPHAN=5
MARK='<!-- work-task-claim id='
TRUSTED='(.author_association == "OWNER" or .author_association == "COLLABORATOR" or .author_association == "MEMBER")'

usage() { sed -n '20,27p' "$0" >&2; exit 64; }

cmd="${1:-}"
url="${2:-}"
[ -n "$cmd" ] && [ -n "$url" ] || usage
if [[ "$url" =~ ^https://github\.com/([^/]+)/([^/]+)/issues/([0-9]+) ]]; then
  repo="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
  number="${BASH_REMATCH[3]}"
else
  echo "not an issue url: $url" >&2
  exit 64
fi

now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

body_for() { # <claim-id> <started>
  printf '%s%s -->\nAn agent is working on this task: `work-task`, claim `%s`, started %s, last seen %s.\nAnother agent that finds this comment leaves the task alone. The comment is removed when the work ends; one not updated for %s minutes counts as abandoned.\n' \
    "$MARK" "$1" "$1" "$2" "$(now_utc)" "$STALE"
}

# TSV per claim, oldest first: comment-id  claim-id  minutes-since-update  started
claims() {
  gh api --paginate "repos/$repo/issues/$number/comments" --jq '.[]
    | select('"$TRUSTED"')
    | select(.body | startswith("'"$MARK"'"))
    | [.id,
       (.body | capture("id=(?<c>[^ ]+) -->").c),
       (((now - (.updated_at | fromdateiso8601)) / 60) | floor),
       ([.body | capture("started (?<s>[^,]+),") | .s] | first // "")]
    | @tsv' | sort -n
}

live() { awk -F'\t' -v s="$STALE" '$3 < s'; }
stale() { awk -F'\t' -v s="$STALE" '$3 >= s'; }
delete_comment() { gh api -X DELETE "repos/$repo/issues/comments/$1" >/dev/null 2>&1 || true; }

# Three tries; a comment already gone (404) counts as deleted.
delete_with_retries() { # <comment-id>
  local n out
  for n in 1 2 3; do
    out=$(gh api -X DELETE "repos/$repo/issues/comments/$1" 2>&1 >/dev/null) && return 0
    printf '%s\n' "$out" >&2
    [[ "$out" == *"HTTP 404"* ]] && return 0
    [ "$n" = 3 ] || sleep "$RETRY"
  done
  return 1
}

claims_with_retries() {
  local n out
  for n in 1 2 3; do
    out=$(claims) && { printf '%s\n' "$out"; return 0; }
    [ "$n" = 3 ] || sleep "$RETRY"
  done
  return 1
}

orphan() { # <claim-id> <comment-id|->
  printf 'orphan %s %s\n' "$1" "$2"
  exit "$ORPHAN"
}

# Our claim comment is up, but acquire can't finish: take it down, or name it.
abandon() {
  trap - ERR
  delete_with_retries "$mine" || orphan "$id" "$mine"
  echo "acquire failed; claim $id removed" >&2
  exit 1
}

case "$cmd" in
  acquire)
    host=$(hostname 2>/dev/null | tr -cd 'A-Za-z0-9.-')
    id="${host:-host}-$(date +%s)-$RANDOM"

    current=$(claims)
    holder=$(printf '%s\n' "$current" | live | head -n1)
    if [ -n "$holder" ]; then
      printf 'taken %s %s\n' "$(cut -f2 <<<"$holder")" "$(cut -f3 <<<"$holder")"
      exit 3
    fi

    if ! mine=$(gh api -X POST "repos/$repo/issues/$number/comments" -f body="$(body_for "$id" "$(now_utc)")" --jq .id); then
      # The response may be lost while the comment went up: look for it by the claim id.
      sleep "$SETTLE"
      found=$(claims_with_retries) || orphan "$id" -
      lost=$(awk -F'\t' -v c="$id" '$2 == c { print $1; exit }' <<<"$found")
      if [ -n "$lost" ]; then
        delete_with_retries "$lost" || orphan "$id" "$lost"
        echo "acquire failed; claim $id was posted and removed" >&2
      else
        echo "acquire failed: the claim comment was not posted" >&2
      fi
      exit 1
    fi
    trap abandon ERR

    # Let a claim posted at nearly the same moment become visible before deciding.
    sleep "$SETTLE"
    current=$(claims)
    winner=$(printf '%s\n' "$current" | live | head -n1)
    if [ "$(cut -f2 <<<"$winner")" != "$id" ]; then
      trap - ERR
      delete_with_retries "$mine" || orphan "$id" "$mine"
      printf 'taken %s %s\n' "$(cut -f2 <<<"$winner")" "$(cut -f3 <<<"$winner")"
      exit 3
    fi
    trap - ERR
    # Abandoned claims of crashed agents go, so the issue doesn't collect them.
    printf '%s\n' "$current" | stale | cut -f1 | while read -r old; do
      [ -n "$old" ] && delete_comment "$old"
    done
    echo "acquired $id"
    ;;

  heartbeat)
    id="${3:?usage: claim.sh heartbeat <issue-url> <claim-id>}"
    current=$(claims)
    winner=$(printf '%s\n' "$current" | live | head -n1)
    mine=$(printf '%s\n' "$current" | awk -F'\t' -v c="$id" '$2 == c { print; exit }')
    # Gone, or not the earliest live claim any more: another agent owns the task now.
    if [ -z "$mine" ] || [ "$(cut -f2 <<<"$winner")" != "$id" ]; then
      echo lost
      exit 3
    fi
    started=$(cut -f4 <<<"$mine")
    gh api -X PATCH "repos/$repo/issues/comments/$(cut -f1 <<<"$mine")" \
      -f body="$(body_for "$id" "${started:-$(now_utc)}")" >/dev/null
    echo ok
    ;;

  release)
    id="${3:?usage: claim.sh release <issue-url> <claim-id>}"
    claims | awk -F'\t' -v c="$id" '$2 == c { print $1 }' | while read -r cid; do
      [ -n "$cid" ] && gh api -X DELETE "repos/$repo/issues/comments/$cid" >/dev/null
    done
    echo released
    ;;

  *) usage ;;
esac
