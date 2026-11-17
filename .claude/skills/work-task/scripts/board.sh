#!/usr/bin/env bash
# board.sh - the GitHub Project calls the backlog agent needs, so the GraphQL and the board IDs
# live in one place instead of being retyped (and mistyped) on every task.
#
# The board: https://github.com/users/renidere/projects/1 (a user project; its columns are the
# Status field's options Backlog, In Progress, Need a Person, Done). The tasks: issues of
# renidere/sound-scout-plugin.
#
# usage:
#   board.sh queue                        every qualifying task in Backlog, oldest first, TSV:
#                                         item-id  issue-url  parent-url|-  title
#   board.sh in-progress                  every task in In Progress with a verdict on its state, TSV:
#                                         item-id  issue-url  verdict  claim-id|-  claim-minutes|-  title
#                                         verdict: closed | working | abandoned | left-<outcome> | unknown
#   board.sh tasks                        every issue on the board, open or closed, TSV:
#                                         status|-  state  issue-url  title
#   board.sh last-outcome <issue-url> [--body]
#                                         the newest outcome (done, failed, skipped, blocked) by
#                                         a trusted author, or `none`; with --body, that comment's
#                                         body after it
#   board.sh issue <issue-url>            title, state (with the date it closed), labels, body,
#                                         open blockers and all comments, as text
#   board.sh blocked-by <issue-url> <blocker-url>...
#                                         link each blocker to the issue as GitHub "blocked by",
#                                         unless it already is; per blocker: added|exists <url>
#   board.sh status <item-id> backlog|need-a-person|in-progress|done
#   board.sh add <issue-url> [backlog|need-a-person]
#                                         put the issue on the board in that column (Backlog by
#                                         default), print its item id
#   board.sh item <issue-url>             TSV: the issue's item id on this board and its Status
#                                         (empty if it isn't on the board)
#   board.sh subissues <epic-url>         TSV per sub-issue: state  url  title
#   board.sh add-subissue <epic-url> <issue-url>
#                                         make the issue a sub-issue of the epic (idempotent)
#   board.sh decisions <epic-url> [--full]
#                                         "Decision:" comments by trusted authors on the epic and
#                                         all its sub-issues: the first 400 characters of each, or
#                                         whole
#
# Only gh is needed: filtering uses gh's built-in --jq, so it runs the same in Git Bash on Windows.

set -euo pipefail

OWNER=renidere
REPO=sound-scout-plugin
PROJECT_NUMBER=1
PROJECT_ID=PVT_kwHOAiGAjM4Br_LM
STATUS_FIELD=PVTSSF_lAHOAiGAjM4Br_LMzhqPUew
OPT_BACKLOG=55461044
OPT_IN_PROGRESS=e44e5bff
OPT_NEED_A_PERSON=094e762a
OPT_DONE=394af4c2
# The repository is public, so a comment steers the work only when its author is the owner or a
# collaborator. REST calls read author_association, GraphQL ones authorAssociation.
PERSON='(.author_association? == "OWNER" or .author_association? == "COLLABORATOR" or .author_association? == "MEMBER")'
GQL_PERSON='(.authorAssociation? == "OWNER" or .authorAssociation? == "COLLABORATOR" or .authorAssociation? == "MEMBER")'
# Shared with claim.sh: the claim comment's marker and when a claim counts as abandoned.
CLAIM_MARK='<!-- work-task-claim id='
STALE=${CLAIM_STALE_MINUTES:-360}
# work-task opens every final comment (done, failed, skipped, blocked) with this marker.
OUTCOME_MARK='<!-- work-task-outcome: '

usage() { sed -n '9,41p' "$0" >&2; exit 2; }

# gh_read <gh args>: a read, retried when GitHub answers an error page, a 5xx, a timeout or
# GraphQL's "Something went wrong" instead of data, which it does under load and then answers the
# same call moments later. The output is held until the call succeeds, so a paginated read that
# failed on a later page never prints a page twice.
gh_read() {
  local err out status pause
  err=$(mktemp)
  for pause in 10 30 60 last; do
    status=0
    out=$(gh "$@" 2>"$err") || status=$?
    if [ "$status" = 0 ]; then
      rm -f "$err"
      [ -z "$out" ] || printf '%s\n' "$out"
      return 0
    fi
    grep -qiE 'invalid character|Something went wrong while executing your query|HTTP 5[0-9][0-9]|timeout|timed out|connection reset|unexpected EOF' "$err" \
      && [ "$pause" != last ] || break
    printf 'board.sh: GitHub failed (%s); retrying in %ss\n' "$(head -n 1 "$err")" "$pause" >&2
    sleep "$pause"
  done
  cat "$err" >&2
  rm -f "$err"
  return "$status"
}

issue_parts() { # <issue-url> -> sets number
  if [[ "$1" =~ ^https://github\.com/$OWNER/$REPO/issues/([0-9]+)$ ]]; then
    number="${BASH_REMATCH[1]}"
  else
    echo "not an issue url of $OWNER/$REPO: $1" >&2
    exit 64
  fi
}

project_items() { # every item of the board with its Status and issue content, one JSON per line
  gh_read api graphql --paginate -f query='
    query($endCursor: String) {
      user(login: "'"$OWNER"'") {
        projectV2(number: '"$PROJECT_NUMBER"') {
          items(first: 100, after: $endCursor) {
            pageInfo { hasNextPage endCursor }
            nodes {
              id
              fieldValueByName(name: "Status") { ... on ProjectV2ItemFieldSingleSelectValue { name } }
              content {
                ... on Issue {
                  url title state createdAt
                  parent { url }
                  subIssues { totalCount }
                  blockedBy(first: 20) { nodes { state } }
                }
              }
            }
          }
        }
      }
    }' --jq '.data.user.projectV2.items.nodes[]'
}

cmd="${1:-}"
[ -n "$cmd" ] || usage
shift

case "$cmd" in
  queue)
    # A task qualifies when it is an open issue in Backlog with no sub-issues (issues with
    # sub-issues are epics: they close when their children are done) and no open "blocked by"
    # link. Pages arrive one by one, so sorting waits for all of them: createdAt is ISO 8601,
    # which sorts correctly as plain text.
    project_items | jq -r '
        select(.fieldValueByName.name? == "Backlog")
        | select(.content.state? == "OPEN"
                 and .content.subIssues.totalCount? == 0
                 and ([.content.blockedBy.nodes[]? | select(.state == "OPEN")] | length) == 0)
        | [.content.createdAt, .id, .content.url, (.content.parent.url // "-"),
           (.content.title | gsub("[\t\n\r]"; " "))]
        | @tsv' \
      | sort | cut -f2-
    ;;

  in-progress)
    # Tells a task an agent is still on from one it left hanging. Claims are removed when work
    # ends, so a claim with no outcome comment after it means the agent never finished:
    #   closed       the issue is closed but the card wasn't moved (the agent stopped right after closing)
    #   working      a live claim: an agent is on it, here or on another machine
    #   abandoned    only stale claims, no outcome after them: the agent died mid-task
    #   left-<o>     the agent's last word is an outcome comment (failed, skipped): left on purpose
    #   unknown      no claim and no outcome: a person moved it, or it predates the agent. Also a
    #                `blocked` last word: that run put the card back in Backlog before commenting,
    #                so someone moved it to In Progress since
    cards=$(project_items | jq -r '
        select(.fieldValueByName.name? == "In Progress" and .content.url? != null)
        | [.id, .content.url] | @tsv')
    while IFS=$'\t' read -r item url; do
      [ -n "$item" ] || continue
      row=$(gh_read api graphql -f url="$url" -f query='
        query($url: URI!) {
          resource(url: $url) {
            ... on Issue {
              url title state
              subIssues { totalCount }
              comments(last: 50) { nodes { body updatedAt authorAssociation } }
            }
          }
        }' --jq '.data.resource
          | select(.subIssues.totalCount? == 0)
          | . as $c
          | [$c.comments.nodes | to_entries[]
             | select(.value | '"$GQL_PERSON"')
             | select(.value.body | startswith("'"$CLAIM_MARK"'"))
             | {i: .key,
                id: (.value.body | capture("id=(?<c>[^ ]+) -->").c),
                age: (((now - (.value.updatedAt | fromdateiso8601)) / 60) | floor)}] as $claims
          | ([$c.comments.nodes | to_entries[]
              | select(.value | '"$GQL_PERSON"')
              | select(.value.body | startswith("'"$OUTCOME_MARK"'"))
              | {i: .key, o: (.value.body | capture("outcome: (?<o>[a-z]+)").o)}] | last) as $outcome
          | ($claims | map(select(.age < '"$STALE"'))) as $live
          | (if $c.state == "CLOSED" then "closed"
             elif ($live | length) > 0 then "working"
             elif $outcome != null and $outcome.i > ([$claims[].i] | max // -1)
               then (if $outcome.o == "blocked" then "unknown" else "left-" + $outcome.o end)
             elif ($claims | length) > 0 then "abandoned"
             else "unknown" end) as $verdict
          | ($live[0] // $claims[-1]) as $claim
          | [$c.url, $verdict, ($claim.id // "-"), ($claim.age // "-" | tostring),
             ($c.title | gsub("[\t\n\r]"; " "))]
          | @tsv')
      [ -z "$row" ] || printf '%s\t%s\n' "$item" "$row"
    done <<< "$cards"
    ;;

  tasks)
    # For new-task's "already on the board?" check. Listed directly, not through search: the
    # search index lags behind a just-created issue.
    project_items | jq -r '
        select(.content.url?)
        | [(.fieldValueByName.name // "-"), .content.state, .content.url,
           (.content.title | gsub("[\t\n\r]"; " "))]
        | @tsv'
    ;;

  last-outcome)
    # What the latest run left the task at. --body is for a `blocked` run, which posts its comment
    # only when the last one doesn't already name the same blockers.
    issue_parts "${1:?usage: board.sh last-outcome <issue-url> [--body]}"
    case "${2:-}" in '' | --body) ;; *) usage ;; esac
    last=$(gh_read api --paginate "repos/$OWNER/$REPO/issues/$number/comments" --jq '.[]
      | select('"$PERSON"')
      | select(.body | startswith("'"$OUTCOME_MARK"'"))
      | "\(.id)\t\(.body | capture("outcome: (?<o>[a-z]+)").o)"' | tail -n 1)
    if [ -z "$last" ]; then
      echo none
      exit 0
    fi
    echo "${last#*$'\t'}"
    [ "${2:-}" != --body ] || gh_read api "repos/$OWNER/$REPO/issues/comments/${last%%$'\t'*}" --jq .body
    ;;

  issue)
    # `gh issue view --comments` prints only the comments when its output isn't a terminal, which
    # is always the case for an agent, so the body has to be asked for explicitly.
    url="${1:?usage: board.sh issue <issue-url>}"
    gh_read api graphql -f url="$url" -f query='
      query($url: URI!) {
        resource(url: $url) {
          ... on Issue {
            url title state closedAt body
            parent { url }
            labels(first: 20) { nodes { name } }
            assignees(first: 10) { nodes { login } }
            blockedBy(first: 20) { nodes { url state } }
            comments(first: 100) { nodes { url author { login } authorAssociation body } }
          }
        }
      }' --jq '.data.resource
        | "# \(.title)\n\(.url) — \(.state)\(if .closedAt then " on \(.closedAt[0:10])" else "" end)\nParent: \(.parent.url // "none")\n"
          + "Labels: \([.labels.nodes[].name] | join(", ") | if . == "" then "none" else . end)\n"
          + "Assignees: \([.assignees.nodes[].login] | join(" ") | if . == "" then "none" else . end)\n"
          + "Blocked by (open): \([.blockedBy.nodes[] | select(.state == "OPEN") | .url] | join(" ") | if . == "" then "none" else . end)\n\n"
          + (.body // "") + "\n"
          + ([.comments.nodes[] | "\n--- comment by \(.author.login // "ghost") (\(.authorAssociation)) \(.url)\n\(.body)"] | join("\n"))'
    ;;

  blocked-by)
    # The queue passes over an issue with an open "blocked by" link, so the link alone keeps a
    # partly done task waiting until its blockers close. The REST endpoint wants the blocker's
    # numeric id, not its number.
    issue_parts "${1:?usage: board.sh blocked-by <issue-url> <blocker-url>...}"
    shift
    [ $# -gt 0 ] || usage
    # The explicit exit: macOS's bash 3.2 doesn't stop on a failed subshell under set -e.
    for blocker in "$@"; do (issue_parts "$blocker") || exit 64; done
    linked=$(gh_read api --paginate "repos/$OWNER/$REPO/issues/$number/dependencies/blocked_by" --jq '.[].html_url')
    for blocker in "$@"; do
      if grep -qxF "$blocker" <<< "$linked"; then
        echo "exists $blocker"
        continue
      fi
      id=$(gh_read api "repos/${blocker#https://github.com/}" --jq .id)
      gh api -X POST "repos/$OWNER/$REPO/issues/$number/dependencies/blocked_by" -F issue_id="$id" >/dev/null
      echo "added $blocker"
    done
    ;;

  status)
    item="${1:?usage: board.sh status <item-id> backlog|need-a-person|in-progress|done}"
    case "${2:-}" in
      backlog) opt=$OPT_BACKLOG ;;
      need-a-person) opt=$OPT_NEED_A_PERSON ;;
      in-progress) opt=$OPT_IN_PROGRESS ;;
      done) opt=$OPT_DONE ;;
      *) usage ;;
    esac
    gh project item-edit --project-id "$PROJECT_ID" --id "$item" \
      --field-id "$STATUS_FIELD" --single-select-option-id "$opt" >/dev/null
    echo "ok: $item -> $2"
    ;;

  add)
    url="${1:?usage: board.sh add <issue-url> [backlog|need-a-person]}"
    to="${2:-backlog}"
    case "$to" in backlog | need-a-person) ;; *) usage ;; esac
    item=$(gh project item-add "$PROJECT_NUMBER" --owner "$OWNER" --url "$url" --format json --jq .id)
    bash "$0" status "$item" "$to" >/dev/null
    echo "$item"
    ;;

  item)
    url="${1:?usage: board.sh item <issue-url>}"
    gh_read api graphql -f url="$url" -f query='
      query($url: URI!) {
        resource(url: $url) {
          ... on Issue {
            projectItems(first: 20) {
              nodes {
                id project { id }
                fieldValueByName(name: "Status") { ... on ProjectV2ItemFieldSingleSelectValue { name } }
              }
            }
          }
        }
      }' --jq '.data.resource.projectItems.nodes[] | select(.project.id == "'"$PROJECT_ID"'")
        | [.id, (.fieldValueByName.name // "-")] | @tsv'
    ;;

  subissues)
    url="${1:?usage: board.sh subissues <epic-url>}"
    gh_read api graphql -f url="$url" -f query='
      query($url: URI!) {
        resource(url: $url) { ... on Issue { subIssues(first: 100) { nodes { state url title } } } }
      }' --jq '.data.resource.subIssues.nodes[] | [.state, .url, (.title | gsub("[\t\n\r]"; " "))] | @tsv'
    ;;

  add-subissue)
    parent="${1:?usage: board.sh add-subissue <epic-url> <issue-url>}"
    child="${2:?usage: board.sh add-subissue <epic-url> <issue-url>}"
    (issue_parts "$parent") || exit 64
    (issue_parts "$child") || exit 64
    if bash "$0" subissues "$parent" | cut -f2 | grep -qxF "$child"; then
      echo "exists $child"
      exit 0
    fi
    pid=$(gh_read api graphql -f url="$parent" -f query='query($url: URI!) { resource(url: $url) { ... on Issue { id } } }' --jq .data.resource.id)
    cid=$(gh_read api graphql -f url="$child" -f query='query($url: URI!) { resource(url: $url) { ... on Issue { id } } }' --jq .data.resource.id)
    gh api graphql -f query='
      mutation($p: ID!, $c: ID!) {
        addSubIssue(input: {issueId: $p, subIssueId: $c}) { issue { url } }
      }' -f p="$pid" -f c="$cid" >/dev/null
    echo "added $child"
    ;;

  decisions)
    url="${1:?usage: board.sh decisions <epic-url> [--full]}"
    # Decision comments run to thousands of words; the headline is enough to know what was decided.
    limit=400
    [ "${2:-}" = --full ] && limit=1000000
    gh_read api graphql -f url="$url" -f query='
      query($url: URI!) {
        resource(url: $url) {
          ... on Issue {
            url
            comments(last: 100) { nodes { body url authorAssociation } }
            subIssues(first: 100) {
              nodes {
                url
                comments(last: 100) { nodes { body url authorAssociation } }
              }
            }
          }
        }
      }' --jq '.data.resource | ([.] + .subIssues.nodes)[]
        | .url as $issue
        | .comments.nodes[]
        | select('"$GQL_PERSON"')
        | select(.body | test("^\\s*(\\*\\*)?Decision:"))
        | "=== \($issue)\n\(.url)\n\(.body[0:'"$limit"'])\n"'
    ;;

  *) usage ;;
esac
