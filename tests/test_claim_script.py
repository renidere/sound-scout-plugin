"""Issue #72: `claim.sh acquire` leaves no claim of its own behind after a `gh` failure.

claim.sh runs against a fake `gh` on PATH that keeps the issue's comments in a JSON file; no
network, and the real `gh` is never reached."""

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAIM = REPO_ROOT / ".claude" / "skills" / "work-task" / "scripts" / "claim.sh"
URL = "https://github.com/owner/repo/issues/7"
MARK = "<!-- work-task-claim id="
ORPHAN = 5

# Вызов метода номер n ведёт себя по <METHOD>_MODES[n]: ok — как GitHub; fail — сбой сети до
# сервера; lost — запрос выполнен, ответ потерян. Вывод GET повторяет фильтр `claims()`.
FAKE_GH = r"""
import json, os, re, sys, time
from datetime import datetime, timezone

args = sys.argv[1:]
method = args[args.index("-X") + 1] if "-X" in args else "GET"
path = next(a for a in args if a.startswith("repos/"))
with open(os.environ["CALLS"], "a") as calls:
    calls.write(method + " " + path + "\n")
counts_file = os.environ["COUNTS"]
counts = json.load(open(counts_file)) if os.path.exists(counts_file) else {}
counts[method] = counts.get(method, 0) + 1
json.dump(counts, open(counts_file, "w"))
modes = os.environ.get(method + "_MODES", "").split()
mode = modes[counts[method] - 1] if counts[method] <= len(modes) else "ok"
if mode == "fail":
    sys.exit("gh: connection reset by peer")

state = os.environ["COMMENTS"]
comments = json.load(open(state))
stamp = "%Y-%m-%dT%H:%M:%SZ"
out = ""
if method == "GET":
    trusted = ("OWNER", "COLLABORATOR", "MEMBER")
    for c in comments:
        if c["author_association"] in trusted and c["body"].startswith(os.environ["MARK"]):
            claim = re.search(r"id=([^ ]+) -->", c["body"]).group(1)
            started = re.search(r"started ([^,]+),", c["body"])
            updated = datetime.strptime(c["updated_at"], stamp).replace(tzinfo=timezone.utc)
            minutes = int((time.time() - updated.timestamp()) // 60)
            row = [c["id"], claim, minutes, started.group(1) if started else ""]
            out += "\t".join(map(str, row)) + "\n"
elif method == "POST":
    body = next(a for a in args if a.startswith("body="))[5:]
    new_id = max([c["id"] for c in comments], default=1000) + 1
    now = datetime.now(timezone.utc).strftime(stamp)
    comments.append(
        {"id": new_id, "body": body, "author_association": "OWNER", "updated_at": now}
    )
    out = f"{new_id}\n"
elif method == "DELETE":
    cid = int(path.rsplit("/", 1)[1])
    if not any(c["id"] == cid for c in comments):
        sys.exit("gh: Not Found (HTTP 404)")
    comments = [c for c in comments if c["id"] != cid]
json.dump(comments, open(state, "w"))
if mode == "lost":
    sys.exit("gh: unexpected EOF")
sys.stdout.write(out)
"""


def iso(minutes_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def claim_comment(cid: int, claim: str, minutes_ago: int, author: str = "OWNER") -> dict:
    body = f"{MARK}{claim} -->\nclaim `{claim}`, started {iso(minutes_ago)}, last seen now.\n"
    return {"id": cid, "body": body, "author_association": author, "updated_at": iso(minutes_ago)}


def run_claim(
    tmp_path: Path, command: str, *args: str, comments: list[dict] | None = None, **modes: str
) -> tuple[subprocess.CompletedProcess, list[dict], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "gh"
    fake.write_text(f"#!{sys.executable}" + FAKE_GH)
    fake.chmod(0o755)
    state = tmp_path / "comments.json"
    if not state.exists():
        state.write_text(json.dumps(comments or []))
    calls = tmp_path / "calls"
    path = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    assert shutil.which("gh", path=path) == str(fake)
    env = {
        **os.environ,
        "PATH": path,
        "GH_TOKEN": "fake",
        "MARK": MARK,
        "COMMENTS": str(state),
        "CALLS": str(calls),
        "COUNTS": str(tmp_path / "counts.json"),
        "CLAIM_SETTLE_SECONDS": "0",
        "CLAIM_RETRY_SECONDS": "0",
        **modes,
    }
    done = subprocess.run(
        ["bash", str(CLAIM), command, URL, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    left = json.loads(state.read_text())
    return done, left, calls.read_text().splitlines() if calls.exists() else []


def acquire(tmp_path: Path, **kwargs) -> tuple[subprocess.CompletedProcess, list[dict], list[str]]:
    return run_claim(tmp_path, "acquire", **kwargs)


def test_acquire_on_a_free_issue_posts_one_claim_and_removes_stale_ones(tmp_path: Path) -> None:
    stale = claim_comment(1001, "old-1-1", minutes_ago=400)
    stranger = claim_comment(1002, "fake-1-1", minutes_ago=1, author="NONE")
    done, left, _ = acquire(tmp_path, comments=[stale, stranger])
    assert done.returncode == 0, done.stderr
    word, claim = done.stdout.split()
    assert done.stdout == f"acquired {claim}\n"
    assert word == "acquired"
    assert [c["id"] for c in left] == [1002, 1003]
    assert left[1]["body"].startswith(f"{MARK}{claim} -->")


def test_acquire_on_a_live_claim_says_taken_and_posts_nothing(tmp_path: Path) -> None:
    holder = claim_comment(1001, "other-1-1", minutes_ago=3)
    done, left, calls = acquire(tmp_path, comments=[holder])
    assert done.returncode == 3
    assert done.stdout == "taken other-1-1 3\n"
    assert left == [holder]
    assert not any(c.startswith("POST") for c in calls)


def test_acquire_losing_a_race_deletes_its_claim_and_says_taken(tmp_path: Path) -> None:
    # Первый GET пуст, второй видит более ранний захват, который появился одновременно.
    state = tmp_path / "comments.json"
    state.write_text("[]")
    early = claim_comment(900, "rival-1-1", minutes_ago=0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_python = sys.executable
    (bin_dir / "sleep").write_text(
        f"#!{real_python}\nimport json, os\n"
        f"p = os.environ['COMMENTS']\nc = json.load(open(p))\n"
        f"json.dump([{json.dumps(early)}] + c, open(p, 'w'))\n"
    )
    (bin_dir / "sleep").chmod(0o755)
    done, left, _ = acquire(tmp_path)
    assert done.returncode == 3, done.stderr
    assert done.stdout == "taken rival-1-1 0\n"
    assert [c["id"] for c in left] == [900]


def test_losing_a_race_with_delete_down_reports_the_orphan(tmp_path: Path) -> None:
    early = claim_comment(900, "rival-1-1", minutes_ago=0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sleep").write_text(
        f"#!{sys.executable}\nimport json, os\n"
        f"p = os.environ['COMMENTS']\nc = json.load(open(p))\n"
        f"if all(x['id'] != 900 for x in c):\n"
        f"    json.dump([{json.dumps(early)}] + c, open(p, 'w'))\n"
    )
    (bin_dir / "sleep").chmod(0o755)
    done, left, _ = acquire(tmp_path, comments=[], DELETE_MODES="fail fail fail")
    assert done.returncode == ORPHAN, done.stderr
    word, claim, comment = done.stdout.split()
    assert word == "orphan"
    assert [c["id"] for c in left] == [900, int(comment)]
    assert left[1]["body"].startswith(f"{MARK}{claim} -->")


def test_failing_first_get_is_an_ordinary_error_distinct_from_taken_and_orphan(
    tmp_path: Path,
) -> None:
    done, left, calls = acquire(tmp_path, GET_MODES="fail")
    assert done.returncode == 1
    assert len({done.returncode, 3, ORPHAN}) == 3
    assert done.stdout == ""
    assert left == []
    assert not any(c.startswith("POST") for c in calls)


def test_get_failing_after_post_with_delete_down_reports_the_orphan(tmp_path: Path) -> None:
    done, left, calls = acquire(tmp_path, GET_MODES="ok fail", DELETE_MODES="fail fail fail")
    assert done.returncode == ORPHAN
    word, claim, comment = done.stdout.split()
    assert word == "orphan"
    assert [c["id"] for c in left] == [int(comment)]
    assert left[0]["body"].startswith(f"{MARK}{claim} -->")
    assert sum(c.startswith("DELETE") for c in calls) == 3


def test_get_failing_after_post_deletes_the_claim_and_fails_plainly(tmp_path: Path) -> None:
    done, left, calls = acquire(tmp_path, GET_MODES="ok fail", DELETE_MODES="fail")
    assert done.returncode == 1
    assert done.stdout == ""
    assert left == []
    assert sum(c.startswith("DELETE") for c in calls) == 2


def test_lost_post_response_finds_the_claim_by_its_id_and_deletes_it(tmp_path: Path) -> None:
    done, left, calls = acquire(tmp_path, POST_MODES="lost")
    assert done.returncode == 1
    assert done.stdout == ""
    assert "was posted and removed" in done.stderr
    assert left == []
    assert calls[-2:] == [
        "GET repos/owner/repo/issues/7/comments",
        "DELETE repos/owner/repo/issues/comments/1001",
    ]


def test_lost_post_response_deletes_only_its_own_claim(tmp_path: Path) -> None:
    stale = claim_comment(1001, "old-1-1", minutes_ago=400)
    done, left, calls = acquire(tmp_path, comments=[stale], POST_MODES="lost")
    assert done.returncode == 1
    assert done.stdout == ""
    assert left == [stale]
    assert [c for c in calls if c.startswith("DELETE")] == [
        "DELETE repos/owner/repo/issues/comments/1002"
    ]


def test_lost_post_response_with_delete_down_reports_the_found_comment(tmp_path: Path) -> None:
    done, left, _ = acquire(tmp_path, POST_MODES="lost", DELETE_MODES="fail fail fail")
    assert done.returncode == ORPHAN
    word, claim, comment = done.stdout.split()
    assert (word, comment) == ("orphan", "1001")
    assert left[0]["body"].startswith(f"{MARK}{claim} -->")


def test_failed_post_that_left_nothing_fails_plainly(tmp_path: Path) -> None:
    done, left, _ = acquire(tmp_path, POST_MODES="fail")
    assert done.returncode == 1
    assert done.stdout == ""
    assert "was not posted" in done.stderr
    assert left == []


def test_failed_post_that_cannot_be_verified_reports_an_unknown_orphan(tmp_path: Path) -> None:
    done, _, calls = acquire(tmp_path, POST_MODES="lost", GET_MODES="ok fail fail fail")
    assert done.returncode == ORPHAN
    word, _claim, comment = done.stdout.split()
    assert (word, comment) == ("orphan", "-")
    assert sum(c.startswith("GET") for c in calls) == 4


def test_release_removes_the_orphan_acquire_reported(tmp_path: Path) -> None:
    done, _, _ = acquire(tmp_path, GET_MODES="ok fail", DELETE_MODES="fail fail fail")
    _, claim, _ = done.stdout.split()
    released, left, _ = run_claim(tmp_path, "release", claim)
    assert released.returncode == 0
    assert released.stdout == "released\n"
    assert left == []
