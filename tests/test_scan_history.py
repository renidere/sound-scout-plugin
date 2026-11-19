"""Acceptance checks of issue #41: scripts/scan-history.sh finds private data anywhere in history.

The "secrets" and private paths below are assembled at run time, so none of them is in git.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_HISTORY = REPO_ROOT / "scripts" / "scan-history.sh"
SCAN_STAGED = REPO_ROOT / ".claude" / "skills" / "work-task" / "scripts" / "scan-staged.sh"
AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"
HF_TOKEN = "hf_" + "a1B2c3D4e5" * 3 + "f6G7"
PEM_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"
PRIVATE_HOME = "/ho" + "me/alice/"


class Repo:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir()
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def commit(self, files: dict[str, str | bytes], message: str = "change") -> str:
        for name, content in files.items():
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
            self.git("add", "-f", "--", name)
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "--short=12", "HEAD")

    def scan(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCAN_HISTORY), *args],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    made = Repo(tmp_path / "repo")
    made.commit(
        {
            "README.md": "# demo\n",
            ".env.example": "TOKEN=\n",
            "tests/test_paths.py": 'ROOT = "/home/someone/rec"\nOTHER = "/home/username/x"\n',
        },
        "init",
    )
    return made


def test_clean_history_with_placeholder_home_paths_is_ok(repo: Repo) -> None:
    done = repo.scan()
    assert (done.returncode, done.stdout) == (0, "ok\n")


def test_media_added_and_removed_later_is_still_found(repo: Repo) -> None:
    added = repo.commit({"audio/take.WAV": b"RIFF", "notes/voice.aiff": b"FORM"})
    repo.git("rm", "-q", "audio/take.WAV", "notes/voice.aiff")
    repo.git("commit", "-q", "-m", "remove")
    done = repo.scan()
    assert done.returncode == 1
    assert f"BLOCK: {added} audio/take.WAV is a media file" in done.stdout
    assert f"BLOCK: {added} notes/voice.aiff is a media file" in done.stdout


def test_a_file_renamed_onto_a_media_name_is_found(repo: Repo) -> None:
    repo.commit({"clip.txt": "plain\n"})
    repo.git("mv", "clip.txt", "clip.mp3")
    repo.git("commit", "-q", "-m", "rename")
    renamed = repo.git("rev-parse", "--short=12", "HEAD")
    done = repo.scan()
    assert done.returncode == 1
    assert done.stdout.splitlines() == [f"BLOCK: {renamed} clip.mp3 is a media file"]


@pytest.mark.parametrize(
    ("name", "why"),
    [
        (".env", "is an env file"),
        ("deploy/.env.local", "is an env file"),
        ("sound-scout.toml", "is the operator's local config"),
        ("data/dev/notes.txt", "should not be committed"),
        ("local/HANDOFF.md", "should not be committed"),
        ("runs/x/result.json", "should not be committed"),
        ("keys/server.pem", "should not be committed"),
    ],
)
def test_forbidden_paths_are_found(repo: Repo, name: str, why: str) -> None:
    commit = repo.commit({name: "x\n"})
    done = repo.scan()
    assert done.returncode == 1
    assert done.stdout.splitlines() == [f"BLOCK: {commit} {name} {why}"]


@pytest.mark.parametrize("secret", [AWS_KEY, HF_TOKEN, PEM_HEADER], ids=["aws", "hf", "pem"])
def test_a_key_is_found_by_commit_file_and_line_without_its_value(repo: Repo, secret: str) -> None:
    commit = repo.commit({"src/settings.py": f"A = 1\nB = 2\nKEY = '{secret}'\n"})
    repo.commit({"src/settings.py": "A = 1\nB = 2\n"}, "drop the key")
    done = repo.scan()
    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        f"BLOCK: {commit} src/settings.py:3 looks like a secret or a private local path"
    ]
    assert secret not in done.stdout + done.stderr


def test_a_private_home_path_is_found_and_placeholders_are_not(repo: Repo) -> None:
    text = f"see /home/runner/work\nrec = '{PRIVATE_HOME}Music/take.ogg'\n"
    commit = repo.commit({"docs/notes.md": text})
    done = repo.scan()
    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        f"BLOCK: {commit} docs/notes.md:2 holds a /home/<user>/ path"
    ]
    assert "alice" not in done.stdout + done.stderr


def test_an_older_revision_is_scanned_without_the_later_commits(repo: Repo) -> None:
    clean = repo.git("rev-parse", "HEAD")
    repo.commit({"take.flac": b"fLaC"})
    assert repo.scan(clean).stdout == "ok\n"
    assert repo.scan().returncode == 1


def test_an_unknown_revision_is_refused(repo: Repo) -> None:
    done = repo.scan("no-such-rev")
    assert done.returncode == 2
    assert "not a commit: no-such-rev" in done.stderr


def _patterns(script: Path) -> str:
    match = re.search(r"^patterns='.*'$", script.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"no patterns line in {script.name}"
    return match.group(0)


def _case_items(script: Path, marker: str) -> set[str]:
    line = next(line for line in script.read_text(encoding="utf-8").splitlines() if marker in line)
    return {item.strip().lower() for item in line.split(")")[0].split("|")}


def test_rules_do_not_drift_from_scan_staged() -> None:
    assert _patterns(SCAN_HISTORY) == _patterns(SCAN_STAGED)
    staged_media = _case_items(SCAN_STAGED, "*.wav|")
    history_media = _case_items(SCAN_HISTORY, "*.wav|")
    assert staged_media <= history_media
    assert {"*.aif", "*.aiff", "*.wma", "*.mkv", "*.avi", "*.3gp", "*.amr"} <= history_media
    not_for_history = {"docs/spec/*", "htmlcov/*", "coverage/*"}
    staged_paths = _case_items(SCAN_STAGED, "data/*|") - not_for_history
    assert staged_paths <= _case_items(SCAN_HISTORY, "data/*|")


def test_history_scan_finds_no_private_data_in_this_repository() -> None:
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if shallow == "true":
        pytest.skip("a shallow checkout has no history to scan")
    done = subprocess.run(
        ["bash", str(SCAN_HISTORY)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300
    )
    assert (done.returncode, done.stdout) == (0, "ok\n"), done.stdout
