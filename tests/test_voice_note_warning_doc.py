"""Lint of the hermes-voice-note direction warning draft and its insert/strip script (#117)."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "hermes-voice-note-warning.md"
SCRIPT = REPO / "scripts" / "voice_note_warning.py"
DECISION = REPO / "docs" / "decisions" / "0013-voice-note-integration.md"

spec = importlib.util.spec_from_file_location("voice_note_warning", SCRIPT)
vnw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vnw)


def draft() -> str:
    return DOC.read_text(encoding="utf-8")


def section(title: str) -> str:
    text = draft()
    start = text.index(f"\n## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


@pytest.fixture(params=vnw.KINDS)
def block(request) -> str:
    return vnw.block(draft(), request.param)


def test_draft_exists_and_states_it_waits_for_the_owner() -> None:
    head = draft().split("\n## ", 1)[0]
    assert "черновик" in head.lower()
    assert "одобрения владельца" in head
    assert "#117" in head and "третий критерий" in head
    assert "ничего не опубликовано" in head


def test_each_block_has_one_start_and_one_end_marker_around_it(block: str) -> None:
    assert block.count(vnw.START) == 1 and block.count(vnw.END) == 1
    assert block.startswith(vnw.START) and block.endswith(vnw.END)


def test_blocks_name_the_old_direction_and_the_stock_sdk_requirement(block: str) -> None:
    flat = " ".join(block.split())
    for term in (
        "отдельн",
        "EXE",
        "второй профиль",
        "не целев",
        "штатном Hermes Desktop Plugin SDK",
        "штатных вызовах Desktop Plugin SDK",
        "host.retainSession",
        "https://github.com/renidere/sound-scout-plugin",
        "ничего не удаляется",
        "остаются доступны",
    ):
        assert term in flat, term
    links = re.findall(r"https://github\.com/renidere/sound-scout-plugin/blob/main/(\S+?)\)", block)
    assert links == ["docs/decisions/0013-voice-note-integration.md"]
    assert DECISION.is_file()


def test_blocks_give_no_install_or_connection_instruction_that_could_contradict_111(
    block: str,
) -> None:
    for command in ("hermes mcp", "hermes config", "plugins.enabled", "Install from Git", "готов"):
        assert command not in block, command


OLD_PLUGIN_FACT = "поэтому в штатном клиенте он не работает"
REQUIREMENT = (
    "Он должен работать в штатном Hermes Desktop только на штатных вызовах Desktop Plugin SDK"
)


def new_plugin_claims(flat: str) -> list[str]:
    """Claims that something already works or passed, except the fact about the old plugin."""
    rest = flat.replace(OLD_PLUGIN_FACT, "")
    return re.findall(r"\bработа(?:ет|ют)\b|провер\w*|принят\w*", rest, re.I)


def test_blocks_state_a_requirement_for_the_new_plugin_not_a_claim(block: str) -> None:
    flat = " ".join(block.split())
    assert flat.count(REQUIREMENT) == 1
    assert "без `host.retainSession`" in flat.split(REQUIREMENT, 1)[1]
    assert new_plugin_claims(flat) == []


def test_the_old_plugin_fact_stays_as_it_is() -> None:
    readme = " ".join(vnw.block(draft(), "readme").split())
    assert (
        "Плагин v0.4.0–v0.4.2 требует `host.retainSession`, которого нет в штатном Hermes Desktop "
        f"Plugin SDK, {OLD_PLUGIN_FACT}." in readme
    )


def test_the_target_line_of_the_draft_is_a_requirement() -> None:
    flat = " ".join(draft().split())
    target = "Целевое требование: плагин `sound-scout` должен работать в штатном Hermes Desktop"
    assert target in flat
    assert "плагин `sound-scout` работает" not in flat


@pytest.mark.parametrize(
    "claim", ["Он работает в штатном Hermes Desktop", "Плагин проверен", "Плагин принят"]
)
def test_a_claim_about_the_new_plugin_is_caught(block: str, claim: str) -> None:
    flat = " ".join(block.split()).replace(REQUIREMENT, claim)
    assert new_plugin_claims(flat) != []


def test_release_section_covers_both_releases() -> None:
    publish = section("Публикация (только после одобрения владельца)")
    assert "for TAG in v0.4.1 v0.4.2" in publish
    assert "--json body" in publish and "--notes-file" in publish
    assert "insert release" in publish and "insert readme" in publish


def test_publish_commands_guard_against_a_second_insertion_and_keep_a_backup() -> None:
    publish = section("Публикация (только после одобрения владельца)")
    assert publish.count('voice_note_warning.py" has README.md') == 1
    assert 'voice_note_warning.py has "$BK/$TAG.current.json"' in publish
    assert publish.index("has") < publish.index("insert")
    assert 'cp "$BK/$TAG.current.json" "$BK/$TAG.before.json"' in publish
    assert "git push --quiet origin HEAD:main" in publish
    assert "mktemp -d" in publish


def test_revert_commands_exist_for_readme_and_releases() -> None:
    revert = section("Откат")
    assert "strip README.md README.md" in revert
    assert "git revert --no-edit" in revert
    assert 'strip "$BK/$TAG.revert.json"' in revert
    assert "$TAG.before.json" in revert
    assert revert.count("gh release edit") == 2


def test_no_absolute_personal_paths() -> None:
    for path in (DOC, SCRIPT):
        assert "/home/" not in path.read_text(encoding="utf-8"), path


ORIGINALS = (
    "# Hermes Voice Note\n\nТекст.\n",
    "## Выпуск\r\n\r\nСтрока.\r\n",
    "\n\nНачало с пустых строк.",
    "",
)


@pytest.mark.parametrize("original", ORIGINALS)
def test_insert_then_strip_returns_the_original(block: str, original: str) -> None:
    published = vnw.insert(original, block)
    assert published.startswith(block)
    assert vnw.has(published)
    assert vnw.insert(published, block) == published
    assert vnw.strip(published) == original


@pytest.mark.parametrize("original", ORIGINALS)
def test_strip_of_a_text_without_the_block_changes_nothing(original: str) -> None:
    assert not vnw.has(original)
    assert vnw.strip(original) == original


@pytest.mark.parametrize("marker", [vnw.START, vnw.END])
def test_a_marker_without_a_whole_block_is_refused(block: str, marker: str) -> None:
    broken = f"{marker}\n# Hermes Voice Note\n"
    assert not vnw.has(broken)
    with pytest.raises(ValueError, match="no whole block"):
        vnw.strip(broken)
    with pytest.raises(ValueError, match="no whole block"):
        vnw.insert(broken, block)


def test_cli_refuses_a_marker_without_a_whole_block_and_writes_nothing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    original = f"{vnw.START}\n# Hermes Voice Note\n"
    readme.write_text(original, encoding="utf-8")
    out = tmp_path / "out.md"
    for args in (("strip", str(readme), str(out)), ("insert", "readme", str(readme), str(out))):
        refused = run(*args)
        assert refused.returncode == 2 and "no whole block" in refused.stderr
        assert not out.exists()
    assert run("has", str(readme)).returncode == 1
    assert readme.read_text(encoding="utf-8") == original


def test_draft_without_a_block_is_refused() -> None:
    with pytest.raises(ValueError, match="no readme-warning block"):
        vnw.block("# Пусто\n", "readme")
    broken = "```markdown readme-warning\nбез маркеров\n```\n"
    with pytest.raises(ValueError, match="not wrapped in the markers"):
        vnw.block(broken, "readme")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_cli_publishes_and_reverts_a_release_body_from_gh_json(tmp_path: Path) -> None:
    body = "## Готовый Windows-клиент\r\n\r\nОписание.\r\n"
    before = tmp_path / "v0.4.1.before.json"
    before.write_text(json.dumps({"body": body}), encoding="utf-8")
    after = tmp_path / "v0.4.1.after.md"

    assert run("has", str(before)).returncode == 1
    assert run("insert", "release", str(before), str(after)).returncode == 0
    published = after.read_bytes().decode("utf-8")
    assert published == vnw.insert(body, vnw.block(draft(), "release"))
    assert run("has", str(after)).returncode == 0

    again = run("insert", "release", str(after), str(after))
    assert again.returncode == 0 and "already present" in again.stderr
    assert after.read_bytes().decode("utf-8") == published

    reverted = tmp_path / "reverted.md"
    assert run("strip", str(after), str(reverted)).returncode == 0
    assert reverted.read_bytes().decode("utf-8") == body
    restored = tmp_path / "restored.md"
    assert run("strip", str(before), str(restored)).returncode == 0
    assert restored.read_bytes().decode("utf-8") == body


def test_cli_inserts_the_readme_block_in_place(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Hermes Voice Note\n", encoding="utf-8")
    assert run("insert", "readme", str(readme), str(readme)).returncode == 0
    text = readme.read_text(encoding="utf-8")
    assert text == vnw.block(draft(), "readme") + "\n\n# Hermes Voice Note\n"
    assert run("strip", str(readme), str(readme)).returncode == 0
    assert readme.read_text(encoding="utf-8") == "# Hermes Voice Note\n"


def decision_section(heading: str) -> str:
    text = DECISION.read_text(encoding="utf-8")
    start = re.search(rf"^#+ {re.escape(heading)}\. ", text, re.M).start()
    end = re.compile(r"^#+ \d", re.M).search(text, start + 1)
    return " ".join(text[start : end.start() if end else len(text)].split())


def test_criterion2_blocks_agree_with_decision_0013_and_the_claude_md_direction() -> None:
    facts = decision_section("2")
    for fact in (
        "host.retainSession",
        "v0.4.0",
        "неподписанным Windows EXE",
        "HERMES_HOME",
        "userData",
    ):
        assert fact in facts, fact
    target = decision_section("3.2")
    assert (
        "Отдельный EXE, форк Hermes, второй профиль и второй клиент не нужны и не допускаются"
        in target
    )
    recorder = decision_section("3.3")
    stock_path = ["profileRoutes", "retainProfile", "session.resume", "file.attach"]
    assert (
        "`profileRoutes` по захваченному owner → `retainProfile` → `session.resume` → `file.attach`"
        in recorder
    )
    direction = " ".join((REPO / "CLAUDE.md").read_text(encoding="utf-8").split())
    assert "рекордер Hermes Desktop на штатном Desktop Plugin SDK" in direction

    readme = " ".join(vnw.block(draft(), "readme").split())
    release = " ".join(vnw.block(draft(), "release").split())
    sdk_list = re.search(r"штатных вызовах Desktop Plugin SDK \(([^)]*)\)", readme)
    assert sdk_list is not None
    assert re.findall(r"`([^`]+)`", sdk_list.group(1)) == stock_path
    for flat in (readme, release):
        assert "отдельного EXE, форка Hermes, второго профиля и второго клиента" in flat
        assert "`host.retainSession`, которого нет в штатном Hermes Desktop Plugin SDK" in flat
        assert flat.count("host.retainSession") == 1 + flat.count("без `host.retainSession`")
        assert "HERMES_HOME" in flat and "userData" in flat
        assert "ещё в разработке" in flat
        assert "один плагин `sound-scout`" in flat
        for step in ("скачай", "запусти", "установи", "Install", "Start-Voice-Note.cmd`**"):
            assert step not in flat, step
    assert "неподписанный EXE" in release
    assert "Плагин v0.4.0–v0.4.2 требует `host.retainSession`" in readme


def sh_blocks(title: str) -> list[str]:
    return re.findall(r"^```sh\n(.*?)^```$", section(title), re.S | re.M)


def test_criterion2_publish_and_revert_commands_run_as_written_and_restore_byte_for_byte(
    tmp_path: Path,
) -> None:
    import os
    import shutil

    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("git and bash are needed")
    readme = "# Hermes Voice Note\n\nУстановка.\n"
    bodies = {
        "v0.4.1": "## Готовый Windows-клиент\n\nОписание.",
        "v0.4.2": "## Windows\r\n\r\nТекст.\r\n",
    }

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(tmp_path / "home"),
    }
    (tmp_path / "home").mkdir()

    def git(*args: str, cwd: Path = tmp_path) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
        ).stdout

    seed = tmp_path / "seed"
    seed.mkdir()
    git("init", "--quiet", "-b", "main", str(seed))
    (seed / "README.md").write_bytes(readme.encode("utf-8"))
    git("add", "README.md", cwd=seed)
    git("commit", "--quiet", "-m", "seed", cwd=seed)
    mirror = tmp_path / "mirror.git"
    git("clone", "--quiet", "--bare", str(seed), str(mirror))
    env |= {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{mirror.as_uri()}.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://github.com/renidere/hermes-voice-note",
    }

    state = tmp_path / "releases"
    state.mkdir()
    for tag, body in bodies.items():
        (state / tag).write_bytes(body.encode("utf-8"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        f"#!{sys.executable} -I\n"
        "import json, sys\n"
        f"state = {str(state)!r}\n"
        "a = sys.argv[1:]\n"
        "assert a[0] == 'release' and a[3:5] == ['-R', 'renidere/hermes-voice-note'], a\n"
        "path = state + '/' + a[2]\n"
        "if a[1] == 'view' and a[5:] == ['--json', 'body']:\n"
        "    print(json.dumps({'body': open(path, encoding='utf-8', newline='').read()}))\n"
        "elif a[1] == 'edit' and a[5] == '--notes-file' and len(a) == 7:\n"
        "    open(path, 'wb').write(open(a[6], 'rb').read())\n"
        "else:\n"
        "    sys.exit(f'unexpected gh call {a}')\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    checkout = tmp_path / "sound-scout-plugin"
    (checkout / "docs").mkdir(parents=True)
    (checkout / "scripts").mkdir()
    shutil.copy(DOC, checkout / "docs" / DOC.name)
    shutil.copy(SCRIPT, checkout / "scripts" / SCRIPT.name)

    publish = sh_blocks("Публикация (только после одобрения владельца)")
    revert = sh_blocks("Откат")
    assert len(publish) == 2 and len(revert) == 3

    def sh(script: str) -> str:
        done = subprocess.run(
            ["bash", "-c", script], cwd=checkout, env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
        return done.stdout

    def published_readme() -> str:
        return git("--git-dir", str(mirror), "show", "main:README.md")

    def release(tag: str) -> str:
        return (state / tag).read_bytes().decode("utf-8")

    for script in publish:
        sh(script)
    assert published_readme() == vnw.insert(readme, vnw.block(draft(), "readme"))
    for tag, body in bodies.items():
        assert release(tag) == vnw.insert(body, vnw.block(draft(), "release"))
    head = git("--git-dir", str(mirror), "rev-parse", "main")

    second = "".join(sh(script) for script in publish)
    assert second.count("the warning is already there, nothing to do") == 3
    assert git("--git-dir", str(mirror), "rev-parse", "main") == head
    for tag in bodies:
        assert release(tag).count(vnw.START) == 1

    sh(revert[0])
    sh(revert[1])
    assert published_readme().encode("utf-8") == readme.encode("utf-8")
    for tag, body in bodies.items():
        assert (state / tag).read_bytes() == body.encode("utf-8")
    again = sh(revert[0]) + sh(revert[1])
    assert again.count("no warning, nothing to do") == 3

    backup = checkout / "local" / "voice-note-warning"
    for tag, body in bodies.items():
        (state / tag).write_text("changed", encoding="utf-8")
        sh(revert[2].replace("TAG=v0.4.1", f"TAG={tag}", 1))
        assert (state / tag).read_bytes() == body.encode("utf-8")
        assert (
            json.loads((backup / f"{tag}.before.json").read_text(encoding="utf-8"))["body"] == body
        )
