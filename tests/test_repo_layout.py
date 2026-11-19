"""Acceptance checks of issue #9 that are about repository files, not the CLI."""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".ogg",
    ".oga",
    ".opus",
    ".m4a",
    ".aac",
    ".flac",
    ".webm",
    ".mp4",
    ".mov",
}


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )


def git_ignores(path: str) -> bool:
    result = git("check-ignore", "-q", path)
    if result.returncode > 1:
        pytest.skip(f"git is unavailable here: {result.stderr.strip()}")
    return result.returncode == 0


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_pins_python_313_and_names_package_and_script(pyproject: dict) -> None:
    assert pyproject["project"]["requires-python"] == ">=3.13,<3.14"
    assert pyproject["project"]["name"] in {"sound-scout", "sound_scout"}
    assert pyproject["project"]["scripts"]["sound-scout"] == "sound_scout.cli:main"
    assert (REPO_ROOT / "src" / "sound_scout" / "__init__.py").is_file()


def test_uv_lock_exists_and_is_not_git_ignored() -> None:
    assert (REPO_ROOT / "uv.lock").is_file()
    assert not git_ignores("uv.lock")


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_uv_lock_matches_pyproject() -> None:
    result = subprocess.run(
        ["uv", "lock", "--check", "--offline"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_example_toml_has_commented_out_fields_and_local_toml_is_ignored() -> None:
    text = (REPO_ROOT / "sound-scout.example.toml").read_text(encoding="utf-8")
    assert tomllib.loads(text) == {}, "example config must be fully commented out"
    for field in ("allowed_roots", "data_root", "budgets", "backend"):
        assert re.search(rf"^#\s*\[?{field}\b", text, re.MULTILINE), field
    assert git_ignores("sound-scout.toml")
    assert not git_ignores("sound-scout.example.toml")


def test_lefthook_pre_commit_runs_ruff_on_staged_files() -> None:
    text = (REPO_ROOT / "lefthook.yml").read_text(encoding="utf-8")
    pre_commit = text.split("pre-commit:", 1)[1]
    assert re.search(r"ruff check\s+\{staged_files\}", pre_commit)
    assert re.search(r"ruff format --check\s+\{staged_files\}", pre_commit)


def test_readme_describes_installation_and_checks() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for fragment in (
        "uv sync",
        "FFmpeg",
        "lefthook install",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pytest",
    ):
        assert fragment in readme, fragment


GATE0_DECISION = "`docs/decisions/0007-gate0-result.md`"
OWNER_CHOICE_ISSUE = "#68"


def flat(text: str) -> str:
    return " ".join(text.split())


def test_readme_status_says_gemma_failed_gate0_and_no_backend_is_chosen() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    status = flat(readme.split("Статус:", 1)[1].split("\n\n", 1)[0])
    for fragment in (
        "Audio backend сейчас не выбран (`backend = none`)",
        "Gemma 4 E4B-it",
        "llama.cpp",
        "только локально",
        "`docs/decisions/0001-audio-backend.md`",
        "Gate 0 не прошла — вердикт NOT READY",
        GATE0_DECISION,
        OWNER_CHOICE_ISSUE,
        "Русская квалификация (Gate 1) не пройдена",
        "подключение к Hermes не выполнено",
        "инструмент не считается готовым",
    ):
        assert fragment in status, fragment
    assert "validated_ru" not in readme
    assert "`ready`" not in status
    assert "слыш" not in status.lower()
    assert not re.search(r"/home/|/Users/|[A-Z]:\\", readme)


def test_claude_md_stack_line_says_no_backend_is_chosen_after_gate0() -> None:
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    stack = flat(claude_md.split("- **Стек:**", 1)[1].split("\n- **", 1)[0])
    assert "пока Gate 0 не скажет иное" not in claude_md
    for fragment in (
        "Backend слухового анализа сейчас не выбран (`backend = none`)",
        "Gemma 4 E4B-it",
        "не прошла Gate 0 — вердикт NOT READY",
        GATE0_DECISION,
        "замену выбирает владелец",
        OWNER_CHOICE_ISSUE,
    ):
        assert fragment in stack, fragment


def test_dependencies_backend_intro_records_gate0_verdict_and_keeps_weights() -> None:
    dependencies = (REPO_ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8")
    section = dependencies.split("## 3. Audio-understanding backend", 1)[1].split("\n## ", 1)[0]
    intro = flat(section.split("### 3.1.", 1)[0])
    for fragment in (
        "**Gemma 4 E4B-it** в GGUF через llama.cpp, только локально",
        "вердиктом **NOT READY**",
        "`docs/decisions/0007-gate0-result.md` §5",
        "backend не упакован, `backend` остаётся `none`",
        OWNER_CHOICE_ISSUE,
    ):
        assert fragment in intro, fragment
    assert "### 3.1. Веса" in section and "SHA-256" in section


CANDIDATE_CHOSEN = re.compile(
    r"выбран[аоы]?[^.]*(?:Borealis|GigaChat|Qwen)|(?:Borealis|GigaChat|Qwen)[^.]*выбран"
)


def test_backend_docs_name_no_replacement_candidate_as_chosen() -> None:
    for name in ("CLAUDE.md", "README.md", "docs/dependencies.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "пока Gate 0 не скажет иное" not in text, name
        assert not re.search(r"/home/|/Users/|[A-Z]:\\", text), name
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    dependencies = (REPO_ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8")
    backend_paragraphs = (
        claude_md.split("- **Стек:**", 1)[1].split("\n- **", 1)[0],
        readme.split("Статус:", 1)[1].split("\n\n", 1)[0],
        dependencies.split("## 3. Audio-understanding backend", 1)[1].split("### 3.1.", 1)[0],
    )
    for paragraph in backend_paragraphs:
        assert "Gemma 4 E4B-it" in paragraph
        assert re.search(r"backend`? (?:= |остаётся `)none", paragraph), paragraph
        assert not re.search(r"\bready\b|validated_ru", paragraph), paragraph
        assert not re.search(CANDIDATE_CHOSEN, paragraph), paragraph
    verdict = (REPO_ROOT / "docs" / "decisions" / "0007-gate0-result.md").read_text("utf-8")
    assert "**Вердикт: NOT READY**" in verdict


def test_readme_backend_link_resolves_to_owner_decision() -> None:
    decision = REPO_ROOT / "docs" / "decisions" / "0001-audio-backend.md"
    assert decision.is_file()
    assert "Gemma 4 E4B-it — **выбрана**" in decision.read_text(encoding="utf-8")


def test_readme_makes_no_claim_that_a_model_hears_speech() -> None:
    readme = " ".join((REPO_ROOT / "README.md").read_text(encoding="utf-8").split())
    lowered = readme.lower()
    for phrase in ("слышит русскую речь", "оценивает произношение"):
        assert phrase not in lowered, phrase
    negated_hearing = re.compile(r"\bне\s+(?:\S+\s+){0,4}?\S*слыш|не доказывает, что \S+ слыш")
    sentences = [s for s in re.split(r"(?<=[.:;])\s+", lowered) if "слыш" in s]
    for sentence in sentences:
        assert negated_hearing.search(sentence), sentence


def test_conftest_synthesizes_wav_and_no_media_files_in_git(wav_file: Path) -> None:
    assert (REPO_ROOT / "tests" / "conftest.py").is_file()
    assert wav_file.is_file() and not wav_file.is_relative_to(REPO_ROOT)
    tracked = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if tracked.returncode != 0:
        pytest.skip(f"git is unavailable here: {tracked.stderr.strip()}")
    media = [p for p in tracked.stdout.split("\0") if Path(p).suffix.lower() in MEDIA_SUFFIXES]
    assert media == []
    for suffix in sorted(MEDIA_SUFFIXES):
        assert git_ignores(f"anything{suffix}"), suffix
