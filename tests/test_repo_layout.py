"""Acceptance checks of issue #9 that are about repository files, not the CLI."""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from sound_scout import mcp_server

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
CLOSED_CHOICE_ISSUE = "[#68](https://github.com/renidere/sound-scout-plugin/issues/68) (закрыта)"
NEXT_EXPERIMENT_ISSUE = "[#119](https://github.com/renidere/sound-scout-plugin/issues/119)"
CONNECT_BACKEND_ISSUE = "[#90](https://github.com/renidere/sound-scout-plugin/issues/90"
ISSUE_40 = "[#40](https://github.com/renidere/sound-scout-plugin/issues/40)"


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
        CLOSED_CHOICE_ISSUE,
        NEXT_EXPERIMENT_ISSUE,
        CONNECT_BACKEND_ISSUE,
        "Русская квалификация (Gate 1) не пройдена",
        "подключён к Hermes 2026-11-19",
        f"{ISSUE_40}, закрыта",
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
    r"выбран[аоы]?[^.]*(?:Borealis|GigaChat|Qwen|MOSS)|(?:Borealis|GigaChat|Qwen|MOSS)[^.]*выбран"
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


def test_backend_status_records_that_borealis_failed_gate0_too() -> None:
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    dependencies = (REPO_ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8")
    for paragraph in (
        claude_md.split("- **Стек:**", 1)[1].split("\n- **", 1)[0],
        readme.split("Статус:", 1)[1].split("\n\n", 1)[0],
        dependencies.split("## 3. Audio-understanding backend", 1)[1].split("### 3.1.", 1)[0],
    ):
        text = flat(paragraph)
        assert "Borealis-5b-it" in text and "NOT READY" in text, text
        assert "`docs/decisions/0010-gate0-borealis.md`" in text, text
    verdict = (REPO_ROOT / "docs" / "decisions" / "0010-gate0-borealis.md").read_text("utf-8")
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


CORPUS_CODE = (
    "scripts/fetch_corpus_sources.py",
    "src/sound_scout/eval/corpus.py",
    "tests/test_eval_corpus.py",
    "tests/test_fetch_corpus_sources.py",
)
CORPUS_DATA_SUFFIXES = {".zip", ".eaf", ".pdf", ".textgrid", ".pitchtier", ".f32"}


def test_criterion_5_corpus_scripts_and_tests_are_in_git_and_no_corpus_data_is() -> None:
    """#78 criterion 5: preparation code and its tests are tracked; audio, sources, labels not."""
    for path in CORPUS_CODE:
        assert (REPO_ROOT / path).is_file(), path
        assert not git_ignores(path), path
    for path in (
        "data/corpus/sources/rinco/RINCO_Russian_Intonation_Corpus-v01.zip",
        "data/corpus/pilot-1/manifest.json",
        "data/corpus/pilot-1/provenance.json",
        "data/corpus/pilot-1/clips/ctl-silence.wav",
        "data/corpus/pilot-1/labels/ctl-silence.json",
    ):
        assert git_ignores(path), path
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if listed.returncode != 0:
        pytest.skip(f"git is unavailable here: {listed.stderr.strip()}")
    paths = [Path(p) for p in listed.stdout.split("\0") if p]
    assert [p for p in paths if p.parts[0] == "data"] == []
    assert [p for p in paths if "labels" in p.parts[:-1]] == []
    assert [p for p in paths if p.suffix.lower() in CORPUS_DATA_SUFFIXES] == []


SYNTHETIC_OPENSTT_CODE = (
    "src/sound_scout/eval/synthetic.py",
    "src/sound_scout/eval/openstt.py",
    "tests/test_eval_synthetic.py",
    "tests/test_eval_openstt.py",
)


def test_criterion_5_of_81_tests_synthesize_audio_and_no_tts_or_openstt_data_is_in_git() -> None:
    """#81 criterion 5: TTS and Open STT outputs, the voice and the archive stay out of git."""
    for path in SYNTHETIC_OPENSTT_CODE:
        assert (REPO_ROOT / path).is_file(), path
        assert not git_ignores(path), path
    for name in SYNTHETIC_OPENSTT_CODE[2:]:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "pilot-1" not in text and "openstt-pool" not in text, name
    for path in (
        "data/corpus/sources/piper-denis/ru_RU-denis-medium.onnx",
        "data/corpus/sources/open-stt/asr_calls_2_val.tar.gz",
        "data/corpus/pilot-1/clips/syn-stress-zamok.wav",
        "data/corpus/pilot-1/synthetic/syn-stress-zamok.json",
        "data/corpus/pilot-1/labels/syn-stress-zamok.json",
        "data/corpus/pilot-1/openstt-pool/0123456789ab.wav",
        "data/corpus/pilot-1/openstt-selection.json",
        "data/corpus/pilot-1/asr-review/ostt-0123456789ab.json",
    ):
        assert git_ignores(path), path
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if listed.returncode != 0:
        pytest.skip(f"git is unavailable here: {listed.stderr.strip()}")
    paths = [Path(p) for p in listed.stdout.split("\0") if p]
    assert [p for p in paths if p.parts[0] == "data"] == []
    assert [p for p in paths if p.suffix.lower() in {".onnx", ".gz", ".tar"}] == []
    assert [p for p in paths if p.suffix.lower() in MEDIA_SUFFIXES] == []
    data_dirs = {"labels", "synthetic", "asr-review", "openstt-pool"}
    assert [p for p in paths if data_dirs & set(p.parts[:-1])] == []


def test_backend_status_records_that_gigachat_failed_gate0_on_p3() -> None:
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    dependencies = (REPO_ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8")
    for paragraph in (
        claude_md.split("- **Стек:**", 1)[1].split("\n- **", 1)[0],
        readme.split("Статус:", 1)[1].split("\n\n", 1)[0],
        dependencies.split("## 3. Audio-understanding backend", 1)[1].split("### 3.1.", 1)[0],
    ):
        text = flat(paragraph)
        assert "GigaChat3.1-Audio-10B-A1.8B" in text and "NOT READY" in text, text
        assert "`docs/decisions/0011-gate0-gigachat-audio.md`" in text, text
        assert "P3" in text and "о слухе модели" in text, text
    verdict = (REPO_ROOT / "docs" / "decisions" / "0011-gate0-gigachat-audio.md").read_text("utf-8")
    assert "**Вердикт: NOT READY**" in verdict


ISSUE = "https://github.com/renidere/sound-scout-plugin/issues/{}"
STALE_HERMES_CLAIM = re.compile(
    r"подключени\w* к Hermes не (?:выполнен|сделан)|Hermes (?:ещё |пока )?не подключ"
)


def stale_hermes_claims(text: str) -> list[str]:
    return STALE_HERMES_CLAIM.findall(flat(text))


def read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def test_docs_name_the_hermes_connection_of_issue_40_as_done() -> None:
    """#111 criterion 1: no document still says that Hermes is not connected."""
    for name in ("README.md", "docs/hermes-install.md", "docs/ACCEPTANCE.md"):
        assert stale_hermes_claims(read(name)) == [], name
    status = flat(read("README.md").split("Статус:", 1)[1].split("\n\n", 1)[0])
    assert "native вызовы из новой сессии `hermes -z` и из Hermes Desktop вернули `partial`" in (
        status
    )
    assert "Подключение на другом хосте — шаг владельца (`docs/hermes-install.md`)" in status


def test_stale_hermes_claim_check_reports_the_old_readme_line() -> None:
    old = "Русская квалификация не пройдена, подключение к Hermes не выполнено — пока это так"
    assert stale_hermes_claims(old) == ["подключение к Hermes не выполнен"]
    assert stale_hermes_claims("Hermes пока не подключён.") == ["Hermes пока не подключ"]
    assert stale_hermes_claims("Сервер подключён к Hermes 2026-11-19.") == []


def test_readme_shows_the_unified_plugin_direction_and_what_main_has() -> None:
    """#111 criterion 1: the direction of decision 0013 and the parts that are on main."""
    readme = read("README.md")
    direction = flat(readme.split("Направление (с 20 ноября 2026):", 1)[1].split("\n\n", 1)[0])
    for fragment in (
        "`docs/decisions/0013-voice-note-integration.md`",
        f"сначала Linux-часть ([#114]({ISSUE.format(114)}))",
        f"затем Windows-клиент (эпик [#115]({ISSUE.format(115)}))",
        "`sound-scout jobs`",
        "`docs/decisions/0018-job-modes-and-retry.md`",
        "`hermes/plugin/sound-scout/`",
        "`GET /health`, `POST /jobs`, `GET /jobs/<job_id>`",
        "`/api/plugins/sound-scout/`",
        "одной заметки с итогом задания в привязанный чат",
        "`hermes/skills/speech-review/`",
        f"([#104]({ISSUE.format(104)}))",
        f"([#103]({ISSUE.format(103)}))",
    ):
        assert fragment in direction, fragment
    for path in (
        "src/sound_scout/jobs.py",
        "docs/mcp.md",
        "hermes/plugin/sound-scout/delivery.py",
        "hermes/skills/speech-review/SKILL.md",
    ):
        assert (REPO_ROOT / path).is_file(), path
    assert "Решения по направлению — 0013, 0015–0018" in direction
    for number in (13, 15, 16, 17, 18):
        assert len(list((REPO_ROOT / "docs" / "decisions").glob(f"00{number}-*.md"))) == 1
    routes = read("hermes/plugin/sound-scout/dashboard/plugin_api.py")
    for route in (
        '@router.get("/health")',
        '@router.post("/jobs")',
        '@router.get("/jobs/{job_id}")',
    ):
        assert route in routes, route
    intro = flat(readme.split("Статус:", 1)[0])
    names = [tool.name for tool in mcp_server.TOOLS]
    assert re.findall(r"`(speech_\w+)`", intro) == names


def test_readme_and_acceptance_record_four_backend_candidates_without_ready() -> None:
    moss = "MOSS-Audio-8B-Thinking-ru проверен диагностическим прогоном, это не формальный Gate 0"
    for name in ("README.md", "docs/ACCEPTANCE.md"):
        text = flat(read(name))
        for candidate in (
            "Gemma 4 E4B-it",
            "Borealis-5b-it",
            "GigaChat3.1-Audio-10B-A1.8B",
            moss,
            "READY не получен",
            "issues/90#issuecomment-7024154195",
            f"[#119]({ISSUE.format(119)})",
        ):
            assert candidate in text, (name, candidate)


TOOL_COUNTS = {3: "трёх", 6: "шести", 9: "девяти"}


def test_hermes_install_keeps_connection_an_owner_step_and_counts_the_tools() -> None:
    """#111 criterion 2 and the allowlist size of the nine MCP tools."""
    doc = flat(read("docs/hermes-install.md"))
    for fragment in (
        "**Подключение к действующему Hermes — шаг владельца по отдельному разрешению**",
        "Агенты эти команды не выполняют",
        "На станции разработки сервер подключён 2026-11-19 по версии этой инструкции для трёх "
        "инструментов R12 (issue #40, закрыта)",
        "Инструменты заданий попадут в эту запись, только когда владелец повторит команду "
        "`tools.include`",
        "подключение на другом хосте — снова шаг владельца по отдельному разрешению",
    ):
        assert fragment in doc, fragment
    names = [tool.name for tool in mcp_server.TOOLS]
    jobs = [name for name in names if name.startswith("speech_job_")]
    total, sync = TOOL_COUNTS[len(names)], TOOL_COUNTS[len(names) - len(jobs)]
    assert f"allowlist из {sync} инструментов R12 и {TOOL_COUNTS[len(jobs)]} инструментов" in doc
    assert doc.count(f"из {total} инструментов") == 2
    assert "ровно девять инструментов" in doc
    assert not re.search(r"\b(?:семи|семь|пяти|пять) инструмент", doc)


def test_acceptance_limits_t09b_to_the_sync_call_and_lists_a01_a14_as_not_passed() -> None:
    """#111 criterion 3: T09b is the synchronous MCP call; A01–A14 are new and not run."""
    text = read("docs/ACCEPTANCE.md")
    section = flat(text.split("## Приёмка единого плагина (A01–A14)", 1)[1].split("\n## ", 1)[0])
    for fragment in (
        "**T09b покрывает только синхронный native вызов MCP.**",
        "**A01–A14 — новые пункты приёмки**, не часть T01–T10",
        "ни один сценарий не выполнялся, значит ни один не пройден",
        f"[#112]({ISSUE.format(112)})",
        f"[#114]({ISSUE.format(114)})",
        f"[#115]({ISSUE.format(115)})",
        "моки не доказывают пользовательский путь",
    ):
        assert fragment in section, fragment
    named = re.findall(r"`(tests/test_\w+\.py)`", section)
    assert {"tests/test_mcp_server.py", "tests/test_staging.py"} <= set(named)
    for test in named:
        assert (REPO_ROOT / test).is_file(), test
    t09b = next(line for line in text.splitlines() if line.startswith("| Hermes (T09b) |"))
    assert "T09b покрывает только синхронный native вызов MCP" in t09b
    assert "A01–A14 пройдены" not in flat(text)
