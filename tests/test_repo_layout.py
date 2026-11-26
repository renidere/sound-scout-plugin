"""Acceptance checks of issue #9 that are about repository files, not the CLI."""

import argparse
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from sound_scout import mcp_server
from sound_scout.cli import build_parser

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
CLOSED_CHOICE_ISSUE = "[#68](https://github.com/renidere/sound-scout-plugin/issues/68) (закрыта)"
ISSUE_40 = "[#40](https://github.com/renidere/sound-scout-plugin/issues/40)"
STALE_OWNER_CHOICE = re.compile(
    r"замену выбирает владелец[^.]*#68|#68[^.]*замену выбирает владелец"
)


def flat(text: str) -> str:
    return " ".join(text.split())


def readme_section(heading: str) -> str:
    """The body of the README section `## <heading>` up to the next `## ` heading."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


WORKS_NOW = "Что работает сейчас"
INTEGRATION = "Интеграция с другими харнесами"
DEPENDENCIES = "Зависимости"
MODELS = "Модели на стенде"
REPLACEMENTS = "Чем можно заменить"


def stale_owner_choice(text: str) -> list[str]:
    return STALE_OWNER_CHOICE.findall(flat(text))


def open_issue_68_mentions(text: str) -> int:
    text = flat(text)
    return text.count("#68") - text.count(CLOSED_CHOICE_ISSUE)


def test_readme_status_says_gemma_failed_gate0_and_no_backend_is_chosen() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    status = flat(readme_section(WORKS_NOW) + readme_section(MODELS))
    for fragment in (
        "Инструмент не готов",
        "русская квалификация (Gate 1) не пройдена",
        "Audio backend не подключён (`backend = none`)",
        "Gemma 4 E4B-it",
        GATE0_DECISION,
    ):
        assert fragment in status, fragment
    assert "validated_ru" not in readme
    assert "`ready`" not in status
    assert "слыш" not in status.lower()
    assert not re.search(r"/home/|/Users/|[A-Z]:\\", readme)


CANDIDATE_CHOSEN = re.compile(
    r"выбран[аоы]?[^.]*(?:Borealis|GigaChat|Qwen|MOSS)|(?:Borealis|GigaChat|Qwen|MOSS)[^.]*выбран"
)


def test_backend_docs_name_no_replacement_candidate_as_chosen() -> None:
    for name in ("CLAUDE.md", "README.md", "docs/dependencies.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "пока Gate 0 не скажет иное" not in text, name
        assert not re.search(r"/home/|/Users/|[A-Z]:\\", text), name
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    dependencies = (REPO_ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8")
    backend_paragraphs = (
        claude_md.split("- **Стек:**", 1)[1].split("\n- **", 1)[0],
        readme_section(MODELS),
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
    dependencies = (REPO_ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8")
    for paragraph in (
        claude_md.split("- **Стек:**", 1)[1].split("\n- **", 1)[0],
        readme_section(MODELS),
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
    dependencies = (REPO_ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8")
    for paragraph in (
        claude_md.split("- **Стек:**", 1)[1].split("\n- **", 1)[0],
        readme_section(MODELS),
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
    status = flat(readme_section(WORKS_NOW))
    assert f"подключён к Hermes 2026-11-19 ({ISSUE_40})" in status


def test_stale_hermes_claim_check_reports_the_old_readme_line() -> None:
    old = "Русская квалификация не пройдена, подключение к Hermes не выполнено — пока это так"
    assert stale_hermes_claims(old) == ["подключение к Hermes не выполнен"]
    assert stale_hermes_claims("Hermes пока не подключён.") == ["Hermes пока не подключ"]
    assert stale_hermes_claims("Сервер подключён к Hermes 2026-11-19.") == []


README_PATH = re.compile(r"`((?:src|docs|hermes|tests|scripts)/[^`\s]+)`")


def readme_paths(text: str) -> list[str]:
    return [path.rstrip("/") for path in README_PATH.findall(text)]


def test_readme_what_works_now_links_existing_code() -> None:
    """#191: each part that works links its code, and every linked path exists."""
    section = readme_section(WORKS_NOW)
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]
    for row in rows[1:]:
        assert readme_paths(row), row
    for path in readme_paths(section):
        assert (REPO_ROOT / path).exists(), path


INSTALL_AND_RUN = "Установка и запуск"
README_JOBS_USAGE = re.compile(r"^uv run sound-scout jobs ([\w|-]+)", re.MULTILINE)


def subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    (action,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return action.choices


def test_readme_shows_a_usage_line_of_each_jobs_subcommand() -> None:
    """#195: a new `jobs` subcommand without a README usage line fails here."""
    in_parser = set(subcommands(subcommands(build_parser())["jobs"]))
    section = readme_section(INSTALL_AND_RUN)
    in_readme = {name for group in README_JOBS_USAGE.findall(section) for name in group.split("|")}
    assert in_readme == in_parser


def test_readme_install_and_run_links_existing_decisions() -> None:
    section = readme_section(INSTALL_AND_RUN)
    decisions = [path for path in readme_paths(section) if path.startswith("docs/decisions/")]
    for name in ("0013-", "0025-", "0030-"):
        assert any(name in path for path in decisions), name
    for path in decisions:
        assert (REPO_ROOT / path).is_file(), path


def test_readme_integration_marks_untested_hosts_and_hermes_only_parts() -> None:
    """#191: other MCP hosts are marked as not checked; the chat note is Hermes-only."""
    section = readme_section(INTEGRATION)
    text = flat(section)
    for fragment in ("`sound-scout serve`", "только Hermes", "`allowed_roots`", "`data_root`"):
        assert fragment in text, fragment
    for path in readme_paths(section):
        assert (REPO_ROOT / path).exists(), path


def test_readme_dependencies_point_to_the_pins_and_do_not_copy_them() -> None:
    section = readme_section(DEPENDENCIES)
    assert "`docs/dependencies.md`" in section
    for name in ("FFmpeg", "`faster-whisper`", "`onnxruntime`", "`praat-parselmouth`", "`mcp`"):
        assert name in section, name
    assert not re.search(r"\b[0-9a-f]{40,64}\b", section)


def test_readme_models_match_the_pinned_asr_and_the_gpu_reference_run() -> None:
    from sound_scout import config

    section = flat(readme_section(MODELS))
    dependencies = flat(read("docs/dependencies.md").split("## 4.", 1)[1].split("\n## ", 1)[0])
    model = config.Asr().model
    revision = re.search(rf"`{re.escape(model)}`, ревизия `([0-9a-f]{{40}})`", dependencies)
    assert revision, dependencies
    assert f"`{model}` @ `{revision[1]}`" in section
    assert "Silero VAD" in section and "нет (`backend = none`)" in section
    for path in readme_paths(readme_section(MODELS)):
        assert (REPO_ROOT / path).exists(), path


def test_readme_replacement_candidates_are_unchecked_and_each_has_a_source() -> None:
    section = readme_section(REPLACEMENTS)
    text = flat(section)
    for fragment in ("не проверены", "решения владельца", "`docs/dependencies.md`", "Gate 0"):
        assert fragment in text, fragment
    assert not re.search(CANDIDATE_CHOSEN, text)
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and "---" not in line and not line.startswith("| Роль")
    ]
    assert len(rows) >= 4
    for row in rows:
        assert re.search(r"\| https://\S+ \|$", row), row


def test_readme_and_acceptance_record_four_backend_candidates_without_ready() -> None:
    moss = "MOSS-Audio-8B-Thinking-ru проверен диагностическим прогоном, это не формальный Gate 0"
    names = ("Gemma 4 E4B-it", "Borealis-5b-it", "GigaChat3.1-Audio-10B-A1.8B")
    readme = flat(readme_section(MODELS))
    for candidate in (
        *names,
        "MOSS-Audio-8B-Thinking-ru, диагностический прогон",
        "issues/90#issuecomment-7024154195",
    ):
        assert candidate in readme, candidate
    acceptance = flat(read("docs/ACCEPTANCE.md"))
    for candidate in (
        *names,
        moss,
        "READY не получен",
        "issues/90#issuecomment-7024154195",
        f"[#119]({ISSUE.format(119)})",
    ):
        assert candidate in acceptance, candidate


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
    """#111 criterion 3: T09b is the synchronous MCP call; A01–A14 are new and live in their own
    table (#112), which is not all passed."""
    text = read("docs/ACCEPTANCE.md")
    section = flat(text.split("## Приёмка единого плагина (A01–A14)", 1)[1].split("\n## ", 1)[0])
    for fragment in (
        "**T09b покрывает только синхронный native вызов MCP.**",
        "**A01–A14 — новые пункты приёмки**, не часть T01–T10",
        "`docs/acceptance/A01-A14.md`",
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


def test_spec_readme_marks_its_17_november_status_as_history_and_links_current_docs() -> None:
    spec_readme = REPO_ROOT / "docs/spec/README.md"
    text = spec_readme.read_text(encoding="utf-8")
    assert "Историческая пометка" in text
    assert "Исторический статус на 17 ноября 2026 года" in text
    assert "Исторический перечень на 17 ноября 2026 года" in text
    # The original wording stays as history under the marks.
    assert "Репозиторий не создан, audio backend не выбран" in text
    assert "1. Явное поручение на реализацию" in text
    for link in re.findall(r"\]\(([^)#]+)\)", text):
        assert (spec_readme.parent / link).resolve().is_file(), link
    assert "../decisions/0013-voice-note-integration.md" in text
    assert "../../README.md" in text
    assert "../ACCEPTANCE.md" in text


def test_claude_md_asr_row_names_the_vad_signal_and_both_decisions() -> None:
    """#138: the asr.py row names `no_vad_speech`, links 0005 and 0021, and the VAD-less status."""
    from sound_scout import asr

    row = next(
        line
        for line in read("CLAUDE.md").splitlines()
        if line.rstrip().endswith("| `src/sound_scout/asr.py` |")
    )
    for fragment in (
        "сигнал `no_vad_speech`",
        "`MIN_VAD_SPEECH_S`",
        "`speech_spans` в `asr/raw.json`",
        "без VAD этап `asr` — `unavailable` с `VAD_UNAVAILABLE_REASON`",
    ):
        assert fragment in row, fragment
    old, new = "docs/decisions/0005-asr-silence-hallucination.md", "0021-asr-vad-hallucination.md"
    links = re.findall(r"`(docs/decisions/[^`]+\.md)`", row)
    assert links == [old, f"docs/decisions/{new}"]
    for link in links:
        assert (REPO_ROOT / link).is_file(), link
    assert "no_vad_speech" in asr.HALLUCINATION_SIGNALS
    assert asr.VAD_UNAVAILABLE_REASON and asr.MIN_VAD_SPEECH_S > 0
    decision = read(f"docs/decisions/{new}")
    assert "`asr` — `unavailable` с причиной `VAD_UNAVAILABLE_REASON`" in flat(decision)
