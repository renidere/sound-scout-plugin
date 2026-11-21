"""Lint of the Hermes connection instructions and the example entry (issue #39, R12, T09b)."""

import re
from pathlib import Path

import pytest

from sound_scout import cli, ingest, jobs, mcp_server
from sound_scout.config import Budgets

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "hermes-install.md"
EXAMPLE = REPO / "docs" / "examples" / "hermes-mcp-entry.example.yaml"
CONFIG_EXAMPLE = REPO / "sound-scout.example.toml"
REQUIRED_SECTIONS = (
    "## Источники",
    "## Установка на хосте backend",
    "## Ограничения вложений Hermes",
    "## Регистрация в Hermes",
    "## Настройка записи",
    "## Итоговая запись",
    "## Readback",
    "## Что считается подключённым",
)
# The constraints of Hermes attachments the owner's decision on issue #39 requires by name.
REQUIRED_TERMS = (
    "~/.hermes/cache/audio",
    "stt.enabled: true",
    "24 часа",
    "20 МБ",
    "terminal.backend: local",
    "Однопользовательский gateway",
    "hermes mcp add speech-review",
    "SOUND_SCOUT_CONFIG",
    "sampling.enabled false",
    "supports_parallel_tool_calls false",
    "tools.include",
    "timeout",
    "connect-timeout",
    "T09b",
)
# A path that starts at the filesystem root (after a space, quote, backtick, bracket or `=`) or
# at a Windows drive; `~/…`, `<placeholder>/…`, relative paths, URLs and the generic system
# program dirs `/usr/bin`, `/usr/local/bin` (the `env.PATH` of the entry, issue #77) do not match.
STATION_PATH = re.compile(
    r"(?:^|[\s\"'`(\[=])/(?!usr/(?:local/)?bin\b)[\w.-]+/|\b[A-Za-z]:\\", re.MULTILINE
)
TIMEOUT_MARGIN_SECONDS = 60


def doc_problems(text: str, sections: tuple[str, ...], terms: tuple[str, ...]) -> list[str]:
    """Everything wrong with a document: station paths, missing sections, missing terms."""
    headings = {line.strip() for line in text.splitlines() if line.startswith("#")}
    problems = [f"absolute path: {match.group(0).strip()}" for match in STATION_PATH.finditer(text)]
    problems += [f"missing section: {section}" for section in sections if section not in headings]
    problems += [f"missing term: {term}" for term in terms if term not in text]
    return problems


def test_install_doc_has_every_section_and_no_station_path() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert doc_problems(text, REQUIRED_SECTIONS, REQUIRED_TERMS) == []
    assert str(Path.home()) not in text


def test_example_entry_has_no_station_path() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    assert doc_problems(text, (), ()) == []
    assert str(Path.home()) not in text


def test_doc_lint_reports_an_absolute_path() -> None:
    text = "\n".join([*REQUIRED_SECTIONS, *REQUIRED_TERMS, "command: /home/someone/.local/bin/uv"])
    assert doc_problems(text, REQUIRED_SECTIONS, REQUIRED_TERMS) == ["absolute path: /home/"]


def test_doc_lint_reports_a_windows_path() -> None:
    assert doc_problems(r"путь `C:\Users\someone\a.wav`", (), ()) == ["absolute path: C:\\"]


def test_doc_lint_accepts_placeholders_home_relative_paths_and_urls() -> None:
    text = (
        "`~/.hermes/cache/audio`, `<домашний каталог>/.hermes/cache/audio`, "
        "`compare/<id>.json`, R09/R10, `/reload-mcp`, `/tools list`, "
        "https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference"
    )
    assert doc_problems(text, (), ()) == []


def test_doc_lint_reports_a_missing_section_and_term() -> None:
    text = "\n".join([*REQUIRED_SECTIONS[1:], *REQUIRED_TERMS[1:]])
    assert doc_problems(text, REQUIRED_SECTIONS, REQUIRED_TERMS) == [
        f"missing section: {REQUIRED_SECTIONS[0]}",
        f"missing term: {REQUIRED_TERMS[0]}",
    ]


def test_example_entry_allows_exactly_the_server_tools() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    include = text.split("include:", 1)[1]
    listed = re.findall(r"^\s+- (\S+)$", include, re.MULTILINE)
    assert listed == [tool.name for tool in mcp_server.TOOLS]


def test_example_entry_sets_sampling_parallelism_and_timeouts_explicitly() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    assert re.search(r"^    sampling:\n      enabled: false$", text, re.MULTILINE)
    assert re.search(r"^    supports_parallel_tool_calls: false$", text, re.MULTILINE)
    assert re.search(r"^    connect_timeout: \d+(\.\d+)?$", text, re.MULTILINE)
    assert re.search(r"^      SOUND_SCOUT_CONFIG: ", text, re.MULTILINE)


def test_example_timeout_leaves_a_margin_over_the_worker_deadline() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    timeout = int(re.search(r"^    timeout: (\d+)$", text, re.MULTILINE).group(1))
    budgets = Budgets()
    worst_call = budgets.worker_deadline_seconds + budgets.max_subprocess_seconds
    assert timeout >= worst_call + TIMEOUT_MARGIN_SECONDS
    assert f"hermes config set mcp_servers.speech-review.timeout {timeout}" in DOC.read_text(
        encoding="utf-8"
    )


def section(text: str, heading: str) -> str:
    """The body of a `## ` section up to the next `## ` heading."""
    body = text.split(f"\n{heading}\n", 1)[1]
    return body.split("\n## ", 1)[0]


def test_criterion1_mcp_add_passes_options_before_args_and_matches_the_example() -> None:
    text = DOC.read_text(encoding="utf-8")
    block = section(text, "## Регистрация в Hermes").split("```sh\n", 1)[1].split("```", 1)[0]
    tokens = block.replace("\\\n", " ").split()
    assert tokens[:4] == ["hermes", "mcp", "add", "speech-review"]
    options = [token for token in tokens if token.startswith("--")]
    args_at = tokens.index("--args")
    assert {"--command", "--connect-timeout", "--env"} <= set(options)
    assert all(
        tokens.index(option) < args_at for option in ("--command", "--connect-timeout", "--env")
    )
    example = EXAMPLE.read_text(encoding="utf-8")
    example_args = re.findall(
        r"^      - (.+)$", example.split("args:", 1)[1].split("env:", 1)[0], re.M
    )
    assert " ".join(tokens[args_at + 1 :]) == " ".join(example_args)


def test_criterion2_readback_reads_the_entry_checks_discovery_and_calls_from_a_new_session() -> (
    None
):
    readback = section(DOC.read_text(encoding="utf-8"), "## Readback")
    assert "hermes config get mcp_servers.speech-review" in readback
    assert "docs/examples/hermes-mcp-entry.example.yaml" in DOC.read_text(encoding="utf-8")
    assert "Discovery" in readback and "/tools list" in readback
    for tool in mcp_server.TOOLS:
        assert f"mcp__speech_review__{tool.name}" in readback
    assert "Реальный вызов из новой сессии" in readback
    real_call = readback.split("Реальный вызов из новой сессии", 1)[1]
    assert "speech_job_submit" in real_call and "speech_job_result" in real_call
    assert "sound-scout jobs start" in real_call
    assert "run_id" in real_call
    assert "через `speech_analyze`" not in " ".join(real_call.split())


def test_criterion3_connected_section_separates_sdk_call_from_native_exposure() -> None:
    connected = section(DOC.read_text(encoding="utf-8"), "## Что считается подключённым")
    assert "MCP SDK" in connected and "native tool exposure" in connected
    assert "T09a" in connected and "T09b" in connected
    checklist = re.findall(r"^- \[ \] ", connected, re.M)
    assert len(checklist) >= 5
    assert "issue #40" in connected


def test_doc_states_that_a_config_error_makes_doctor_exit_with_the_error_code() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "всегда завершается с кодом 0" not in text
    assert (
        f"`sound-scout doctor: {cli.CONFIG_ERROR}: …` в stderr и код {cli.ERROR_EXIT_CODE}" in text
    )


# Advice for recordings over ~2 minutes: «присылайте» up to `;` or `.` (issue #75, R02).
ADVICE = re.compile(r"присылайте ([^;.]+)")
ACCEPTED_ADVICE = "ogg (Opus), mp3 или m4a (AAC)"


def advised_format_problems(text: str) -> list[str]:
    """Every format name the advice to send recordings names that ingest does not accept."""
    accepted = {
        name
        for container, codecs in ingest.SUPPORTED_FORMATS.items()
        for name in (*container.split(","), *codecs)
    }
    clauses = ADVICE.findall(" ".join(text.split()))
    if not clauses:
        return ["no advice on formats to send"]
    names = [name for clause in clauses for name in re.findall(r"[a-z0-9]+", clause.lower())]
    return [f"not accepted: {name}" for name in names if name not in accepted]


@pytest.mark.parametrize("path", [DOC, CONFIG_EXAMPLE], ids=["install-doc", "config-example"])
def test_advice_names_only_formats_ingest_accepts(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert advised_format_problems(text) == []
    assert ACCEPTED_ADVICE in " ".join(ADVICE.findall(" ".join(text.split())))


def test_format_advice_check_rejects_flac_and_a_missing_advice() -> None:
    old_doc = "превышает предел примерно через 2 минуты — присылайте ogg/mp3/flac."
    old_config = "#   длиннее присылайте ogg/mp3/flac;"
    assert advised_format_problems(old_doc) == ["not accepted: flac"]
    assert advised_format_problems(old_config) == ["not accepted: flac"]
    assert advised_format_problems("несжатый WAV — около 2 минут") == [
        "no advice on formats to send"
    ]


DEPENDENCIES = REPO / "docs" / "dependencies.md"
ENTRY_PATH = re.compile(r"^      PATH: (\S+)$", re.MULTILINE)


def entry_path_problems(example: str) -> list[str]:
    """What is wrong with the `env.PATH` of the example entry (issue #77, R09, R11)."""
    env = example.split("\n    env:\n", 1)[1].split("\n    connect_timeout:", 1)[0]
    found = ENTRY_PATH.findall(env)
    if not found:
        return ["no env.PATH"]
    dirs = found[0].split(":")
    problems = [f"Hermes dir: {d}" for d in dirs if ".hermes" in d]
    problems += [f"not absolute: {d}" for d in dirs if not d.startswith("/")]
    return problems


def test_criterion77_example_entry_sets_env_path_to_system_dirs_without_hermes() -> None:
    example = EXAMPLE.read_text(encoding="utf-8")
    assert entry_path_problems(example) == []
    assert ".hermes" not in ENTRY_PATH.search(example).group(1)


def test_criterion77_example_yaml_parses_with_env_path_of_absolute_dirs_without_hermes() -> None:
    yaml = pytest.importorskip("yaml")
    entry = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))["mcp_servers"]["speech-review"]
    dirs = entry["env"]["PATH"].split(":")
    assert "/usr/bin" in dirs
    assert [d for d in dirs if not d.startswith("/") or ".hermes" in d] == []
    assert "SOUND_SCOUT_CONFIG" in entry["env"]


def test_entry_path_check_rejects_a_missing_path_and_a_hermes_dir() -> None:
    example = EXAMPLE.read_text(encoding="utf-8")
    without = ENTRY_PATH.sub("", example)
    assert entry_path_problems(without) == ["no env.PATH"]
    hermes = ENTRY_PATH.sub(
        "      PATH: ~/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin:/usr/bin", example
    )
    assert entry_path_problems(hermes) == [
        "Hermes dir: ~/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin",
        "not absolute: ~/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin",
    ]


def test_criterion77_mcp_add_passes_the_same_path_and_the_doc_explains_why() -> None:
    text = DOC.read_text(encoding="utf-8")
    entry_path = ENTRY_PATH.search(EXAMPLE.read_text(encoding="utf-8")).group(1)
    block = section(text, "## Регистрация в Hermes").split("```sh\n", 1)[1].split("```", 1)[0]
    tokens = block.replace("\\\n", " ").split()
    env_values = tokens[tokens.index("--env") + 1 : tokens.index("--args")]
    assert f"PATH={entry_path}" in env_values
    assert f"hermes config set mcp_servers.speech-review.env.PATH {entry_path}" in text
    registration = " ".join(section(text, "## Регистрация в Hermes").split())
    assert "~/.hermes/tools" in registration and "shutil.which" in registration
    assert "decoder: unvalidated" in registration


def test_criterion77_readback_wants_decoder_ready_under_hermes_as_in_doctor() -> None:
    readback = " ".join(section(DOC.read_text(encoding="utf-8"), "## Readback").split())
    assert "speech_capabilities" in readback and "sound-scout doctor" in readback
    assert "`decoder` под Hermes тоже `ready`" in readback


def test_criterion77_dependencies_say_ffmpeg_is_found_on_path() -> None:
    programs = section(DEPENDENCIES.read_text(encoding="utf-8"), "## 2. Системные программы")
    programs = " ".join(programs.split())
    assert "ищутся через `PATH`" in programs and "`env.PATH`" in programs


def test_doc_lint_allows_only_the_generic_system_program_dirs() -> None:
    allowed = "`PATH=/usr/local/bin:/usr/bin:/bin`, обычно `/usr/bin`"
    assert doc_problems(allowed, (), ()) == []
    assert doc_problems("PATH=/opt/ffmpeg/bin", (), ()) == ["absolute path: =/opt/"]


def test_update_section_says_to_restart_the_jobs_runner_after_an_update() -> None:
    update = " ".join(section(DOC.read_text(encoding="utf-8"), "## Обновление").split())
    assert "перезапустите runner" in update
    assert "uv sync --frozen" in update
    assert f"systemctl --user stop {jobs.RUNNER_UNIT}" in update
    assert "sound-scout jobs start" in update and "sound-scout jobs recover" in update
    assert "Restart=no" in update and "--collect" in update
