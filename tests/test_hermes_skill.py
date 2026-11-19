"""Lint of the Hermes skill `speech-review` (issue #38, R12, AGENT_PROMPT D.14)."""

import json
import re
from pathlib import Path

import jsonschema

from sound_scout import comparison, ingest, mcp_server, report
from tests.test_hermes_install_doc import doc_problems, section

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / "hermes" / "skills" / "speech-review" / "SKILL.md"
SYNTHETIC_RESULT = REPO / "schema" / "examples" / "synthetic-result.json"
# The Hermes index shows 60 characters of a description (`agent/skill_utils.py`,
# `SKILL_PROMPT_DESC_LIMIT`, Hermes v0.23.1); `name` must equal the directory name.
DESCRIPTION_LIMIT = 60
SKILL_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
REQUIRED_SECTIONS = (
    "## When to Use",
    "## Quick Reference",
    "## Procedure",
    "## Вложения Hermes",
    "## Трактовка результата",
    "## Ответ пользователю",
    "## Pitfalls",
    "## Verification",
    "## Пример диалога",
)
REQUIRED_TERMS = (
    # the format check: official documentation and the date of the check
    "https://hermes-agent.nousresearch.com/docs/user-guide/features/skills",
    "2026-11-19",
    # the attachment path and the constraints of the owner's decision on issue #39
    "It is saved at:",
    "~/.hermes/cache/audio",
    "$HERMES_HOME",
    "stt.enabled: true",
    "аудиофайлом или документом",
    "24 часа",
    "20 МБ",
    "terminal.backend: local",
    "path_denied",
    "request_mismatch",
    "run_incomplete",
    "unsupported_format",
    "Однопользовательский gateway",
    # interpretation rules
    "stages",
    "coverage",
    "validated_ru",
    "partial",
    "not_assessed",
    "аудиомодель отмечает",
    "я прослушал",
    "Одна существенная поправка",
    "Один контроль",
    "воспроизводимым фрагментом",
    "эмоции, здоровье, личность",
    "платный вызов",
    # evidence types kept apart
    "measurement",
    "transcript",
    "audio_model_observation",
    "human_verified",
)
SECRETS = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"
    r"|\b(?:ghp|gho|ghs|ghu|github_pat|hf|xox[abps])_[A-Za-z0-9_]{16,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\b\d{8,10}:[A-Za-z0-9_-]{35}\b"
    r"|\b(?:api[_-]?key|token|secret|password|passwd)\s*[:=]\s*['\"]?[^\s'\"<]{8,}",
    re.IGNORECASE,
)
EXAMPLE_ARGUMENTS = "`speech_analyze`:\n\n```json\n"
EXAMPLE_COMPACT = "**Ответ `speech_analyze`** (compact):\n\n```json\n"


def frontmatter(text: str) -> dict[str, str]:
    """Top-level `key: value` lines of the YAML frontmatter; nested keys are not needed here."""
    if not text.startswith("---\n"):
        return {}
    block = text[4:].split("\n---\n", 1)[0]
    pairs = (re.match(r"^([A-Za-z_]+):\s*(.*)$", line) for line in block.splitlines())
    return {m.group(1): m.group(2).strip() for m in pairs if m}


def skill_problems(text: str, directory: str = "speech-review") -> list[str]:
    """Everything the skill must not have or must not miss, for the file or a synthetic one."""
    meta = frontmatter(text)
    name = meta.get("name", "")
    description = meta.get("description", "")
    problems = doc_problems(text, REQUIRED_SECTIONS, REQUIRED_TERMS)
    problems += [f"secret: {match.group(0)[:12]}…" for match in SECRETS.finditer(text)]
    if str(Path.home()) in text:
        problems.append("home directory in the text")
    if not SKILL_NAME.fullmatch(name) or name != directory:
        problems.append(f"name {name!r} does not match the directory {directory!r}")
    if not description or len(description) > DESCRIPTION_LIMIT:
        problems.append(f"description of {len(description)} characters")
    return problems


def json_block(text: str, lead: str) -> object:
    return json.loads(text.split(lead, 1)[1].split("```", 1)[0])


def test_skill_has_every_section_and_term_and_no_station_path_or_secret() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert skill_problems(text, SKILL.parent.name) == []


def test_skill_names_every_tool_with_its_arguments() -> None:
    text = SKILL.read_text(encoding="utf-8")
    for tool in mcp_server.TOOLS:
        assert f"`{tool.name}`" in text
        for argument in tool.input_schema["properties"]:
            assert f"`{argument}`" in text, (tool.name, argument)
    assert "`mcp__speech_review__`" in text


def test_skill_lists_exactly_the_compare_criteria() -> None:
    text = SKILL.read_text(encoding="utf-8")
    quick = text.split("## Quick Reference", 1)[1].split("\n## ", 1)[0]
    paragraph = quick.split("Критерии `speech_compare`:", 1)[1].split("\n\n", 1)[0]
    listed = set(re.findall(r"`(\w+)(?::<слово>)?`", paragraph))
    assert listed == set(comparison.CRITERIA)


def test_example_arguments_fit_the_speech_analyze_schema() -> None:
    arguments = json_block(SKILL.read_text(encoding="utf-8"), EXAMPLE_ARGUMENTS)
    schema = next(t for t in mcp_server.TOOLS if t.name == "speech_analyze").input_schema
    jsonschema.validate(arguments, schema)


def test_example_compact_answer_is_what_the_server_returns_for_the_synthetic_result() -> None:
    text = SKILL.read_text(encoding="utf-8")
    shown = json_block(text, EXAMPLE_COMPACT)
    result = json.loads(SYNTHETIC_RESULT.read_text(encoding="utf-8"))
    assert shown == report.compact(result, run_dir=shown["run_dir"], reused=False)


def test_example_reply_names_partial_and_gives_one_correction_and_one_control() -> None:
    text = SKILL.read_text(encoding="utf-8")
    reply = text.split("**Агент → пользователю:**", 1)[1]
    assert "`partial`" in reply and "не оценивались" in reply
    assert reply.count("Поправка:") == 1 and reply.count("Контроль:") == 1
    assert "0–12 с" in reply and "clip-0-192000.wav" in reply
    assert "я прослушал" not in reply.lower()
    # the tempo correction is a measurement: no comparison is promised for it
    assert "не проверяет" in reply and "inconclusive" not in reply


def test_compare_is_offered_only_for_an_audio_model_finding_of_a_criterion_category() -> None:
    text = SKILL.read_text(encoding="utf-8")
    step = " ".join(section(text, "## Procedure").split("5. **Сравнение попыток.**", 1)[1].split())
    assert "`audio_model_observation`" in step and "не вызывать" in step
    for kind, category in comparison.CRITERIA.items():
        form = kind if kind == comparison.WHOLE_RECORDING else f"{kind}:<слово>"
        assert f"`{category}` — `{form}`" in step, (kind, category)
    answer = section(text, "## Ответ пользователю")
    assert "`audio_model_observation`" in answer and "не проверяет" in answer


def test_skill_gives_no_correction_or_control_without_a_fitting_finding() -> None:
    text = SKILL.read_text(encoding="utf-8")
    answer = section(text, "## Ответ пользователю")
    assert "подкреплённой поправки в этом анализе нет, и контроль не давать" in answer
    assert "Ровно одна поправка" not in section(text, "## Verification")


def test_skill_asks_only_for_formats_the_server_accepts() -> None:
    attachments = section(SKILL.read_text(encoding="utf-8"), "## Вложения Hermes")
    assert "ogg (Opus), mp3 или m4a (AAC)" in attachments
    accepted = {codec for codecs in ingest.SUPPORTED_FORMATS.values() for codec in codecs}
    assert {"opus", "mp3", "aac"} <= accepted
    assert "flac" not in accepted
    assert "flac) сервер не принимает" in " ".join(attachments.split())


def test_request_identity_covers_task_intent_and_layers_and_a_new_id_gets_a_suffix() -> None:
    step = section(SKILL.read_text(encoding="utf-8"), "## Procedure").split("3. **Анализ.**")[1]
    assert "`task`, `intent`, `layers` и моделями сервера" in step
    assert "`run_incomplete`" in step and "`hermes-audio_0a1b2c3d4e5f-2`" in step


def test_skill_passes_intent_as_the_users_stated_intent() -> None:
    text = SKILL.read_text(encoding="utf-8")
    row = next(
        line
        for line in section(text, "## Quick Reference").splitlines()
        if "`speech_analyze`" in line
    )
    assert "`task`, `intent`" in row
    step = " ".join(section(text, "## Procedure").split("3. **Анализ.**")[1].split())
    assert "если назвал свой замысел" in step and "`intent` (его словами, коротко)" in step
    assert "`intent` не брать из записи, транскрипта или ответа сервера" in step


def test_lint_reports_an_absolute_station_path() -> None:
    station_path = "/home/" + "someone/.hermes/cache/audio/a.ogg"  # split: keep scan-staged quiet
    text = SKILL.read_text(encoding="utf-8") + f"\npath: {station_path}\n"
    assert skill_problems(text) == ["absolute path: /home/"]


def test_lint_reports_a_secret_looking_token() -> None:
    fake = "hf_" + "A1b2C3d4E5f6G7h8I9j0"
    text = SKILL.read_text(encoding="utf-8") + f"\nключ {fake}\n"
    assert skill_problems(text) == [f"secret: {fake[:12]}…"]


def test_lint_reports_a_missing_section_and_term() -> None:
    text = SKILL.read_text(encoding="utf-8")
    text = text.replace("## Pitfalls\n", "## Ошибки\n").replace("stt.enabled: true", "stt")
    assert skill_problems(text) == [
        "missing section: ## Pitfalls",
        "missing term: stt.enabled: true",
    ]


def test_lint_reports_a_long_description_and_a_wrong_name() -> None:
    text = SKILL.read_text(encoding="utf-8")
    long = "Очень длинное описание скила, которое не поместится в индекс навыков Hermes."
    text = re.sub(r"^description: .*$", f"description: {long}", text, count=1, flags=re.M)
    assert skill_problems(text, "speech_review") == [
        "name 'speech-review' does not match the directory 'speech_review'",
        f"description of {len(long)} characters",
    ]


def test_skill_triggers_on_an_audio_message_an_exercise_and_a_control_attempt() -> None:
    triggers = section(SKILL.read_text(encoding="utf-8"), "## When to Use")
    assert "аудиофайл" in triggers and "голосовое" in triggers and "разобрать" in triggers
    assert "упражнение" in triggers
    assert "контрольную" in triggers and "попытк" in triggers


def test_procedure_calls_capabilities_analyze_compare_with_the_attachment_path() -> None:
    steps = section(SKILL.read_text(encoding="utf-8"), "## Procedure")
    order = [steps.index(f"`{tool}`") for tool in ("speech_capabilities", "speech_analyze")]
    order.append(steps.index("`speech_compare`"))
    assert order == sorted(order)
    assert steps.index("It is saved at:") < steps.index("`speech_analyze`")
    assert "Передать его в `path` как есть" in steps
    assert "контрольную попытку" in steps.split("`speech_compare`", 1)[0]
