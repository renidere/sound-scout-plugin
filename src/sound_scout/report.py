"""The report layer (R08, R11, spec § 1 and § 4): `result.json`, `report.md` and the compact answer.

`write` takes a result document the server assembled, keeps private values out of it, checks it
against the contract and the run's files (`findings.validate`) and writes the full `result.json`
and a Russian `report.md` into the run directory, both created exclusively. Every finding goes
into the report: the three-item limit of the compact answer never hides the rest (spec § 1).

Private values never reach either file or the compact answer (R09, R10): `source.path` is
dropped; the data root, the run directory, the allowed roots, the operator config path, the home
directory and the user name are replaced by placeholders in every string; any other absolute
path is replaced too. The contents of `sound-scout.toml` have no field in the contract, and the
paths it holds are among the replaced values.

`compact` is the tool response for Hermes: statuses, coverage, up to three findings in priority
order, limitations and relative links to `report.md`, `result.json` and clips, at most
`COMPACT_MAX_BYTES` of UTF-8 JSON, never PCM, base64 or full diagnostics (R08).

Priority (docs/decisions/0004-report-priority.md): first the impact on intelligibility, then the
confidence of the evidence, then time. Impact is structural, never guessed from the wording: a
negative finding before a neutral one before a positive one, then the category in
`INTELLIGIBILITY_ORDER`. Confidence is the evidence basis in `CONFIDENCE_ORDER`: a human label
before an audio model observation before a measurement before the transcript; the model's
verbal confidence never becomes a number (R05).

The phrase «аудиомодель отмечает» is used only for audio model evidence of a completed audio
review; without it the report says `partial` and `not_assessed` with the reasons the result
carries (R11). An ASR segment flagged `suspect_hallucination` is listed as an interval with its
signals, never with its text (docs/decisions/0005). `cost` builds the contract's `cost`: zero
only for a local run, `unknown` for an external provider without data — a zero cloud cost is
never made up (§ 4).
"""

import contextlib
import copy
import getpass
import json
import os
import re
from pathlib import Path

from sound_scout import config as config_module
from sound_scout import findings, paths, schema
from sound_scout.config import Config
from sound_scout.runs import Run

RESULT_FILE_NAME = "result.json"
REPORT_FILE_NAME = "report.md"
COMPACT_MAX_BYTES = 4096
COMPACT_FINDINGS = 3
COMPACT_RANGES = 4
COMPACT_LIST_ITEMS = 5
LOCAL_PROVIDER = "local"
LOCAL_COST_NOTE = "local: локальная обработка, внешних вызовов и списаний нет"
UNKNOWN_COST_NOTE = "стоимость внешнего вызова неизвестна: данных провайдера нет"
AUDIO_MODEL_PHRASE = "Аудиомодель отмечает"
POLARITY_ORDER = ("negative", "neutral", "positive")
# Direct damage to understanding first: an unclear word, an unfinished phrase, a repeat, a broken
# phrase, a misplaced stress; then what only measurements or the transcript show.
INTELLIGIBILITY_ORDER = (
    "pronunciation",
    "completeness",
    "repetition_self_correction",
    "pause_phrasing",
    "semantic_stress",
    "tempo",
    "pause_length",
    "level",
    "pitch_range",
    "filler_words",
    "content",
)
SUSPECT_SEGMENTS_LEAD = (
    "Сегменты ASR с пометкой `suspect_hallucination` (`uncertain`): их текст не считается "
    "произнесённым и в отчёт не выводится."
)
CONFIDENCE_ORDER = ("human_verified", "audio_model_observation", "measurement", "transcript")
PRIVATE_PATH, PRIVATE_USER = findings.REDACTIONS
_TEXT_LIMITS = (400, 200, 100, 40)
# Two or more segments, the first with an ASCII letter or digit and each with a word character:
# a path, never the phoneme notation of an observation (/ш/, /ть/, (/о/), /р/-/л/).
_ABSOLUTE_PATHS = re.compile(
    r"(?<![\w.:/\\~-])/[^\s/]*[A-Za-z0-9][^\s/]*(?:/[^\s/]*\w[^\s/]*)+"
    r"|(?<![\w/])~/\S*|(?<![\w])[A-Za-z]:\\\S*"
)
_REPORT_KINDS = {"result_json": RESULT_FILE_NAME, "report_md": REPORT_FILE_NAME}
_STAGE_LABELS = {
    "prepare": "подготовка",
    "asr": "ASR",
    "acoustics": "акустика",
    "audio_review": "audio review",
    "report": "отчёт",
}
_STATUS_LABELS = {
    "complete": "запрошенные слои и охват выполнены; это не доказательство безошибочности модели",
    "partial": "частичный результат: выполнены не все запрошенные слои или не весь охват",
    "blocked": "анализ заблокирован",
    "failed": "анализ завершился ошибкой",
}
_BASIS_LEADS = {
    "measurement": "Измерение показывает",
    "transcript": "По транскрипту ASR",
    "audio_model_observation": AUDIO_MODEL_PHRASE,
    "human_verified": "Человеческая разметка",
}
_ASSESSMENT_LABELS = {"diction": "дикция", "prosody": "просодия", "content": "содержание"}


class ReportError(Exception):
    """The report cannot be written for this run and result."""


def write(run: Run, result: dict, *, config: Config) -> dict:
    """Write `result.json` and `report.md` into the run directory; return the written document.

    Raises `schema.ResultValidationError` for a document the contract or the run's files reject,
    `ReportError` for a result of another run, `paths.PathDenied` for a run outside the data root
    and `FileExistsError` when the run already holds its report.
    """
    schema.validate(result)
    if result["run_id"] != run.run_id:
        raise ReportError(f"result belongs to run {result['run_id']!r}, not {run.run_id!r}")
    document = _private_values_removed(result, _private_values(run, config))
    document["artifacts"] = [a for a in document["artifacts"] if a["kind"] not in _REPORT_KINDS] + [
        _report_artifact(kind, name) for kind, name in _REPORT_KINDS.items()
    ]
    findings.validate(document, run=run)
    markdown = render(document)
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    created: list[Path] = []
    try:
        for name, text in ((RESULT_FILE_NAME, payload), (REPORT_FILE_NAME, markdown)):
            with paths.open_output(run.path, name, config) as out:
                created.append(run.path / name)
                out.write(text.encode("utf-8"))
    except BaseException:
        for path in reversed(created):
            with contextlib.suppress(OSError):
                os.unlink(path)
        raise
    return document


def compact(
    result: dict,
    *,
    run_dir: str | None = None,
    reused: bool | None = None,
    head: dict | None = None,
) -> dict:
    """The tool response: at most `COMPACT_MAX_BYTES` of UTF-8 JSON, links relative to the run.

    `head`, fields of the caller such as the job of `jobs.result`, then `run_dir`, the run
    directory relative to the data root, and `reused`, whether the result is the one of an
    earlier request with the same `request_id` and input (R11), go first and count in the limit:
    `analyze --json` and `speech_analyze` answer with the same document.
    """
    document = _private_values_removed(result, [])
    head = dict(head or {})
    if run_dir is not None:
        head["run_dir"] = run_dir
    if reused is not None:
        head["reused"] = reused
    for limit in _TEXT_LIMITS:
        answer = head | _compact(document, limit)
        if len(json.dumps(answer, ensure_ascii=False).encode("utf-8")) <= COMPACT_MAX_BYTES:
            return answer
    raise ValueError(f"compact answer does not fit in {COMPACT_MAX_BYTES} bytes")


def prioritized(items: list[dict]) -> list[dict]:
    """Findings by impact on intelligibility, then confidence of the evidence, then time."""

    def rank(finding: dict) -> tuple:
        return (
            _position(POLARITY_ORDER, finding["polarity"]),
            _position(INTELLIGIBILITY_ORDER, finding["category"]),
            _position(CONFIDENCE_ORDER, finding["evidence_basis"]),
            finding["start"],
            finding["id"],
        )

    return sorted(items, key=rank)


def cost(
    provider: str | None = None,
    *,
    amount: float | None = None,
    currency: str | None = None,
    estimated: bool = False,
) -> dict:
    """The contract's `cost`: `provider=None` is a local run, zero and marked local; an external
    provider without an amount is `unknown`; an amount needs its ISO currency."""
    if provider is None:
        return {
            "kind": "actual",
            "amount": 0,
            "currency": None,
            "provider": LOCAL_PROVIDER,
            "note": LOCAL_COST_NOTE,
        }
    if amount is None:
        return {
            "kind": "unknown",
            "amount": None,
            "currency": None,
            "provider": provider,
            "note": UNKNOWN_COST_NOTE,
        }
    if currency is None or not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError(f"an external cost needs an ISO 4217 currency, got {currency!r}")
    kind = "estimated" if estimated else "actual"
    return {
        "kind": kind,
        "amount": amount,
        "currency": currency,
        "provider": provider,
        "note": None,
    }


def render(result: dict) -> str:
    """`report.md` in Russian from a contract-valid result document."""
    artifacts = {a["id"]: a for a in result["artifacts"]}
    review_completed = result["stages"]["audio_review"]["status"] == "completed"
    provenance = result["provenance"]
    lines = [
        "# Отчёт Sound Scout",
        "",
        f"- Запрос: `{result['request_id']}`; запуск: `{result['run_id']}`.",
        f"- Статус: `{result['status']}` — {_STATUS_LABELS[result['status']]}.",
        f"- Источник: `{result['source']['id']}`, {_seconds(result['source']['duration_s'])}, "
        f"каналов: {result['source']['channels']}, SHA-256 `{result['source']['sha256']}`.",
    ]
    if provenance["synthetic"]:
        lines.append(
            f"- **Синтетический прогон**: не доказательство способности слышать. "
            f"{provenance['note'] or ''}".rstrip()
        )
    lines += [
        "",
        "## Этапы",
        "",
        "| этап | запрошен | статус | причина | время, мс |",
        "|---|---|---|---|---|",
    ]
    for name, stage in result["stages"].items():
        lines.append(
            f"| {_STAGE_LABELS[name]} | {'да' if stage['requested'] else 'нет'} | "
            f"`{stage['status']}` | {_cell(stage['reason'])} | {_cell(stage['elapsed_ms'])} |"
        )
    coverage = result["coverage"]
    lines += [
        "",
        "## Охват",
        "",
        f"- Запрошено: {_ranges(coverage['requested'])}.",
        f"- Прослушано audio backend: {_ranges(coverage['reviewed'])} "
        f"({_seconds(_total(coverage['reviewed']))} из {_seconds(_total(coverage['requested']))}).",
        f"- Не прослушано: {_ranges(coverage['unreviewed'])}. Вне прослушанного охвата "
        "наблюдений о звучании нет и положительных выводов тоже.",
        "",
        "## Оценка по слоям",
        "",
    ]
    for name, assessment in result["assessment_status"].items():
        if assessment["status"] == "assessed":
            lines.append(f"- {_ASSESSMENT_LABELS[name]}: `assessed`.")
        else:
            lines.append(
                f"- {_ASSESSMENT_LABELS[name]}: `not_assessed` — не оценивалось, это не значит "
                f"«всё хорошо». Причина: {assessment['reason']}."
            )
    lines += ["", "## Измерения", ""]
    if not result["measurements"]:
        lines.append("Измерений нет.")
    for m in result["measurements"]:
        value = m["value"]
        if m["name"] == schema.WORD_MEASUREMENT and isinstance(value, list):
            lines += _word_prominence_lines(m)
            continue
        if value is None:
            shown = f"не получено: {m['reason']}"
        elif isinstance(value, list):
            shown = f"{len(value)} интервалов, всего {_seconds(_total(value))}"
        elif isinstance(value, dict):
            shown = ", ".join(f"{k} = {v:g}" for k, v in value.items()) + f" {m['unit']}"
        else:
            shown = f"{value:g} {m['unit']}"
        lines.append(
            f"- `{m['name']}`: {shown}. Метод: `{m['method']}`, настройки: "
            f"`{json.dumps(m['settings'], ensure_ascii=False, sort_keys=True)}`."
        )
        lines += [f"  - ограничение: {text}" for text in m["limitations"]]
    raw = result["transcript"]["raw"]
    lines += ["", "## Транскрипт", ""]
    if raw is None:
        lines.append("Сырого транскрипта нет: ASR не выполнен или его результат не сохранён.")
    else:
        artifact = artifacts[raw["artifact_ref"]]
        lines.append(
            f"Сырой транскрипт ASR: [`{artifact['path']}`]({artifact['path']}), SHA-256 "
            f"`{artifact['sha256']}`. Он не переписывается; исправления ниже — отдельный слой."
        )
    for c in result["transcript"]["corrections"]:
        lines.append(
            f"- {_span(c)}: «{c['original']}» → «{c['corrected']}» (источник `{c['source']}`, "
            f"`{c['certainty']}`){': ' + c['note'] if c['note'] else ''}."
        )
    if result["transcript"]["suspect_segments"]:
        lines += ["", SUSPECT_SEGMENTS_LEAD, ""]
    lines += [
        f"- {_span(s)}: признаки " + ", ".join(f"`{signal}`" for signal in s["signals"]) + "."
        for s in result["transcript"]["suspect_segments"]
    ]
    lines += ["", f"## Наблюдения ({len(result['findings'])})", ""]
    if not result["findings"]:
        lines.append("Наблюдений нет. Это не значит «всё хорошо» за пределами охвата и оценки.")
    for f in prioritized(result["findings"]):
        lead = _BASIS_LEADS[f["evidence_basis"]]
        if f["evidence_basis"] == "audio_model_observation" and not review_completed:
            lead = "Наблюдение без выполненного audio review"
        lines += [
            f"### `{f['id']}` · {f['category']} · {f['polarity']} · {_span(f)}",
            "",
            f"{lead}: {f['observation']}",
            "",
            f"- Основание: `{f['evidence_basis']}`; уверенность: {f['confidence_note'] or '—'}.",
            f"- Влияние на понятность: {f['impact'] or '—'}.",
            f"- Действие: {f['correction'] or 'не требуется'}.",
            f"- Клипы: {_links(f['evidence_refs'], artifacts)}.",
            "",
        ]
    if result["unlocalized"]:
        lines += ["## Наблюдения без точного места", ""]
    for u in result["unlocalized"]:
        reason = "нет таймкодов" if u["reason"] == "no_timecodes" else "весь клип"
        lead = (
            AUDIO_MODEL_PHRASE if review_completed else "Наблюдение без выполненного audio review"
        )
        lines.append(
            f"- `{u['id']}` · {u['category']} ({reason}). {lead}: {u['observation']} "
            f"Влияние: {u['impact'] or '—'}. Действие: {u['correction'] or 'не требуется'}. "
            f"Клипы: {_links(u['evidence_refs'], artifacts)}."
        )
    capabilities = result["capabilities"]
    if lines[-1]:
        lines.append("")
    lines += ["## Ограничения", ""]
    lines += [f"- {text}" for text in provenance["limitations"]]
    lines += [
        f"- слой {name} недоступен: {layer['reason']}"
        for name, layer in capabilities["layers"].items()
        if not layer["available"]
    ]
    lines.append(
        f"- квалификация на русском: validated_ru = {_flag(capabilities['validated_ru'])}, "
        f"дикция = {_flag(capabilities['diction_validated_ru'])}, "
        f"просодия = {_flag(capabilities['prosody_validated_ru'])}."
    )
    lines += ["", "## Пределы", ""]
    lines += [f"- `{name}`: {json.dumps(value)}" for name, value in result["limits"].items()]
    lines += ["", "## Ошибки", ""]
    lines += [f"- `{e['stage'] or '—'}` / `{e['code']}`: {e['message']}" for e in result["errors"]]
    if not result["errors"]:
        lines.append("Ошибок нет.")
    lines += ["", "## Стоимость", "", f"- {_cost_text(result['cost'])}"]
    lines += [
        "",
        "## Происхождение",
        "",
        f"- backend: `{provenance['backend']}`, модель: {_cell(provenance['model'])}, "
        f"revision: {_cell(provenance['revision'])}.",
        f"- входные модальности: {', '.join(provenance['input_modalities']) or '—'}; "
        f"SHA-256 audio payload{_windows_note(result['artifacts'])}: "
        f"{_cell(provenance['audio_payload_sha256'])}; "
        f"prompt: {_cell(provenance['prompt_sha256'])}; схема: `{provenance['schema_sha256']}`.",
        "- версии: "
        + ", ".join(f"{k} {v}" for k, v in sorted(provenance["versions"].items()))
        + ".",
        "",
        "## Артефакты",
        "",
    ]
    lines += [
        f"- `{a['id']}` ({a['kind']}): [`{a['path']}`]({a['path']})" for a in result["artifacts"]
    ]
    return "\n".join(lines) + "\n"


def _word_prominence_lines(m: dict) -> list[str]:
    """Counts and limitations only: values compare within one ASR segment, so no ranking here;
    the method and rule names stay in `result.json` (0038)."""
    value = m["value"]
    missing: dict[str, int] = {}
    for item in value:
        if item["value"] is None:
            missing[item["reason"]] = missing.get(item["reason"], 0) + 1
    segments = len({item["transcript_ref"]["pointer"].partition(".words")[0] for item in value})
    with_value = len(value) - sum(missing.values())
    codes = ", ".join(f"`{code}` — {count}" for code, count in sorted(missing.items()))
    validation = (m.get("validation") or {}).get("decision")
    lines = [
        f"- `{m['name']}`: индекс наличия pitch accent у {with_value} слов из {len(value)} "
        f"в {segments} сегментах ASR; без значения — {sum(missing.values())}"
        + (f" ({codes})" if codes else "")
        + ". Значения по словам — в `result.json`."
        + (f" Проверка: `{validation}`." if validation else "")
    ]
    return lines + [f"  - ограничение: {text}" for text in m["limitations"]]


def _windows_note(artifacts: list[dict]) -> str:
    """With several review windows the provenance field names the first window's payload only."""
    windows = sum(a["kind"] == "review_provenance" for a in artifacts)
    if windows < 2:
        return ""
    return f" первого окна из {windows} (payload каждого окна — в артефактах review_provenance)"


def _compact(result: dict, limit: int) -> dict:
    artifacts = {a["id"]: a for a in result["artifacts"]}
    top = []
    for f in prioritized(result["findings"])[:COMPACT_FINDINGS]:
        top.append(
            {
                "id": _cut(f["id"], limit),
                "category": _cut(f["category"], limit),
                "polarity": f["polarity"],
                "start": f["start"],
                "end": f["end"],
                "evidence_basis": f["evidence_basis"],
                "observation": _cut(f["observation"], limit),
                "impact": _cut(f["impact"], limit),
                "correction": _cut(f["correction"], limit),
                "clips": _clip_paths(f["evidence_refs"], artifacts),
            }
        )
    coverage = {
        name: [{"start": r["start"], "end": r["end"]} for r in ranges[:COMPACT_RANGES]]
        for name, ranges in result["coverage"].items()
    }
    coverage["requested_s"] = round(_total(result["coverage"]["requested"]), 3)
    coverage["reviewed_s"] = round(_total(result["coverage"]["reviewed"]), 3)
    coverage["ranges_truncated"] = any(
        len(ranges) > COMPACT_RANGES for ranges in result["coverage"].values()
    )
    limitations = list(result["provenance"]["limitations"])
    limitations += [
        f"{name}: not_assessed — {a['reason']}"
        for name, a in result["assessment_status"].items()
        if a["status"] == "not_assessed"
    ]
    if result["provenance"]["synthetic"]:
        limitations.insert(0, "synthetic: не доказательство способности слышать")
    clips = sorted({path for f in top for path in f["clips"]})
    return {
        "run_id": _cut(result["run_id"], limit),
        "status": result["status"],
        "stages": {
            name: {"status": stage["status"], "reason": _cut(stage["reason"], limit)}
            for name, stage in result["stages"].items()
        },
        "coverage": coverage,
        "assessment_status": {
            name: {"status": a["status"], "reason": _cut(a["reason"], limit)}
            for name, a in result["assessment_status"].items()
        },
        "findings": top,
        "findings_total": len(result["findings"]),
        "unlocalized_total": len(result["unlocalized"]),
        "limitations": [_cut(text, limit) for text in limitations[:COMPACT_LIST_ITEMS]],
        "errors": [
            {"stage": e["stage"], "code": _cut(e["code"], limit)}
            for e in result["errors"][:COMPACT_LIST_ITEMS]
        ],
        "cost": {k: result["cost"][k] for k in ("kind", "amount", "currency", "provider")},
        "links": {"report": REPORT_FILE_NAME, "result": RESULT_FILE_NAME, "clips": clips},
    }


def _private_values(run: Run, config: Config) -> list[str]:
    values = [str(run.path), str(config.data_root), *map(str, config.allowed_roots)]
    values += [str(config_module.config_path().absolute()), str(Path.home())]
    return values


def _private_values_removed(result: dict, values: list[str]) -> dict:
    document = copy.deepcopy(result)
    document["source"].pop("path", None)
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = ""
    ordered = sorted({v for v in values if len(v) > 1}, key=len, reverse=True)
    user_pattern = re.compile(rf"(?<!\w){re.escape(user)}(?!\w)") if len(user) > 2 else None

    def clean(value: object) -> object:
        if isinstance(value, str):
            for private in ordered:
                value = value.replace(private, PRIVATE_PATH)
            value = _ABSOLUTE_PATHS.sub(PRIVATE_PATH, value)
            return user_pattern.sub(PRIVATE_USER, value) if user_pattern else value
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(document)


def _report_artifact(kind: str, name: str) -> dict:
    return {
        "id": name.replace(".", "-"),
        "kind": kind,
        "path": name,
        "sha256": None,
        "source_start": None,
        "source_end": None,
    }


def _position(order: tuple[str, ...], value: str) -> int:
    return order.index(value) if value in order else len(order)


def _cut(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "…"


def _clip_paths(refs: list[dict], artifacts: dict[str, dict]) -> list[str]:
    return [
        artifacts[ref["ref"]]["path"]
        for ref in refs
        if ref["type"] == "clip" and ref["ref"] in artifacts
    ]


def _links(refs: list[dict], artifacts: dict[str, dict]) -> str:
    links = [f"[`{path}`]({path})" for path in _clip_paths(refs, artifacts)]
    return ", ".join(links) or "—"


def _total(ranges: list[dict]) -> float:
    return sum(r["end"] - r["start"] for r in ranges)


def _seconds(value: float) -> str:
    return f"{value:.2f} с"


def _span(item: dict) -> str:
    return f"{item['start']:.2f}–{item['end']:.2f} с"


def _ranges(ranges: list[dict]) -> str:
    return ", ".join(_span(r) for r in ranges) or "нет"


def _cell(value: object) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _flag(value: bool) -> str:
    return "да" if value else "нет"


def _cost_text(cost_: dict) -> str:
    if cost_["kind"] == "unknown":
        text = "неизвестна (`unknown`)"
    elif cost_["provider"] == LOCAL_PROVIDER or cost_["currency"] is None:
        text = f"{cost_['amount']:g} (`{cost_['kind']}`, {cost_['provider'] or '—'})"
    else:
        text = f"{cost_['amount']:g} {cost_['currency']} (`{cost_['kind']}`)"
    provider = f"; провайдер: {cost_['provider']}" if cost_["provider"] else ""
    note = f"; {cost_['note']}" if cost_["note"] else ""
    return text + provider + note + "."
