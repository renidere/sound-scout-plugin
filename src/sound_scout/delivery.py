"""Delivery of one final note per ended attempt to the chat bound at submit (decision 0013 § 3.7,
Decision in #102): the state of `job.json["delivery"]` and the text of the note.

The plugin's delivery loop (`hermes/plugin/sound-scout/delivery.py`) calls these functions through
`sound-scout jobs deliver-next|deliver-end`; it sends, Sound Scout keeps the state. Hermes does not
deduplicate notes, so a note is sent at most once per attempt: `next_delivery` writes the marker
`{"state": "sending", ...}` under `jobs.lock` before it answers with the text, that is before any
send, and nothing here ever puts a job back to send except `released` (the send function returned
before sending, because no gateway in the calling process owns the session) or a later ended
attempt.

`delivery` of a job: `null` (not delivered yet), the marker `sending` (`tag`, `attempt`,
`loop_id`, `claimed_at`), or how it ended: `delivered` (`confirmed_by`: `send` or `history`),
`delivery_unconfirmed` or `delivery_failed` (`error`), each with `claimed_at` and `ended_at`. A
job whose `job.json` has no `delivery` key (submitted before this field) is never delivered. The
job `state` does not change: delivery is an axis of its own, like the result `status`.

One note per attempt (Decision in #187, decision 0018 "Повтор и доставка"): `attempt` of a note is
the number of attempts the job had when its note was claimed (0 for a job cancelled before any
attempt). When a job ends again after `jobs retry` with more attempts than its note counts, that
note moves to `job.json["deliveries"]` and the job gets a new one. The first note of a job is
tagged `sound-scout-job:<job_id>`, as before #187; each later one `sound-scout-job:<job_id>-a<n>`,
`n` its attempt, so no two notes of a job share a tag. A retry whose attempt never ran (cancelled
in the queue) adds no attempt and gets no note. A note written before #187 has no `attempt`: it
covers every attempt the job has, so a job retried before the update gets no late note; `jobs
retry` writes that number into it before it queues the job again, so the next attempt gets one.
The text of a note is about its own attempt: when `jobs.result` answers with an earlier attempt,
the note says that its attempt gave no result. A redelivery requested for a note that a later
ended attempt supersedes is not sent: the later attempt's note replaces it, and the request stays
in `deliveries` as `redelivery_requested`.

A second delivery is only an explicit request through `sound-scout jobs redeliver` or the MCP tool
`speech_job_redeliver`, both `request_redelivery` (#122, decision 0013 § 3.1): it takes only
`delivery_unconfirmed`, appends `{requested_at, previous}` to `job.json["redeliveries"]` and sets
`delivery` to `redelivery_requested`. `next_delivery` then answers `redeliver` with the same text
and tag; its `sending` marker and the outcome carry `redelivery: true`. The loop looks for the tag
in the whole chat history first and sends once only when the tag is not there; nothing redelivers
on its own.

The address is always `binding.session_id` of the job; there is no other address.

No note for an attempt the chat already saw (decision 0030, Decision in #190). `mark_seen` (`jobs
seen`, called only by the plugin's `post_tool_call` hook after a successful `speech_job_result`
in a turn of the bound chat) writes `seen_at` into the job's last attempt when the job has ended
and that attempt is the one the turn read. For such an attempt `next_delivery` writes the
delivery state `seen_in_chat` (with the tag its note would have had, so the `-a<n>` tags of later
attempts do not shift) and sends nothing. A note must be decided before its send: once admitted,
Hermes writes it after the running turn and it cannot be taken back. So for a note not yet sent,
`next_delivery` answers `probe` (nothing written) until the loop names the chat in `idle_chats`
(its admission queue in Hermes is empty: send now) or in `busy_chats` (a turn runs or waits in
it: skip the job, look further). `TURN_WAIT_SECONDS` after the attempt ended (for a redelivery
request: after `requested_at`) the note goes without a probe: a stuck queue row, an unreadable
queue or a turn that never reads the result must not hold the note for long, and a note admitted
behind a running turn still waits for that turn in the Hermes FIFO. The seen mark does not apply
to `redelivery_requested`, an explicit request of the operator; the wait does.

The note of a control job (Decision in #110) has one comparison block after the layers, from the
`compare` summary of the attempt that answers (`jobs.result`), or of the last attempt without a
result: the verdict with a fixed label and, for `inconclusive`, the reasons by
`COMPARE_REASON_LABELS`; never the `message` of a reason, a window or a word of the transcript.
"""

import re
from collections.abc import Collection
from datetime import UTC, datetime

from sound_scout import jobs, paths, report
from sound_scout.config import Config

TAG_PREFIX = "sound-scout-job:"
DEFAULT_PROFILE = "default"
SENDING = "sending"
DELIVERED = "delivered"
UNCONFIRMED = "delivery_unconfirmed"
FAILED = "delivery_failed"
RELEASED = "released"
REDELIVERY_REQUESTED = "redelivery_requested"
SEEN_IN_CHAT = "seen_in_chat"
NOT_SEEN = "not_seen"
# The turn of the owner's chats reads the result about a minute after the job ends.
TURN_WAIT_SECONDS = 600
# outcome of `end_delivery` -> (delivery state, confirmed_by)
OUTCOMES = {
    "delivered": (DELIVERED, "send"),
    "delivered_by_history": (DELIVERED, "history"),
    "unconfirmed": (UNCONFIRMED, None),
    "failed": (FAILED, None),
    RELEASED: (None, None),
}
ENDED_STATES = (jobs.FINISHED, jobs.CANCELLED, jobs.FAILED)
ERROR_MAX_CHARS = 300
DELIVERY_NOT_SENDING = "delivery_not_sending"
NOT_UNCONFIRMED = "delivery_not_unconfirmed"
LAYER_NAMES = {"content": "содержание", "measurements": "измерения", "sound": "звучание"}
STAGE_STATUS_LABELS = {
    "completed": "выполнено",
    "unavailable": "недоступно",
    "skipped": "пропущено",
    "failed": "ошибка",
}
COMPARE_HEAD = "Сравнение с исходной попыткой"
VERDICT_LABELS = {
    "improved": "улучшение по этому критерию; один контроль не доказывает устойчивый навык",
    "not_improved": "улучшения по этому критерию нет",
    "inconclusive": "сравнение недостоверно",
}
# Every reason code of compare.v1; `{side}` is «попытка A» or «попытка B».
COMPARE_REASON_LABELS = {
    "audio_review_missing": "{side}: audio review `{status}`",
    "criterion_not_reviewed": "{side}: окно критерия не входит в прослушанный диапазон",
    "word_not_found": "{side}: слово критерия не найдено в сыром транскрипте",
    "level_differs": "уровень речи попыток различается больше допустимого",
    "level_unknown": "{side}: уровень речи не измерен",
    "channel_differs": "канал попыток различается",
    "channel_unknown": "{side}: канал неизвестен",
    "format_differs": "формат записи попыток различается",
    "format_unknown": "{side}: формат записи неизвестен",
    "task_differs": "задачи попыток различаются",
    "task_unknown": "{side}: задача не записана",
    "intent_differs": "замысел (intent) попыток различается",
    "backend_differs": "audio backend, модель, ревизия или промпт попыток различаются",
    "same_recording": "обе попытки — одна и та же запись",
    "unlocalized_observation": "{side}: замечание аудиомодели этой категории без таймкодов",
    "no_baseline_issue": "{side}: в исходной попытке нет замечания по критерию",
}
NO_BACKEND = " — audio backend не выбран (`backend = none`)"
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]*")


def tag(job_id: str, attempt: int | None = None) -> str:
    """The mark of a note in the note and in the chat history: of the job's first note without
    `attempt`, of a later note with the attempt it reports. See the module docstring."""
    return f"{TAG_PREFIX}{job_id}" if attempt is None else f"{TAG_PREFIX}{job_id}-a{attempt}"


def _noted_attempt(job: dict, marker: dict) -> int:
    """The number of attempts the note of `marker` counts; see the module docstring."""
    return marker.get("attempt", len(job["attempts"]))


def _superseded(job: dict, marker: object) -> bool:
    """Whether the job ended again after a retry since its settled note was claimed."""
    return (
        isinstance(marker, dict)
        and marker.get("state") != SENDING
        and _noted_attempt(job, marker) < len(job["attempts"])
    )


def next_delivery(
    profile: str,
    loop_id: str,
    config: Config,
    idle_chats: Collection[str] = (),
    busy_chats: Collection[str] = (),
) -> dict:
    """What the loop `loop_id` of the Hermes profile `profile` does next, under `jobs.lock`.

    `check`: a `sending` marker of another loop (one stopped in the middle of a send); the loop
    looks for the tag in the chat history and never sends. `send`: the oldest ended job with a
    `session_id` and `delivery: null` (or a note of an earlier attempt, now moved to
    `deliveries`), now marked `sending`, with the note text; `redeliver`: the same for a job with
    `redelivery_requested`, whose tag the loop looks for in the chat history before it sends.
    A job whose last attempt the chat saw gets `seen_in_chat` on the way and no answer. `probe`
    (nothing written): the job waits for the turn check of its chat, see the module docstring;
    a job of a chat in `busy_chats` is passed over. `none` otherwise.
    """
    jobs._check_identifier("profile", profile)
    jobs._check_identifier("loop_id", loop_id)
    for chat_id in (*idle_chats, *busy_chats):
        jobs._check_identifier("chat", chat_id)
    with jobs._store(config.data_root) as (jobs_dir, _):
        mine = sorted(
            (
                job
                for job in map(jobs._load, jobs._job_dirs(jobs_dir))
                if (job["binding"].get("profile") or DEFAULT_PROFILE) == profile
            ),
            key=lambda job: (job["submitted_at"], job["job_id"]),
        )
        for job in mine:
            marker = job.get("delivery")
            stale = isinstance(marker, dict) and marker.get("state") == SENDING
            if stale and marker.get("loop_id") != loop_id:
                return _address("check", job) | {"tag": marker["tag"]}
        for job in mine:
            if job["state"] not in ENDED_STATES or job["binding"].get("session_id") is None:
                continue
            marker = job.get("delivery", False)
            if _superseded(job, marker):
                job.setdefault("deliveries", []).append(marker)
                marker = None
            requested = isinstance(marker, dict) and marker.get("state") == REDELIVERY_REQUESTED
            if marker is not None and not requested:
                continue
            if requested:
                note_tag, attempt = marker["tag"], _noted_attempt(job, marker)
                since = marker.get("requested_at")
            else:
                attempt = len(job["attempts"])
                note_tag = tag(job["job_id"], attempt if job.get("deliveries") else None)
                last = job["attempts"][attempt - 1] if attempt else {}
                if last.get("seen_at"):
                    job["delivery"] = {
                        "state": SEEN_IN_CHAT,
                        "tag": note_tag,
                        "attempt": attempt,
                        "seen_at": last["seen_at"],
                        "ended_at": jobs._now(),
                    }
                    jobs._save(jobs_dir / job["job_id"], job)
                    continue
                since = last.get("ended_at") or job["updated_at"]
            session_id = job["binding"]["session_id"]
            if not _waited(since) and session_id not in idle_chats:
                if session_id in busy_chats:
                    continue
                return _address("probe", job)
            text = note_text(job, config, note_tag, attempt)
            job["delivery"] = {
                "state": SENDING,
                "tag": note_tag,
                "attempt": attempt,
                "loop_id": loop_id,
                "claimed_at": jobs._now(),
            }
            if requested:
                job["delivery"] |= {"redelivery": True, "requested_at": marker["requested_at"]}
            jobs._save(jobs_dir / job["job_id"], job)
            action = "redeliver" if requested else "send"
            return _address(action, job) | {"tag": note_tag, "text": text}
    return {"action": "none"}


def _waited(since: object) -> bool:
    """Whether `TURN_WAIT_SECONDS` passed since `since`; a time that cannot be read has passed,
    so it never holds a note."""
    try:
        elapsed = datetime.now(UTC) - datetime.fromisoformat(since)
    except (TypeError, ValueError):
        return True
    return elapsed.total_seconds() >= TURN_WAIT_SECONDS


def mark_seen(job_id: str, session_id: str, attempt: int, config: Config) -> dict:
    """Mark the last attempt of an ended job seen in its bound chat `session_id`, after a turn of
    that chat read its result; the answer is the job's status. See the module docstring.

    `invalid_request` for a malformed `session_id` or `attempt`, `job_not_found`, `not_seen` (and
    nothing changes) for another chat or an unbound job, a job not ended, or an `attempt` that is
    not the job's last. A second mark changes nothing. `updated_at` does not change: the staging
    retention counts from it."""
    jobs._check_identifier("session_id", session_id)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise jobs.JobError(jobs.INVALID_REQUEST, f"attempt must be an integer >= 1: {attempt!r}")
    with jobs._store(config.data_root):
        job_dir = jobs._job_dir(job_id, config)
        job = jobs._load(job_dir)
        if job["binding"].get("session_id") != session_id:
            raise jobs.JobError(NOT_SEEN, f"job {job_id} is not bound to the chat of the turn")
        if job["state"] not in ENDED_STATES:
            raise jobs.JobError(NOT_SEEN, f"job {job_id} is {job['state']}, not ended")
        last = job["attempts"][-1] if job["attempts"] else None
        if last is None or last.get("number") != attempt:
            raise jobs.JobError(
                NOT_SEEN,
                f"attempt {attempt} is not the last attempt of job {job_id} "
                f"({last.get('number') if last else None})",
            )
        if not last.get("seen_at"):
            last["seen_at"] = jobs._now()
            jobs._save(job_dir, job)
        return jobs._view(job_dir, job, config)


def request_redelivery(job_id: str, config: Config) -> dict:
    """The operator's request to deliver the note of a `delivery_unconfirmed` job once more;
    the answer is the job's status. Any other delivery state is refused and `job.json` is left
    as it is. `updated_at` does not change: the staging retention counts from it."""
    with jobs._store(config.data_root):
        job_dir = jobs._job_dir(job_id, config)
        job = jobs._load(job_dir)
        marker = job.get("delivery")
        current = marker.get("state") if isinstance(marker, dict) else marker
        if current != UNCONFIRMED:
            raise jobs.JobError(
                NOT_UNCONFIRMED,
                f"job {job_id} delivery is {current!r}; only {UNCONFIRMED} can be delivered again",
            )
        now = jobs._now()
        job.setdefault("redeliveries", []).append({"requested_at": now, "previous": marker})
        job["delivery"] = {
            "state": REDELIVERY_REQUESTED,
            "tag": marker["tag"],
            "attempt": _noted_attempt(job, marker),
            "requested_at": now,
        }
        jobs._save(job_dir, job)
        return jobs._view(job_dir, job, config)


def end_delivery(job_id: str, outcome: str, error: str | None, config: Config) -> dict:
    """Record how the send of a `sending` job ended; the answer is the job's status.

    `released` puts `delivery` back to `null`: the send returned before it sent anything; for a
    redelivery marker back to `redelivery_requested`, so the next try checks the history again.
    """
    if outcome not in OUTCOMES:
        raise jobs.JobError(
            jobs.INVALID_REQUEST, f"outcome must be one of {list(OUTCOMES)}, got {outcome!r}"
        )
    with jobs._store(config.data_root):
        job_dir = jobs._job_dir(job_id, config)
        job = jobs._load(job_dir)
        marker = job.get("delivery")
        if not isinstance(marker, dict) or marker.get("state") != SENDING:
            current = marker.get("state") if isinstance(marker, dict) else marker
            raise jobs.JobError(
                DELIVERY_NOT_SENDING, f"job {job_id} delivery is {current!r}, not {SENDING!r}"
            )
        state, confirmed_by = OUTCOMES[outcome]
        redelivery = marker.get("redelivery") is True
        if state is None and redelivery:
            job["delivery"] = {
                "state": REDELIVERY_REQUESTED,
                "tag": marker["tag"],
                "attempt": _noted_attempt(job, marker),
                "requested_at": marker["requested_at"],
            }
        elif state is None:
            job["delivery"] = None
        else:
            ended = {"state": state, "tag": marker["tag"], "attempt": _noted_attempt(job, marker)}
            if confirmed_by is not None:
                ended["confirmed_by"] = confirmed_by
            else:
                text = paths.without_paths(
                    error or "no error text", job["request"]["input"]["path"], config
                )
                ended["error"] = text[:ERROR_MAX_CHARS]
            if redelivery:
                ended["redelivery"] = True
            job["delivery"] = ended | {"claimed_at": marker["claimed_at"], "ended_at": jobs._now()}
        jobs._save(job_dir, job)
        return jobs._view(job_dir, job, config)


def note_text(
    job: dict, config: Config, note_tag: str | None = None, attempt: int | None = None
) -> str:
    """The plain Russian note of an ended job: the result status, every requested layer with its
    stage status and, unless `completed`, the stage reason of the compact result, what was not
    assessed, the report and the tag in the last line (`note_tag`,
    by default the tag of the job's first note). With `attempt`, the note is about the job's
    first `attempt` attempts: a result of an earlier attempt than the last of them reads as none.

    Never the transcript, a path of the input or a root, nor any "all done": `finished` is the
    queue's state, the layers say what was covered. An unreadable `result.json` reads as no result.
    """
    try:
        answer = jobs.result(job["job_id"], config)
    except jobs.JobError as error:
        if error.code != jobs.NO_RESULT:
            raise
        answer = None
    except (ValueError, KeyError):
        # An unreadable result.json must not block every later note of the profile.
        answer = None
    if attempt is not None:
        job = job | {"attempts": job["attempts"][:attempt]}
        last = job["attempts"][-1].get("number") if job["attempts"] else None
        if answer is not None and answer["attempt"] != last:
            answer = None
    if answer is not None:
        status = answer["status"]
        lines = [
            f"Sound Scout: анализ записи закончен — `{status}`: {report._STATUS_LABELS[status]}."
        ]
        if job["state"] == jobs.CANCELLED:
            lines.append("Задание отменено.")
    else:
        head = f"Sound Scout: анализ записи не дал результата — задание `{job['state']}`"
        code = (job.get("state_reason") or "").partition(":")[0]
        if _REASON_CODE.fullmatch(code):
            head += f", код `{code}`"
        lines = [head + "."]
    requested = job["request"]["layers"]
    for layer in jobs.LAYERS:
        if layer not in requested:
            continue
        stage = jobs.LAYER_STAGES[layer]
        name = f"{LAYER_NAMES[layer]} ({report._STAGE_LABELS[stage]})"
        if answer is None:
            lines.append(f"- {name}: не выполнено")
        else:
            stage_status = answer["stages"][stage]["status"]
            line = f"- {name}: `{stage_status}` — {STAGE_STATUS_LABELS[stage_status]}"
            # The compact reason is cut; report masks known roots, home, user name and other
            # absolute file names only up to their first space (report._ABSOLUTE_PATHS).
            reason = " ".join((answer["stages"][stage].get("reason") or "").split())
            if stage_status != "completed" and reason:
                line += f"; причина: {reason}"
            lines.append(line)
    if answer is not None:
        if answer["findings_total"] > 0:
            lines.append(f"Находок: {answer['findings_total']}; подробности — в отчёте.")
        not_assessed = [
            report._ASSESSMENT_LABELS.get(name, name)
            for name, item in answer["assessment_status"].items()
            if item["status"] == "not_assessed"
        ]
        if not_assessed:
            lines.append(f"Не оценено: {', '.join(not_assessed)}.")
        lines.append(f"Отчёт: {answer['run_dir']}/{report.REPORT_FILE_NAME}")
    if job["request"].get("control"):
        lines += _compare_lines(job, answer)
    lines.append(note_tag or tag(job["job_id"]))
    return "\n".join(lines)


def _compare_lines(job: dict, answer: dict | None) -> list[str]:
    """The comparison block of a control job's note; see the module docstring."""
    attempts = job["attempts"]
    if answer is not None:
        attempts = [item for item in attempts if item.get("number") == answer["attempt"]]
    summary = attempts[-1].get("compare") if attempts else None
    summary = summary or {"state": "not_run", "reason": job["state"]}
    state = summary.get("state")
    if state == "refused":
        return [f"{COMPARE_HEAD} не выполнено: код `{summary.get('code')}`."]
    if state == "failed":
        return [f"{COMPARE_HEAD} не выполнено: внутренняя ошибка."]
    if state != "completed":
        return [
            f"{COMPARE_HEAD} не выполнено: контрольный анализ не закончен "
            f"(`{summary.get('reason')}`)."
        ]
    verdict = summary["verdict"]
    lines = [
        f"{COMPARE_HEAD} по критерию `{summary['criterion']}`: `{verdict}` — "
        f"{VERDICT_LABELS[verdict]}."
    ]
    if verdict == "inconclusive":
        lines.append(
            f"Причины: {'; '.join(_reason_label(summary, item) for item in summary['reasons'])}."
        )
    lines.append(f"Сравнение: {summary['path']}")
    return lines


def _reason_label(summary: dict, reason: dict) -> str:
    code, label = reason["code"], reason["attempt"]
    text = COMPARE_REASON_LABELS.get(code, code)
    side = f"попытка {label.upper()}" if label else ""
    status = (summary.get("audio_review") or {}).get(label)
    text = text.format(side=side, status=status)
    if code == "audio_review_missing" and (summary.get("backend") or {}).get(label) == "none":
        text += NO_BACKEND
    return text


def _address(action: str, job: dict) -> dict:
    return {"action": action, "job_id": job["job_id"], "session_id": job["binding"]["session_id"]}
