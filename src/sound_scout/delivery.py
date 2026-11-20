"""Delivery of one final note per job to the chat bound at submit (decision 0013 § 3.7, Decision
in #102): the state of `job.json["delivery"]` and the text of the note.

The plugin's delivery loop (`hermes/plugin/sound-scout/delivery.py`) calls these functions through
`sound-scout jobs deliver-next|deliver-end`; it sends, Sound Scout keeps the state. Hermes does not
deduplicate notes, so a job is delivered at most once: `next_delivery` writes the marker
`{"state": "sending", ...}` under `jobs.lock` before it answers with the text, that is before any
send, and nothing here ever puts a job back to send except `released` (the send function returned
before sending, because no gateway in the calling process owns the session).

`delivery` of a job: `null` (not delivered yet), the marker `sending` (`tag`, `loop_id`,
`claimed_at`), or how it ended: `delivered` (`confirmed_by`: `send` or `history`),
`delivery_unconfirmed` or `delivery_failed` (`error`), each with `claimed_at` and `ended_at`. A
job whose `job.json` has no `delivery` key (submitted before this field) is never delivered. The
job `state` does not change: delivery is an axis of its own, like the result `status`.

A second delivery is only an explicit request through `sound-scout jobs redeliver` or the MCP tool
`speech_job_redeliver`, both `request_redelivery` (#122, decision 0013 § 3.1): it takes only
`delivery_unconfirmed`, appends `{requested_at, previous}` to `job.json["redeliveries"]` and sets
`delivery` to `redelivery_requested`. `next_delivery` then answers `redeliver` with the same text
and tag; its `sending` marker and the outcome carry `redelivery: true`. The loop looks for the tag
in the whole chat history first and sends once only when the tag is not there; nothing redelivers
on its own.

The address is always `binding.session_id` of the job; there is no other address.
"""

import re

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
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]*")


def tag(job_id: str) -> str:
    """The job's mark in the note and in the chat history."""
    return f"{TAG_PREFIX}{job_id}"


def next_delivery(profile: str, loop_id: str, config: Config) -> dict:
    """What the loop `loop_id` of the Hermes profile `profile` does next, under `jobs.lock`.

    `check`: a `sending` marker of another loop (one stopped in the middle of a send); the loop
    looks for the tag in the chat history and never sends. `send`: the oldest ended job with a
    `session_id` and `delivery: null`, now marked `sending`, with the note text; `redeliver`: the
    same for a job with `redelivery_requested`, whose tag the loop looks for in the chat history
    before it sends. `none` otherwise.
    """
    jobs._check_identifier("profile", profile)
    jobs._check_identifier("loop_id", loop_id)
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
            requested = isinstance(marker, dict) and marker.get("state") == REDELIVERY_REQUESTED
            if marker is not None and not requested:
                continue
            text = note_text(job, config)
            job["delivery"] = {
                "state": SENDING,
                "tag": tag(job["job_id"]),
                "loop_id": loop_id,
                "claimed_at": jobs._now(),
            }
            if requested:
                job["delivery"] |= {"redelivery": True, "requested_at": marker["requested_at"]}
            jobs._save(jobs_dir / job["job_id"], job)
            action = "redeliver" if requested else "send"
            return _address(action, job) | {"tag": tag(job["job_id"]), "text": text}
    return {"action": "none"}


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
        job["delivery"] = {"state": REDELIVERY_REQUESTED, "tag": marker["tag"], "requested_at": now}
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
                "requested_at": marker["requested_at"],
            }
        elif state is None:
            job["delivery"] = None
        else:
            ended = {"state": state, "tag": marker["tag"]}
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


def note_text(job: dict, config: Config) -> str:
    """The plain Russian note of an ended job: the result status, every requested layer with its
    stage status, what was not assessed, the report and the tag in the last line.

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
            lines.append(f"- {name}: `{stage_status}` — {STAGE_STATUS_LABELS[stage_status]}")
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
    lines.append(tag(job["job_id"]))
    return "\n".join(lines)


def _address(action: str, job: dict) -> dict:
    return {"action": action, "job_id": job["job_id"], "session_id": job["binding"]["session_id"]}
