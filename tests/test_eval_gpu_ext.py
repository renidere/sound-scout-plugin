"""The extended set of the GPU run #149: its clips, plan, checks, run and analysis.

Clips come from generators and `fixture` (no RINCO, no Piper); the real set needs the pinned
RINCO archive and Piper voice and is tested only where they are on disk. The fake vLLM answers
what a test tells it to: these tests prove the contract of the set, not what a model hears.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sound_scout import config
from sound_scout.eval import gate0, gpu_ext, synthetic, vast_fake

REPO = Path(__file__).resolve().parent.parent
# a task tree (`.work-task/<n>`) has no `data/`: the main checkout's is the one with the sources
CORPUS = REPO / "data" if (REPO / "data").is_dir() else REPO.parents[1] / "data"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("set") / "gpu-ext-fixture"
    pins = gpu_ext.fixture(out)
    manifest, entries = gpu_ext.load_set(
        out, manifest_sha256=pins["manifest_sha256"], plan_sha256=pins["plan_sha256"]
    )
    return {"dir": out, "pins": pins, "manifest": manifest, "entries": entries}


def test_every_clip_id_is_neutral_and_the_roles_stay_in_the_manifest(built) -> None:
    clips = built["manifest"]["clips"]
    assert all(gpu_ext.ID_PATTERN.fullmatch(clip["id"]) for clip in clips)
    assert len({clip["id"] for clip in clips}) == len(clips)
    roles = {clip["role"] for clip in clips}
    assert {"silence-3", "silence-20", "rv-1", "rv-2", "speech-noise"} <= roles
    for entry in built["entries"]:
        for role in ("silence", "defect", "ok", "accent", "rv-"):
            assert f'"clip_id": "{role}' not in entry["text"]
    reviews = [e for e in built["entries"] if e["variant"] == "A"]
    assert reviews and all(f'"clip_id": "{e["clip"]}"' in e["text"] for e in reviews)


def test_the_plan_has_every_block_in_order_with_its_request_counts(built) -> None:
    entries = built["entries"]
    assert [e["n"] for e in entries] == list(range(1, len(entries) + 1))
    blocks = [e["block"] for e in entries]
    assert sorted(set(blocks), key=blocks.index) == list(gpu_ext.BLOCKS)
    counts = gpu_ext.block_counts(entries)
    assert counts["E-C"] == {"requests": 48, "A": 12}
    assert counts["E-P"] == {"requests": 4 * 6, "A": 8}
    assert counts["E-R"] == {"requests": 12 * 5, "A": 0}
    assert counts["E-A"] == {"requests": 12, "A": 12}
    assert counts["E-D"] == {"requests": 10 * 5 + 3, "A": 3}
    assert all(e["max_tokens"] == gpu_ext.REVIEW_MAX_TOKENS for e in entries if e["variant"] == "A")
    assert {e["variant"] for e in entries if e["block"] == "E-P"} == {"Q-S", "K1", "A"}


def test_a_pair_keeps_its_two_takes_side_by_side_and_the_kinds_alternate(built) -> None:
    entries = [e for e in built["entries"] if e["block"] == "E-P"]
    for start in range(0, len(entries), 6):
        group = entries[start : start + 6]
        assert len({e["pair"] for e in group}) == 1
        assert [(e["variant"], e["take"]) for e in group] == [
            ("Q-S", "ok"), ("Q-S", "defect"), ("K1", "ok"), ("K1", "defect"),
            ("A", "ok"), ("A", "defect"),
        ]  # fmt: skip
    sentences = synthetic.GPU_EXT_SENTENCES
    kinds = [s.defect for s in gpu_ext.pair_order(sentences)]
    assert kinds == ["lexical_stress", "word_distortion"] * 10


def test_repeats_point_at_the_same_request_of_an_earlier_block(built) -> None:
    by_n = {e["n"]: e for e in built["entries"]}
    repeats = [e for e in built["entries"] if e["block"] == "E-D"]
    for repeat in repeats:
        original = by_n[repeat["repeat_of"]]
        assert original["block"] != "E-D"
        assert (original["clip"], original["text"], original["variant"]) == (
            repeat["clip"],
            repeat["text"],
            repeat["variant"],
        )
    roles = {c["id"]: c["role"] for c in built["manifest"]["clips"]}
    review_roles = {roles[e["clip"]] for e in repeats if e["variant"] == "A"}
    assert "silence-20" in review_roles


def test_the_transcript_of_p_text_keeps_no_mark_of_the_annotation() -> None:
    value = "*нУ() и надо /сказА́ть, что такая +привЫчка: т.е. *ра-ньше ёлка"
    assert gpu_ext.transcript_of(value) == "ну и надо сказать что такая привычка те раньше елка"
    assert not gpu_ext.eligible_value(value)
    assert gpu_ext.eligible_value("на этом мы /кончаем")


def test_targets_are_single_forms_of_two_vowels_or_more_by_their_hash() -> None:
    words = [
        {"form": "мы", "marked": True},
        {"form": "кончаем", "marked": True},
        {"form": "этом", "marked": False},
        {"form": "этом", "marked": False},
        {"form": "сегодня", "marked": False},
    ]
    assert gpu_ext.targets("u#1", words) == {"marked": "кончаем", "unmarked": "сегодня"}
    assert gpu_ext.targets("u#1", words[:2]) is None


def test_utterances_are_chosen_by_hash_two_per_wav_within_2_to_30_s() -> None:
    words = [{"form": "слово", "marked": True}, {"form": "другое", "marked": False}]
    candidates = [
        {"member": f"m{k % 3}.wav", "number": k, "start": 0.0, "end": end, "words": words}
        for k, end in enumerate([1.5, 2.0, 3.0, 4.0, 5.0, 30.5, 6.0, 7.0, 8.0, 9.0])
    ]
    chosen = gpu_ext.select_utterances(candidates, count=5)
    per_wav = {}
    for c in chosen:
        assert 2.0 <= c["end"] - c["start"] <= 30.0
        per_wav[c["member"]] = per_wav.get(c["member"], 0) + 1
    assert max(per_wav.values()) <= gpu_ext.ER_PER_WAV and len(chosen) == 5
    again = gpu_ext.select_utterances(list(reversed(candidates)), count=5)
    assert [c["number"] for c in again] == [c["number"] for c in chosen]


def _refused(match: str, **kwargs) -> None:
    with pytest.raises(gpu_ext.ExtError, match=match):
        gpu_ext.check_clips(**kwargs)


def test_the_set_refuses_pilot_clips_long_clips_excluded_members_and_role_ids(built) -> None:
    clips = built["manifest"]["clips"]
    gpu_ext.check_clips(clips)
    _refused("pilot-1 or of a Gate 0", clips=clips, forbidden_sha256={clips[5]["wav_sha256"]})
    _refused("over 30", clips=[{**clips[0], "duration_s": 30.01}, *clips[1:]])
    member = next(c["member"] for c in clips if c.get("member"))
    _refused("excluded RINCO", clips=clips, forbidden_members={member})
    _refused("neutral", clips=[{**clips[0], "id": "E-silence"}, *clips[1:]])
    _refused("twice", clips=[clips[0], *clips])
    with pytest.raises(gpu_ext.ExtError, match="not in the set"):
        gpu_ext.check_plan([{**built["entries"][0], "clip": "B1"}], clips)


def test_loading_refuses_a_set_off_its_pins_or_with_a_changed_clip(built, tmp_path) -> None:
    with pytest.raises(gpu_ext.ExtError, match="manifest.json differs"):
        gpu_ext.load_set(built["dir"], manifest_sha256="0" * 64)
    with pytest.raises(gpu_ext.ExtError, match="plan.json differs"):
        gpu_ext.load_set(built["dir"], plan_sha256="0" * 64)
    copy = tmp_path / "copy"
    subprocess.run(["cp", "-r", str(built["dir"]), str(copy)], check=True)
    clip = copy / built["manifest"]["clips"][0]["file"]
    data = bytearray(clip.read_bytes())
    data[-1] ^= 1
    clip.write_bytes(bytes(data))
    with pytest.raises(gpu_ext.ExtError, match="differs from its SHA-256"):
        gpu_ext.load_set(copy)


def test_the_no_speech_clips_are_deterministic_and_carry_no_speech() -> None:
    speech = [gate0.tone(2.0)]
    reversed_ = [(np.linspace(-0.5, 0.5, 16000), "слово")]
    first = gpu_ext.no_speech_clips(speech, reversed_)
    second = gpu_ext.no_speech_clips(speech, reversed_)
    assert [role for role, *_ in first] == [
        "silence-3", "silence-20", "pink", "white", "brown", "tone", "sweep", "chord",
        "clicks", "speech-noise", "rv-1",
    ]  # fmt: skip
    for (_, a, _, _), (_, b, _, _) in zip(first, second, strict=True):
        assert np.array_equal(a, b)
    by_role = {role: samples for role, samples, _, _ in first}
    rms = {r: 20 * np.log10(np.sqrt(np.mean(by_role[r] ** 2))) for r in ("white", "brown")}
    assert rms == pytest.approx({"white": -30.0, "brown": -30.0}, abs=0.01)
    assert np.array_equal(by_role["rv-1"], reversed_[0][0][::-1])


def test_the_statistics_give_the_exact_textbook_numbers() -> None:
    assert gpu_ext.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert gpu_ext.mcnemar_exact(3, 3) == 1.0
    assert gpu_ext.binomial_greater(8, 10, 0.25) == pytest.approx(0.000415802, rel=1e-6)
    items = [(True, True), (True, False), (False, False), (False, False)]
    assert gpu_ext.balanced_accuracy(items) == 0.75
    assert gpu_ext.balanced_accuracy([(True, True)]) is None
    assert gpu_ext.auc([(1.0, True), (0.0, False), (0.5, True)]) == 1.0
    groups = {"a": [(True, True), (False, False)], "b": [(True, False), (False, True)]}
    low, high = gpu_ext.bootstrap_ba(groups)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert gpu_ext.bootstrap_ba(groups) == (low, high)


def _attempt(tmp: Path, built: dict, answer) -> Path:
    attempt = tmp / "attempt"
    (attempt / gate0.RESPONSES_DIR).mkdir(parents=True)
    lines = []
    for entry in built["entries"]:
        content = answer(entry)
        body = {"choices": [{"message": {"content": content}}]}
        (attempt / gate0.RESPONSES_DIR / f"{entry['n']}.json").write_text(json.dumps(body))
        lines.append({**gpu_ext._keys(entry), "sent": True, "http_status": 200})
    (attempt / gpu_ext.TRACE_FILE).write_text("".join(json.dumps(x) + "\n" for x in lines))
    return attempt


def _right(entry: dict) -> str:
    if entry["variant"] == "A":
        return json.dumps(vast_fake.EMPTY_REVIEW)
    if entry["variant"] in ("P", "P-text") and entry.get("target"):
        return "да" if entry["target"] == "marked" else "нет"
    if entry["variant"] == "Q-S":
        return "да" if entry["take"] == "ok" else "нет"
    return "нет речи"


def test_analysis_of_always_yes_is_chance_and_of_right_answers_is_perfect(built, tmp_path):
    yes = gpu_ext.analyze(built["dir"], _attempt(tmp_path / "yes", built, lambda e: "да"))
    assert yes["R"]["R1_balanced_accuracy"] == 0.5
    assert yes["S"]["S1"] == {**yes["S"]["S1"], "right": 0, "share": 0.0, "pairs_complete": 4}
    right = gpu_ext.analyze(built["dir"], _attempt(tmp_path / "right", built, _right))
    assert right["R"]["R1_balanced_accuracy"] == 1.0
    assert right["R"]["R1_ci95"] == [1.0, 1.0]
    assert right["R"]["R2_mcnemar"]["b"] == right["R"]["R2_mcnemar"]["c"] == 0
    assert right["R"]["verdict"] == "insufficient"  # 12 utterances < MIN_UTTERANCES
    assert right["S"]["S1"]["share"] == 1.0 and right["S"]["S1"]["verdict"] == "insufficient"
    assert right["C"]["P_no_speech"]["share"] == 1.0
    assert right["C"]["A_any_observation"]["share"] == 0.0
    assert right["D"]["identical_content"] == {"n": 53, "share": 1.0}
    assert right["blocks"]["E-R"]["answered"] == 60


def test_a_pronunciation_observation_on_the_defect_word_is_a_hit_and_on_ok_a_false_alarm(
    built, tmp_path
) -> None:
    clips = {c["id"]: c for c in built["manifest"]["clips"]}

    def answer(entry):
        if entry["variant"] == "A" and entry["block"] == "E-P":
            clip = clips[entry["clip"]]
            start = clip["word_start_s"] + (0.0 if entry["take"] == "defect" else 1.0)
            item = {
                "category": "pronunciation",
                "observation": "слово сказано иначе",
                "start_seconds": start,
                "end_seconds": min(start + 0.2, clip["duration_s"]),
                "correction": None,
                "confidence_note": "уверенно",
                "impact": "заметно",
            }
            return json.dumps({**vast_fake.EMPTY_REVIEW, "observations": [item]})
        return _right(entry)

    found = gpu_ext.analyze(built["dir"], _attempt(tmp_path, built, answer))
    assert found["S"]["S2_hits"] == {"n": 4, "share": 1.0}
    assert found["S"]["S3_false_alarms"] == {"n": 4, "share": 0.0}
    assert found["S"]["S2_vs_S3_mcnemar"]["b"] == 4


class _Client:
    """A client whose answers the test gives: a status per request, `None` for a timeout."""

    def __init__(self, statuses) -> None:
        self.statuses = list(statuses)
        self.model = "qwen25omni"
        self.calls = 0

    def request(self, path, body=None, timeout=60):
        self.calls += 1
        if path == "/tokenize":
            return 200, b'{"count": 10}', None
        status = self.statuses.pop(0) if self.statuses else 200
        if status is None:
            return None, None, "TimeoutError: timed out"
        if status == "refused":
            return None, None, "ConnectionRefusedError: refused"
        return status, json.dumps({"choices": [{"message": {"content": "да"}}]}).encode(), None

    def text_tokens(self, text):
        return 10


def _run(built, tmp_path, client, **kwargs) -> tuple[dict, list[dict]]:
    config_ = config.Config(allowed_roots=(tmp_path,), data_root=built["dir"].parent)
    attempt = gpu_ext.new_attempt(built["dir"], tmp_path.name, config_)
    summary = gpu_ext.run_blocks(
        built["dir"], built["manifest"], built["entries"], attempt, client, config_, **kwargs
    )
    lines = [json.loads(x) for x in (attempt / gpu_ext.TRACE_FILE).read_text().splitlines()]
    return summary, lines


def test_two_timeouts_close_a_block_and_the_next_block_still_runs(built, tmp_path) -> None:
    client = _Client([200] * 48 + [None, None])
    summary, lines = _run(built, tmp_path, client, probe_=lambda client: True)
    ep = [x for x in lines if x["block"] == "E-P"]
    assert [x.get("limit") for x in ep[:2]] == ["request_timeout"] * 2
    assert {x.get("not_sent") for x in ep[2:]} == {"block_timeouts"}
    assert summary["blocks"]["E-P"]["block_stop"] == "block_timeouts"
    assert all(x["sent"] for x in lines if x["block"] == "E-R")
    assert summary["stop_reason"] is None


def test_a_timeout_the_probe_does_not_answer_ends_the_model(built, tmp_path) -> None:
    summary, lines = _run(built, tmp_path, _Client([None]), probe_=lambda client: False)
    assert summary["stop_reason"] == "server_unresponsive"
    assert sum(x["sent"] for x in lines) == 1
    assert {x["not_sent"] for x in lines[1:]} == {"server_unresponsive"}


def test_the_stop_check_and_the_transport_check_end_the_run(built, tmp_path) -> None:
    ends = iter([None] * 30 + ["soft_end"] * 1000)
    summary, lines = _run(built, tmp_path / "soft", _Client([]), stop_check=lambda: next(ends))
    assert sum(x["sent"] for x in lines) == 30 and summary["stop_reason"] == "soft_end"
    checks = []
    summary, lines = _run(
        built,
        tmp_path / "gone",
        _Client(["refused"]),
        transport_check=lambda: checks.append(1) or "server_exited",
    )
    assert checks == [1] and summary["stop_reason"] == "server_exited"
    assert sum(x["sent"] for x in lines) == 1


def test_blocks_report_their_end_after_each_block_that_sent_anything(built, tmp_path) -> None:
    ended = []
    _run(
        built,
        tmp_path,
        _Client([]),
        on_block_end=lambda block, record: ended.append((block, record["sent"])),
    )
    assert [b for b, _ in ended] == list(gpu_ext.BLOCKS) and all(n for _, n in ended)


def test_gpu_ext_sentences_use_new_words_with_both_takes_valid() -> None:
    old = {s.word for s in synthetic.SENTENCES}
    new = [s.word for s in synthetic.GPU_EXT_SENTENCES]
    assert len(set(new)) == 20 and not old & set(new)
    kinds = [s.defect for s in synthetic.GPU_EXT_SENTENCES]
    assert kinds.count("lexical_stress") == 10
    subtypes = [s.subtype for s in synthetic.GPU_EXT_SENTENCES if s.defect != "lexical_stress"]
    assert sorted(subtypes) == sorted(
        ["syllable_drop"] * 4 + ["ending_slur"] * 3 + ["consonant_substitution"] * 3
    )
    with pytest.raises(synthetic.SyntheticError, match="take"):
        synthetic.synthesize(None, synthetic.GPU_EXT_SENTENCES[0], take="maybe")


def _real_sources() -> bool:
    return (CORPUS / "corpus" / "pilot-1" / "provenance.json").is_file() and (
        CORPUS / "corpus" / "sources" / "piper-denis"
    ).is_dir()


@pytest.mark.slow
@pytest.mark.skipif(not _real_sources(), reason="the pinned RINCO archive and Piper voice")
def test_the_real_set_is_built_from_rinco_without_pilot_1_and_piper_twice(tmp_path) -> None:
    pytest.importorskip("piper")
    found = gpu_ext.prepare(CORPUS, tmp_path / "set")
    assert {k: v["requests"] for k, v in found["counts"].items()} == {
        "E-C": 48, "E-P": 120, "E-R": 480, "E-A": 16, "E-D": 53,
    }  # fmt: skip
    manifest, _ = gpu_ext.load_set(tmp_path / "set", manifest_sha256=found["manifest_sha256"])
    er = [c for c in manifest["clips"] if c["block"] == "E-R"]
    assert len(er) == 96 and all(2.0 <= c["duration_s"] <= 30.0 for c in er)
    gpu_ext.check_clips(manifest["clips"], forbidden_sha256=gpu_ext.pilot_sha256(CORPUS))
    second = gpu_ext.prepare(CORPUS, tmp_path / "again")
    assert second["plan_sha256"] == found["plan_sha256"]
    assert second["manifest_sha256"] == found["manifest_sha256"]
    left = json.loads((tmp_path / "set" / "manifest.json").read_text())["clips"]
    right = json.loads((tmp_path / "again" / "manifest.json").read_text())["clips"]
    assert [c["wav_sha256"] for c in left] == [c["wav_sha256"] for c in right]


def test_the_cli_writes_the_analysis_once(built, tmp_path) -> None:
    attempt = _attempt(tmp_path, built, _right)
    argv = [sys.executable, "-m", "sound_scout.eval.gpu_ext", "analyze"]
    argv += ["--set", str(built["dir"]), "--attempt", str(attempt)]
    done = subprocess.run(argv, capture_output=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads((attempt / gpu_ext.ANALYSIS_FILE).read_text())["kind"] == "gpu_ext_analysis"
    again = subprocess.run(argv, capture_output=True, timeout=120, check=False)
    assert again.returncode == 2 and b"File exists" in again.stderr
    named = argv + ["--name", "analysis-utterances.json"]
    assert subprocess.run(named, capture_output=True, timeout=120, check=False).returncode == 0
    assert (attempt / "analysis-utterances.json").is_file()
    bad = subprocess.run(argv + ["--name", "../x.json"], capture_output=True, timeout=120)
    assert bad.returncode == 2 and b"plain file name" in bad.stderr


def test_the_prosody_rule_auc_skips_target_words_the_series_left_unscored(built, tmp_path) -> None:
    clips = [c for c in built["manifest"]["clips"] if c["block"] == "E-R"]
    rows = []
    for number, clip in enumerate(clips):
        words = [
            {"form": clip["targets"]["marked"], "marked": True, "score": {"index": 1.0}},
            {
                "form": clip["targets"]["unmarked"],
                "marked": False,
                "score": None if number % 2 else {"index": -1.0},
            },
        ]
        rows.append({"member": clip["member"], "utterance": clip["utterance"], "words": words})
    path = tmp_path / "utterances.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    found = gpu_ext.analyze(
        built["dir"], _attempt(tmp_path / "a", built, _right), utterances_path=path
    )
    assert found["R"]["prosody_rule_v1_auc"] == 1.0
    assert found["R"]["prosody_rule_v1_words"] == len(clips) + len(clips) // 2
