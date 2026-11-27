"""prosody-rupro-1 (#199): rule check, import and join of RuPro, exclusions, frozen trees, verdict.

Every table, EXB, graph, archive and recording here is synthetic, and the transcriber is a
stand-in that names voiced runs: these tests prove the contract of
`sound_scout.eval.prosody_rupro` (what is joined, excluded, checked, recorded and refused), never
that the index finds accented words. Only the real run recorded in docs/eval can say that.
"""

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sound_scout import config
from sound_scout.eval import gate0, prosody
from sound_scout.eval import prosody_confirm as pc
from sound_scout.eval import prosody_rupro as pr
from tests.test_eval_prosody import words_signal

REPO = Path(__file__).resolve().parent.parent
RATE = pr.SAMPLE_RATE
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


# --- synthetic sources ---------------------------------------------------------------------


def table_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, pr.TABLE_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                pr.BILINGUAL: "no",
                pr.SETTING: "formal",
                pr.IP_VALUE: "IP",
                pr.POS: "NOUN",
                **row,
            }
        )
    return buffer.getvalue().encode()


def row(doc: str, ip: str, token: str, text: str, label: str = "None", **extra) -> dict:
    return {
        pr.DOC: doc,
        pr.IP_ID: ip,
        pr.TOKEN_ID: token,
        pr.DIPL: text,
        pr.WORD_PA: label,
        **extra,
    }


def exb_bytes(events: list[tuple[float, float, str]]) -> bytes:
    points = sorted({t for s, e, _ in events for t in (s, e)})
    ids = {t: f"T{i}" for i, t in enumerate(points)}
    timeline = "".join(f'<tli id="{ids[t]}" time="{t}"/>' for t in points)
    body = "".join(
        f'<event start="{ids[s]}" end="{ids[e]}">{text}</event>' for s, e, text in events
    )
    return (
        "<?xml version='1.0' encoding='utf-8'?><basic-transcription><basic-body>"
        f"<common-timeline>{timeline}</common-timeline>"
        f'<tier id="TIE0" category="dipl" type="t">{body}</tier>'
        '<tier id="TIE1" category="norm" type="t"><event start="T0" end="T1">x</event></tier>'
        "</basic-body></basic-transcription>"
    ).encode()


def graphml(docs: dict[str, list[tuple[str, float | None, float | None, str | None]]]) -> bytes:
    """Per document its dipl tokens (text, start, end, word_pa); start None — no base token."""
    keys = {"dipl": "default_ns::dipl", "time": "annis::time", "pa": "default_ns::word_pa",
            "base": "default_ns::base", "type": "annis::node_type"}  # fmt: skip
    parts = ['<?xml version="1.0" encoding="UTF-8"?><graphml>']
    parts += [
        f'<key id="{k}" for="node" attr.name="{n}" attr.type="string"/>' for k, n in keys.items()
    ]
    parts.append("<graph>")
    edges = []
    for doc, tokens in docs.items():
        node = f"{pr.GRAPH_PREFIX}{doc}#"
        for i, (text, start, end, label) in enumerate(tokens):
            parts.append(f'<node id="{node}d{i}"><data key="dipl">{text}</data></node>')
            if i + 1 < len(tokens):
                edges.append((f"{node}d{i}", f"{node}d{i + 1}", "Ordering/default_ns/dipl"))
            if start is None:
                continue
            parts.append(f'<node id="{node}v{i}"><data key="time">{start}-{end}</data></node>')
            parts.append(f'<node id="{node}b{i}"><data key="base">{text}</data></node>')
            parts.append(f'<node id="{node}w{i}"><data key="pa">{label}</data></node>')
            edges += [
                (f"{node}b{i}", f"{node}d{i}", "Pointing/default_ns/a"),
                (f"{node}b{i}", f"{node}v{i}", "Coverage/VIRTUAL/timelineRelationMapping"),
                (f"{node}w{i}", f"{node}v{i}", "Coverage/default_ns/"),
            ]
    parts.append(
        '<node id="RUEG-RuPro/RU/BILINGUAL/USbi01MR_fsR#d0"><data key="dipl">x</data></node>'
    )
    parts += [f'<edge source="{s}" target="{t}" label="{label}"/>' for s, t, label in edges]
    parts.append("</graph></graphml>")
    return "".join(parts).encode()


def wav16(samples: np.ndarray) -> bytes:
    return gate0.wav_bytes(gate0.pcm16(samples))


DOCS = ("RUmo11FR_fsR", "RUmo11FR_isR", "RUmo12MR_fsR", "RUmo12MR_isR", "RUmo01FR_fsR")


def recording(raised: int):
    """Four voiced words (word `raised` higher and louder) and their EXB events and graph tokens."""
    samples, words = words_signal(4, raised=raised)
    total = len(samples) / RATE
    events = [(0.0, 0.25, "(-)")]
    graph = [("(-)", 0.0, 0.25, "None")]
    for w in words:
        events.append((round(w["start"] - 0.05, 3), round(w["end"] + 0.05, 3), w["word"]))
        graph.append((w["word"], round(w["start"], 3), round(w["end"], 3), None))
    last = events[-1][1]
    events += [(last, round(last + 0.1, 3), "э:"), (round(last + 0.1, 3), round(total, 3), "(-)")]
    graph += [("э:", last, round(last + 0.1, 3), "None"), ("(-)", None, None, None)]
    return samples, events, graph


def station_sources(sources: Path, marked: int = 2) -> dict:
    """A RuPro station of `DOCS`: one IP of four words per document, word `marked` accented."""
    rows, graph, members = [], {}, {}
    token = 1000
    for number, doc in enumerate(DOCS):
        samples, events, tokens = recording(raised=2)
        ip = str(500 + number)
        for k in range(1, 5):
            label = "H*" if k == marked else "None"
            token += 1
            setting = "formal" if doc.endswith("fsR") else "informal"
            rows.append(row(doc, ip, str(token), f"слово{k}", label, **{pr.SETTING: setting}))
            if k == 3:  # a second POS row of the same token
                rows.append(row(doc, ip, str(token), "слово3", label, **{pr.SETTING: setting}))
        graph[doc] = [
            (t, s, e, ("H*" if t == f"слово{marked}" else "None") if lab is None and s else lab)
            for t, s, e, lab in tokens
        ]
        members[pr.mp3_name(doc)] = wav16(samples)
        base = pr.member_path(sources, pr.EXB_PREFIX)
        base.mkdir(parents=True, exist_ok=True)
        (base / f"{doc}.exb").write_bytes(exb_bytes(events))
        (base / f"{doc}.meta").write_text(f"setting=x\nspeaker-id={doc.split('_')[0]}\n")
    rows.append(row("USbi01MR_fsR", "9", "9", "x", **{pr.BILINGUAL: "yes"}))
    (sources / pr.TABLE_FILE).write_bytes(table_bytes(rows))
    rupro = pr.member_path(sources, pr.RUPRO_MEMBER)
    rupro.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(rupro, "w") as archive:
        archive.writestr(pr.GRAPH_MEMBER, graphml(graph))
        for name, data in members.items():
            archive.writestr(name, data)
    pr.member_path(sources, pr.LICENSE_MEMBER).write_text("CC0\n")
    return graph


def pin_sources(monkeypatch, sources: Path) -> None:
    """Pin the synthetic sources the way the real ones are pinned."""
    sha = pr.sha256_file
    table = sources / pr.TABLE_FILE
    monkeypatch.setattr(pr, "TABLE_BYTES", table.stat().st_size)
    monkeypatch.setattr(pr, "TABLE_SHA256", sha(table))
    pinned = {}
    for member in (pr.LICENSE_MEMBER, pr.RUPRO_MEMBER):
        path = pr.member_path(sources, member)
        pinned[member] = (path.stat().st_size, sha(path))
    monkeypatch.setattr(pr, "PINNED_MEMBERS", pinned)
    docs = sorted(DOCS)
    monkeypatch.setattr(pr, "DOC_COUNT", len(docs))
    monkeypatch.setattr(pr, "DOCS_SHA256", pr.list_sha256(docs))
    lines = [f"{m} {sha(pr.member_path(sources, m))}" for m in pr.exb_members(docs)]
    monkeypatch.setattr(pr, "EXB_LIST_SHA256", pr.list_sha256(lines))


def make_config(tmp_path: Path) -> config.Config:
    (tmp_path / "in").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    return config.Config(allowed_roots=(tmp_path / "in",), data_root=(tmp_path / "data").resolve())


FROZEN_INFO = {"commit": "c" * 40, "manifest_sha256": "m" * 64}


@pytest.fixture
def station(tmp_path, monkeypatch):
    """A data root with the RuPro sources pinned, Q1 and the frozen tree taken as holding."""
    settings = make_config(tmp_path)
    sources = pr.sources_dir(settings)
    sources.mkdir(parents=True)
    station_sources(sources)
    pin_sources(monkeypatch, sources)
    provenance = settings.data_root / prosody.PILOT_PROVENANCE
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"clips": [{"member": "INTON-EAF/x.wav"}]}))
    confirm = settings.data_root / pc.OUTPUT_DIR / pc.PREPARE_FILE
    confirm.parent.mkdir(parents=True)
    confirm.write_text(json.dumps({"fresh_clips": [], "calibration": []}))
    monkeypatch.setattr(prosody, "q1", lambda _: [])
    monkeypatch.setattr(pc, "frozen_problems", lambda: [])
    monkeypatch.setattr(pc, "reference_hashes", lambda _: {"rinco": set(), "pilot_1": set()})
    monkeypatch.setattr(pr, "frozen_state", lambda tree=None: ([], dict(FROZEN_INFO)))
    monkeypatch.setattr(prosody, "memory_scope", lambda: (str(prosody.MEMORY_MAX_BYTES), "0"))
    monkeypatch.setattr(prosody, "pin_cpus", lambda: [0])
    monkeypatch.setattr(pr, "MIN_PAIRS", 3)
    monkeypatch.setattr(pr, "MIN_IPS_WITH_PAIRS", 2)
    monkeypatch.setattr(pr, "MIN_SPEAKERS_WITH_PAIRS", 2)
    return settings


def voiced_runs(samples: np.ndarray, threshold: float = 0.05) -> list[tuple[float, float]]:
    frame = RATE // 100
    count = len(samples) // frame
    levels = [np.sqrt(np.mean(samples[i * frame : (i + 1) * frame] ** 2)) for i in range(count)]
    runs, start = [], None
    for i, level in enumerate([*levels, 0.0]):
        if level > threshold and start is None:
            start = i
        elif level <= threshold and start is not None:
            runs.append((start / 100, i / 100))
            start = None
    return runs


def run_transcriber(samples, run_dir: Path) -> dict:
    """Names the voiced runs of a cut `слово1`, `слово2`, … with their times."""
    run_dir.mkdir(parents=True)
    runs = voiced_runs(np.asarray(samples))
    words = [{"word": f" слово{k}", "start": s, "end": e} for k, (s, e) in enumerate(runs, 1)]
    return prosody.transcribed("completed", None, words, trusted_segments=int(bool(words)))


def passed_sanity(monkeypatch) -> None:
    monkeypatch.setattr(pc, "sanity", lambda *_: {"passed": True, "r0": True, "s2": True})


# --- the rule ------------------------------------------------------------------------------


def test_rule_sha256_of_0037_is_the_one_recorded_before_the_index() -> None:
    assert pr.check_rule(REPO / pr.DECISION_DOC) == pr.RULE_SHA256
    assert pr.RULE_SHA256 in (REPO / pr.DECISION_DOC).read_text(encoding="utf-8")


def test_a_changed_rule_block_stops_every_step_before_any_data(tmp_path, station) -> None:
    changed = tmp_path / "0037.md"
    text = (REPO / pr.DECISION_DOC).read_text(encoding="utf-8")
    changed.write_text(text.replace("AUC >= 0,75", "AUC >= 0,70"), encoding="utf-8")
    for step in (pr.survey, pr.prepare, pr.run_check_fresh, pr.run_sanity, pr.run_main):
        with pytest.raises(pr.RuproError, match="SHA-256"):
            step(station, changed)
    assert not (station.data_root / pr.OUTPUT_DIR / pr.PREPARE_FILE).exists()


def test_the_rule_block_must_be_one_prosody_rupro_rule_v1_block() -> None:
    block = f"{pr.RULE_BEGIN}\n```text\n{pr.RULE_VERSION} x\n```\n{pr.RULE_END}"
    assert len(pr.rule_sha256(block)) == 64
    with pytest.raises(pr.RuproError, match="one"):
        pr.rule_sha256(block + block)
    with pytest.raises(pr.RuproError, match="does not start"):
        pr.rule_sha256(block.replace(pr.RULE_VERSION, "PROSODY-RULE v1"))


def test_the_thresholds_are_those_of_series_3_and_the_minimum_of_the_decision() -> None:
    assert (prosody.PASS_AUC, prosody.PASS_CI_LOW, prosody.PASS_COVERAGE) == (0.75, 0.70, 0.70)
    assert (pr.MIN_PAIRS, pr.MIN_IPS_WITH_PAIRS, pr.MIN_SPEAKERS_WITH_PAIRS) == (1200, 60, 20)
    assert pr.MAIN_WALL_SECONDS == 180 * 60 and pr.MAX_SANITY_ATTEMPTS == 2


# --- labels and events ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("None", 0),
        ("H*", 1),
        ("L+H*", 1),
        ("L*+H", 1),
        ("H+L*", 1),
        ("L+^H*", 1),
        ("!H*", 1),
        ("L*", 1),
        ("H*+L", 1),
        ("H*, L+H*", 1),
        ("L%", None),
        ("!H%", None),
        ("L*+^H%", None),
        ("L+ H*", None),
        ("H", None),
        ("L*+H*", None),
        ("H*,", None),
        ("", None),
        ("none", None),
    ],
)
def test_word_pa_gives_0_1_or_ambiguous(value, expected) -> None:
    assert pr.accent(value) == expected


@pytest.mark.parametrize(
    ("text", "kind"),
    [("э:", "hesitation"), ("м:", "hesitation"), ("[laughing]", "nonspeech"), ("с/", "cutoff"),
     ("UNK", "unk"), ("(UNK)", "unk"), ("дом", "other")],
)  # fmt: skip
def test_an_unmatched_exb_event_gets_its_class(text, kind) -> None:
    assert pr.event_class(text) == kind


@pytest.mark.parametrize("text", ["(-)", "(--)", " (-) ", "(1.2)", "(0,5)", "(N.N)"])
def test_pauses_are_left_out_of_the_alignment(text) -> None:
    assert pr.is_pause(text)
    assert not pr.is_pause("(UNK)") and not pr.is_pause("дом")


# --- the table, EXB, meta and graph --------------------------------------------------------


def test_the_table_keeps_monolingual_tokens_once_with_their_rows_and_ips() -> None:
    data = table_bytes(
        [
            row("RUmo11FR_fsR", "1", "10", "мама", "H*"),
            row("RUmo11FR_fsR", "1", "11", "мыла"),
            row("RUmo11FR_fsR", "1", "11", "мыла", **{pr.POS: "VERB"}),
            row("RUmo11FR_fsR", "1", "12", "раму"),
            row("RUmo11FR_fsR", "2", "12", "раму"),
            row("USbi01MR_fsR", "3", "13", "x", **{pr.BILINGUAL: "yes"}),
        ]
    )
    docs, stats = pr.read_table(data)
    assert list(docs) == ["RUmo11FR_fsR"] and stats == {"rows": 6, "monolingual_rows": 5}
    tokens = docs["RUmo11FR_fsR"]["tokens"]
    assert [t.text for t in tokens] == ["мама", "мыла", "раму"]
    assert tokens[1].rows == (3, 4) and tokens[1].pos == ("NOUN", "VERB")
    assert tokens[2].ips == ("1", "2") and tokens[0].word_pa == ("H*",)


def test_the_table_refuses_missing_columns_and_tokens_out_of_order() -> None:
    with pytest.raises(pr.RuproError, match="columns missing"):
        pr.read_table(b"a\tb\n1\t2\n")
    data = table_bytes([row("RUmo11FR_fsR", "1", "11", "a"), row("RUmo11FR_fsR", "1", "10", "b")])
    with pytest.raises(pr.RuproError, match="order"):
        pr.read_table(data)


def test_exb_events_come_in_timeline_order_and_meta_names_the_speaker() -> None:
    events = pr.parse_exb(exb_bytes([(0.5, 1.0, "b"), (0.0, 0.5, "a")]))
    assert [(e.start, e.end, e.text) for e in events] == [(0.0, 0.5, "a"), (0.5, 1.0, "b")]
    assert pr.parse_meta("a=b\nspeaker-id=RUmo11FR\n") == "RUmo11FR"
    assert pr.parse_meta("a=b\n") is None
    broken = exb_bytes([(0.0, 0.5, "a")]).replace(b'category="dipl"', b'category="x"')
    with pytest.raises(pr.RuproError, match="dipl"):
        pr.parse_exb(broken)


def test_the_graph_gives_each_dipl_token_its_base_time_and_word_pa() -> None:
    data = graphml(
        {
            "RUmo11FR_fsR": [
                ("мама", 0.1, 0.4, "H*"),
                ("(-)", 0.4, 0.6, "None"),
                ("x", None, None, None),
            ],
            "RUmo11FR_isR": [("a", 0.0, 1.0, "None")],
        }
    )
    found = pr.parse_graph(io.BytesIO(data), {"RUmo11FR_fsR"})
    assert list(found) == ["RUmo11FR_fsR"]
    tokens = found["RUmo11FR_fsR"]
    assert tokens[0] == pr.GraphToken("мама", 0.1, 0.4, "H*")
    assert tokens[2] == pr.GraphToken("x", None, None, None)


def test_open_time_ends_give_no_time() -> None:
    assert pr._time_parts("1.5-") == (1.5, None)
    assert pr._time_parts("-2.5") == (None, 2.5)
    assert pr._time_parts("1.0-2.0") == (1.0, 2.0)


# --- the join ------------------------------------------------------------------------------


def doc_parts(texts, labels=None, events=None, graph=None, ip_values=None):
    labels = labels or ["None"] * len(texts)
    tokens = [
        pr.Token(str(10 + i), t, (2 + i,), ("1",), (labels[i],), ("NOUN",))
        for i, t in enumerate(texts)
    ]
    if events is None:
        events = [pr.Event(i * 0.5, i * 0.5 + 0.5, t) for i, t in enumerate(texts)]
    if graph is None:
        graph = [
            pr.GraphToken(e.text, e.start + 0.05, e.end - 0.05, labels[i])
            for i, e in enumerate(events)
        ]
    table = {"tokens": tokens, "ip_values": ip_values or {"1": ["IP"]}}
    return table, events, graph


def joined(doc="RUmo11FR_fsR", mp3=60.0, **parts):
    table, events, graph = doc_parts(**parts)
    ips, stats = pr.join_doc(doc, table, events, graph, mp3, doc.split("_")[0])
    return ips, stats


def test_a_clean_ip_joins_with_labels_provenance_and_the_union_interval() -> None:
    ips, stats = joined(texts=["мама", "мыла", "раму"], labels=["H*", "None", "L+H*"])
    (ip,) = ips
    assert ip["reasons"] == [] and ip["reason"] is None
    assert [w["marked"] for w in ip["words"]] == [1, 0, 1] and ip["pairs"] == 2
    assert ip["words"][0] | {} == ip["words"][0] and ip["words"][0]["rows"] == [2]
    assert ip["words"][0]["graph_word_pa"] == "H*" and ip["words"][0]["token_id"] == "10"
    assert ip["exb"] == [0.0, 1.5] and ip["rupro"] == [0.05, 1.45] and ip["cut"] == [0.0, 1.5]
    assert ip["speaker"] == "RUmo11FR" and ip["setting"] == "fsR"
    assert stats["matched_tokens"] == 3 and stats["unmatched_tokens"] == 0


def test_rupro_words_outside_exb_widen_the_cut() -> None:
    table, events, graph = doc_parts(["мама", "мыла"])
    graph = [pr.GraphToken("мама", -0.0, 0.4, "None"), pr.GraphToken("мыла", 0.5, 1.3, "None")]
    events = [pr.Event(0.1, 0.5, "мама"), pr.Event(0.5, 1.0, "мыла")]
    (ip,), _ = pr.join_doc("RUmo11FR_fsR", table, events, graph, 60.0, "RUmo11FR")
    assert ip["cut"] == [0.0, 1.3] and ip["reasons"] == []


def test_pauses_and_hesitations_of_exb_do_not_break_the_join() -> None:
    texts = ["мама", "мыла"]
    spans = [(0.0, 0.2, "(-)"), (0.2, 0.6, "мама"), (0.6, 0.8, "э:"), (0.8, 1.2, "[laughing]"),
             (1.2, 1.6, "мыла"), (1.6, 1.8, "с/")]  # fmt: skip
    events = [pr.Event(*span) for span in spans]
    graph = [pr.GraphToken(e.text, e.start, e.end, "None") for e in events]
    (ip,), stats = joined(texts=texts, events=events, graph=graph)
    assert ip["reasons"] == [] and [w["exb_event"] for w in ip["words"]] == [1, 4]
    assert stats["unmatched_exb_events"] == {"cutoff": 1, "hesitation": 1, "nonspeech": 1}
    assert stats["exb_pauses"] == 1


def test_a_token_whose_text_exb_lacks_excludes_its_ip() -> None:
    events = [pr.Event(0.0, 0.5, "мама"), pr.Event(0.5, 1.0, "мыло")]
    graph = [pr.GraphToken(e.text, e.start, e.end, "None") for e in events]
    (ip,), stats = joined(texts=["мама", "мыла"], events=events, graph=graph)
    assert ip["reason"] == "unmatched_token" and stats["unmatched_tokens"] == 1
    assert stats["unmatched_exb_events"] == {"other": 1}


def test_an_alignment_that_depends_on_the_direction_excludes_its_ip() -> None:
    first, differ = pr.align_both(["и", "дом"], ["и", "кот", "и", "дом"])
    assert first == {0: 0, 1: 3} and differ == {0}
    events = [pr.Event(i * 0.5, i * 0.5 + 0.5, t) for i, t in enumerate(["и", "кот", "и", "дом"])]
    graph = [pr.GraphToken(e.text, e.start, e.end, "None") for e in events]
    (ip,), stats = joined(texts=["и", "дом"], events=events, graph=graph)
    assert ip["reason"] == "ambiguous_match" and stats["ambiguous_tokens"] == 1


def test_a_label_the_graph_does_not_have_excludes_its_ip() -> None:
    table, events, graph = doc_parts(["мама", "мыла"], labels=["H*", "None"])
    graph[0] = pr.GraphToken("мама", 0.05, 0.45, "L*")
    (ip,), _ = pr.join_doc("RUmo11FR_fsR", table, events, graph, 60.0, "RUmo11FR")
    assert ip["reason"] == "label_mismatch"


def test_an_ambiguous_label_or_a_token_in_two_ips_excludes_the_ips() -> None:
    (ip,), _ = joined(texts=["мама", "мыла"], labels=["L%", "None"])
    assert ip["reason"] == "ambiguous_label"
    table, events, graph = doc_parts(["мама", "мыла", "раму"], ip_values={"1": ["IP"], "2": ["IP"]})
    tokens = table["tokens"]
    table["tokens"] = [tokens[0], pr.Token("11", "мыла", (3,), ("1", "2"), ("None",), ("NOUN",)),
                       pr.Token("12", "раму", (4,), ("2",), ("None",), ("NOUN",))]  # fmt: skip
    ips, _ = pr.join_doc("RUmo11FR_fsR", table, events, graph, 60.0, "RUmo11FR")
    assert [ip["reason"] for ip in ips] == ["ambiguous_ip", "ambiguous_ip"]


def test_an_ip_whose_layer_value_is_not_ip_is_excluded() -> None:
    (ip,), _ = joined(texts=["мама"], ip_values={"1": ["(-)"]})
    assert ip["reason"] == "ip_value"


def test_a_graph_of_another_length_or_without_a_time_gives_no_rupro_times() -> None:
    table, events, graph = doc_parts(["мама", "мыла"])
    (ip,), stats = pr.join_doc("RUmo11FR_fsR", table, events, graph[:1], 60.0, "RUmo11FR")
    assert ip["reason"] == "no_rupro_times" and not stats["graph_counts_equal"]
    graph[1] = pr.GraphToken("мыла", None, None, "None")
    (ip,), _ = pr.join_doc("RUmo11FR_fsR", table, events, graph, 60.0, "RUmo11FR")
    assert ip["reason"] == "no_rupro_times"


def test_a_graph_token_of_another_text_gives_its_token_no_rupro_time_or_label() -> None:
    table, events, graph = doc_parts(["мама", "мыла"], labels=["H*", "None"])
    graph[0] = pr.GraphToken("папа", 0.05, 0.45, "L*")
    (ip,), stats = pr.join_doc("RUmo11FR_fsR", table, events, graph, 60.0, "RUmo11FR")
    assert ip["reasons"] == ["no_rupro_times"] and stats["graph_texts_differ"] == 1
    assert ip["words"][0]["graph_word_pa"] is None and ip["rupro"] is None
    _, stats = joined(texts=["мама", "мыла"])
    assert stats["graph_texts_differ"] == 0


@pytest.mark.parametrize(
    ("events", "graph", "mp3", "reason"),
    [
        ([(0.0, 0.5), (0.5, 1.0)], [(1.2, 1.4), (1.4, 2.1)], 60.0, "interval_disagree"),
        ([(0.0, 0.5), (0.5, 1.0)], [(0.0, 0.5), (0.5, 2.0)], 60.0, None),
        ([(0.0, 0.5), (0.5, 1.0)], [(0.0, 0.5), (0.5, 2.01)], 60.0, "interval_disagree"),
        ([(0.0, 0.5), (0.5, 0.5)], [(0.0, 0.5), (0.5, 0.9)], 60.0, "bad_interval"),
        ([(0.0, 0.6), (0.5, 1.0)], [(0.0, 0.5), (0.5, 0.9)], 60.0, "bad_interval"),
        ([(0.0, 0.5), (0.5, 1.0)], [(0.0, 0.5), (0.6, 0.6)], 60.0, "bad_interval"),
        ([(0.0, 0.5), (0.5, 1.0)], [(0.0, 0.5), (0.5, 0.9)], 0.95, "bad_interval"),
        ([(0.0, 0.5), (0.5, 1.0)], [(0.0, 0.5), (0.5, 0.9)], None, "bad_interval"),
        ([(0.0, 0.5), (None, 1.0)], [(0.0, 0.5), (0.5, 0.9)], 60.0, "bad_interval"),
    ],
)
def test_the_interval_rule(events, graph, mp3, reason) -> None:
    texts = ["мама", "мыла"]
    events = [pr.Event(s, e, t) for (s, e), t in zip(events, texts, strict=True)]
    graph = [pr.GraphToken(t, s, e, "None") for (s, e), t in zip(graph, texts, strict=True)]
    (ip,), _ = joined(texts=texts, events=events, graph=graph, mp3=mp3)
    assert ip["reason"] == reason


def test_the_three_import_check_documents_never_enter_the_sample() -> None:
    assert pr.IMPORT_CHECK_DOCS == ("RUmo01FR_fsR", "RUmo01FR_isR", "RUmo02FR_fsR")
    for doc in pr.IMPORT_CHECK_DOCS:
        (ip,), _ = joined(doc=doc, texts=["мама", "мыла"], labels=["H*", "None"])
        assert ip["reasons"] == ["import_check"]
    (ip,), _ = joined(doc="RUmo02FR_isR", texts=["мама"])
    assert ip["reasons"] == []


def test_summarize_counts_kept_material_and_every_exclusion() -> None:
    clean, _ = joined(texts=["мама", "мыла", "раму"], labels=["H*", "None", "None"])
    check, _ = joined(doc="RUmo01FR_fsR", texts=["мама"])
    stats = {"a": {"tokens": 3, "exb_events": 3, "exb_pauses": 0, "matched_tokens": 3,
                   "unmatched_tokens": 0, "ambiguous_tokens": 0, "graph_counts_equal": True,
                   "graph_texts_differ": 1, "unmatched_exb_events": {"hesitation": 2}}}  # fmt: skip
    found = pr.summarize(clean + check, stats)
    assert found["kept"]["ips"] == 1 and found["kept"]["pairs"] == 2
    assert (
        found["kept"]["speakers_with_pairs"] == 1
        and found["excluded_by_reason"]["import_check"] == 1
    )
    assert sum(found["excluded_by_reason"].values()) == 1
    assert found["unmatched_exb_events"] == {"hesitation": 2}
    assert found["graph_texts_differ"] == 1


# --- sources on disk and fetch -------------------------------------------------------------


def test_verify_names_a_changed_or_missing_source(tmp_path, monkeypatch) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    station_sources(sources)
    pin_sources(monkeypatch, sources)
    assert pr.verify(sources)[0] == []
    exb = pr.member_path(sources, f"{pr.EXB_PREFIX}{DOCS[0]}.exb")
    exb.write_bytes(exb.read_bytes() + b" ")
    assert pr.verify(sources)[0] == ["the EXB and meta files differ from the pinned list"]
    exb.unlink()
    assert any("missing" in p for p in pr.verify(sources)[0])
    (sources / pr.TABLE_FILE).write_bytes(b"x")
    assert pr.verify(sources)[0] == [f"{pr.TABLE_FILE}: size or SHA-256 differs from the pin"]


def test_the_source_pins_are_written_in_dependencies() -> None:
    text = (REPO / "docs/dependencies.md").read_text(encoding="utf-8")
    for value in (pr.TABLE_SHA256, pr.RUEG_MD5, pr.DOCS_SHA256, pr.EXB_LIST_SHA256, pr.OSF_URL,
                  pr.RUEG_URL, *(sha for _, sha in pr.PINNED_MEMBERS.values())):  # fmt: skip
        assert value in text
    sizes = (pr.TABLE_BYTES, pr.RUEG_BYTES, *(size for size, _ in pr.PINNED_MEMBERS.values()))
    for size in sizes:
        assert f"| {size} |" in text, size


class Response:
    def __init__(self, data: bytes, status: int = 200):
        self.data, self.status, self.at = data, status, 0

    def read(self, size: int = -1) -> bytes:
        size = len(self.data) - self.at if size < 0 else size
        chunk = self.data[self.at : self.at + size]
        self.at += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def fake_opener(table: bytes, archive: bytes, log: list):
    def opener(request, timeout):
        if isinstance(request, str):
            log.append("table")
            return Response(table)
        first, last = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        log.append((first, last))
        return Response(archive[first : last + 1], 206)

    return opener


def remote_archive(sources: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("RUEG-1.0_corpora/core/de/huge.bin", os.urandom(1 << 16))
        for member in (pr.LICENSE_MEMBER, pr.RUPRO_MEMBER, *pr.exb_members(list(DOCS))):
            archive.write(pr.member_path(sources, member), member)
    return buffer.getvalue()


def test_fetch_reads_only_the_pinned_members_by_range_and_checks_them(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    station_sources(origin)
    pin_sources(monkeypatch, origin)
    archive = remote_archive(origin)
    monkeypatch.setattr(pr, "RUEG_BYTES", len(archive))
    log: list = []
    target = tmp_path / "sources"
    found = pr.fetch(target, fake_opener((origin / pr.TABLE_FILE).read_bytes(), archive, log))
    assert found == {"fetched": 2 + 2 * len(DOCS) + 1, "verified": True}
    assert pr.verify(target)[0] == []
    assert not (target / pr.RUEG_DIR / "RUEG-1.0_corpora/core/de").exists()
    read = sum(last - first + 1 for first, last in (e for e in log if e != "table"))
    assert read < len(archive)
    log.clear()
    assert pr.fetch(target, fake_opener(b"", archive, log))["fetched"] == 0


def test_fetch_refuses_a_member_with_another_sha256_and_keeps_no_part(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    station_sources(origin)
    pin_sources(monkeypatch, origin)
    archive = remote_archive(origin)
    monkeypatch.setattr(pr, "RUEG_BYTES", len(archive))
    size, _ = pr.PINNED_MEMBERS[pr.LICENSE_MEMBER]
    monkeypatch.setitem(pr.PINNED_MEMBERS, pr.LICENSE_MEMBER, (size, "0" * 64))
    target = tmp_path / "sources"
    with pytest.raises(pr.RuproError, match="not the pin"):
        pr.fetch(target, fake_opener((origin / pr.TABLE_FILE).read_bytes(), archive, []))
    licence = pr.member_path(target, pr.LICENSE_MEMBER)
    assert not licence.exists() and not list(licence.parent.glob(".*part"))


def test_a_range_answer_without_206_is_refused() -> None:
    reader = pr.RangeReader("https://example.invalid/x", 10, lambda *_, **__: Response(b"x" * 10))
    with pytest.raises(pr.RuproError, match="206"):
        reader.read(4)


# --- freshness -----------------------------------------------------------------------------


def test_check_fresh_finds_a_reused_hash_and_a_document_named_in_pilot_1(tmp_path, monkeypatch):
    settings = make_config(tmp_path)
    provenance = settings.data_root / prosody.PILOT_PROVENANCE
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"clips": [{"member": "x/RUmo12MR_fsR.wav"}]}))
    monkeypatch.setattr(pc, "reference_hashes", lambda _: {"rinco": set(), "pilot_1": set()})
    with pytest.raises(pr.RuproError, match="prosody-confirm-1/prepare.json is missing"):
        pr.check_fresh(settings, {"a.mp3": "6" * 64}, ["RUmo11FR_fsR"])
    confirm = settings.data_root / pc.OUTPUT_DIR / pc.PREPARE_FILE
    confirm.parent.mkdir(parents=True)
    fresh = [{"member_sha256": "1" * 64, "wav_sha256": "2" * 64, "pcm_sha256": "3" * 64}]
    confirm.write_text(
        json.dumps({"fresh_clips": fresh, "calibration": [{"wav_sha256": "4" * 64}]})
    )
    monkeypatch.setattr(pc, "reference_hashes", lambda _: {"rinco": {"5" * 64}, "pilot_1": set()})
    clean = pr.check_fresh(settings, {"a.mp3": "6" * 64}, ["RUmo11FR_fsR"])
    assert clean["fresh"] and clean["references"]["prosody_confirm_1"] == 4
    for reused in ("2" * 64, "4" * 64, "5" * 64):
        found = pr.check_fresh(settings, {"a.mp3": "6" * 64, "b.pcm": reused}, ["RUmo11FR_fsR"])
        assert not found["fresh"] and found["overlap"] == ["b.pcm"]
    named = pr.check_fresh(settings, {"a.mp3": "6" * 64}, ["RUmo12MR_fsR"])
    assert not named["fresh"] and named["docs_in_pilot_1"] == ["RUmo12MR_fsR"]


# --- the metric and the verdict ------------------------------------------------------------


def record(speaker: str, doc: str, wins: list[bool]) -> dict:
    """One IP: one accented word and one plain word per entry; True — the accented one is higher."""
    words = []
    for win in wins:
        words.append(
            {
                "marked": True,
                "score": {"index": 1.0 if win else -1.0, "z_f0": 0.0, "z_rms": 0.0, "z_dur": 0.0},
            }
        )
        words.append({"marked": False, "score": {"index": 0.0, "z_f0": 0.0, "z_rms": 0.0,
                                                  "z_dur": 0.0}})  # fmt: skip
    return {"member": speaker, "folder": doc.rsplit("_", 1)[1], "doc": doc, "words": words,
            "asr_words": 0, "asr_excluded": 0, "asr_unmatched": 0}  # fmt: skip


def test_the_ci_resamples_speakers_with_both_of_their_documents() -> None:
    records = [
        record("RUmo11FR", "RUmo11FR_fsR", [True] * 4),
        record("RUmo11FR", "RUmo11FR_isR", [False] * 4),
        record("RUmo12MR", "RUmo12MR_fsR", [True, False]),
        record("RUmo13FR", "RUmo13FR_isR", []),
    ]
    found = pr.rupro_metric(records)
    assert found["speakers"] == 3 and found["ips"] == 4 and found["ips_with_pairs"] == 3
    assert found["speakers_with_pairs"] == 2
    by_speaker = [(4.0 * 1 + 0.0, 8), (1.0, 2), (0.0, 0)]
    assert found["ci95"] == list(prosody.bootstrap_ci(by_speaker))
    by_doc = [(4.0, 4), (0.0, 4), (1.0, 2), (0.0, 0)]
    assert found["ci95_by_doc"] == list(prosody.bootstrap_ci(by_doc))
    assert found["ci95"] != found["ci95_by_doc"] and found["auc"] == pytest.approx(0.5)
    assert set(found["auc_by_setting"]) == {"fsR", "isR"}


def verdict_of(**changes) -> str:
    found = {"coverage": 0.8, "pairs": 1200, "ips_with_pairs": 60, "speakers_with_pairs": 20,
             "auc": 0.75, "ci95": [0.70, 0.8]} | changes  # fmt: skip
    return pr.decide(found)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, pr.PASS),
        ({"coverage": 0.699}, pr.INCONCLUSIVE),
        ({"pairs": 1199}, pr.INCONCLUSIVE),
        ({"ips_with_pairs": 59}, pr.INCONCLUSIVE),
        ({"speakers_with_pairs": 19}, pr.INCONCLUSIVE),
        ({"coverage": 0.5, "auc": 0.9}, pr.INCONCLUSIVE),
        ({"auc": 0.749}, pr.FAIL),
        ({"ci95": [0.699, 0.8]}, pr.FAIL),
        ({"auc": None, "ci95": None}, pr.FAIL),
    ],
)
def test_decide_follows_the_rule(changes, expected) -> None:
    assert verdict_of(**changes) == expected


# --- frozen trees --------------------------------------------------------------------------

OLD = "prosody-confirm-1"


def git(tree: Path, *args: str) -> str:
    command = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def synthetic_tree(tmp_path: Path, experiment: str = OLD) -> Path:
    tree = tmp_path / "tree"
    for name in pr.EXPERIMENTS[experiment]["modules"]:
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n")
    for document in pr.EXPERIMENTS[experiment]["documents"]:
        (tree / document).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / document, tree / document)
    (tree / "uv.lock").write_text("lock\n")
    (tree / "pyproject.toml").write_text("[project]\n")
    git(tree, "init", "-q")
    git(tree, "add", ".")
    git(tree, "commit", "-q", "-m", "x")
    return tree


def fake_probe(outside: Path | None = None):
    def probe(tree: Path, modules: list[str]) -> dict:
        found = {}
        for name, module in pr.MODULES.items():
            if module in modules:
                found[module] = str((outside or tree) / name)
        return {"modules": found, "prefix": str(tree / ".venv"), "python": "3.13.16",
                "versions": {"faster-whisper": "1.2.1", "ctranslate2": "4.9.0"}}  # fmt: skip

    return probe


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A clean git tree of the old experiment, frozen, with pins of its synthetic modules."""
    monkeypatch.setitem(pr.EXPERIMENTS[OLD], "commit", None)
    tree = synthetic_tree(tmp_path)
    data_root = tmp_path / "data"
    (data_root / OLD / "sanity").mkdir(parents=True)
    (data_root / OLD / "sanity" / "attempt-1.json").write_text("{}")
    pins = {m: hashlib.sha256((tree / m).read_bytes()).hexdigest() for m in pr.PINNED_FILES}
    pr.freeze(tree, OLD, data_root, probe_=fake_probe())
    return tree, data_root, pins


@needs_git
def test_verify_frozen_passes_an_unchanged_tree(frozen) -> None:
    tree, data_root, pins = frozen
    found = pr.verify_frozen(tree, data_root, pins, probe_=fake_probe())
    assert found["verified"], found["problems"]
    assert found["modules"] == 4 and found["data_files"] == 1 and found["uv_lock_sha256"]
    manifest = json.loads((tree / pr.FROZEN_FILE).read_text())
    assert manifest["files"]["uv.lock"] == hashlib.sha256(b"lock\n").hexdigest()
    with pytest.raises(pr.RuproError, match="frozen once"):
        pr.freeze(tree, OLD, data_root, probe_=fake_probe())


@needs_git
def test_verify_frozen_fails_a_module_imported_from_outside_the_tree(frozen, tmp_path) -> None:
    tree, data_root, pins = frozen
    found = pr.verify_frozen(tree, data_root, pins, probe_=fake_probe(outside=tmp_path / "other"))
    assert not found["verified"]
    assert sum("imported from outside the tree" in p for p in found["problems"]) == 4


@needs_git
def test_verify_frozen_fails_a_changed_file_a_foreign_pin_and_changed_data(frozen) -> None:
    tree, data_root, pins = frozen
    (tree / "src/sound_scout/asr.py").write_text("# changed\n")
    (data_root / OLD / "sanity" / "attempt-1.json").write_text("{ }")
    found = pr.verify_frozen(tree, data_root, pins | {pr.CONFIRM_PATH: "0" * 64}, fake_probe())
    problems = found["problems"]
    assert not found["verified"]
    assert "the tree has changes to tracked files" in problems
    assert "src/sound_scout/asr.py: SHA-256 differs from the manifest" in problems
    assert f"{pr.CONFIRM_PATH}: manifest SHA-256 is not the 0034 pin" in problems
    assert "sound_scout.asr: the imported file differs from the manifest" in problems
    assert "the data files differ from the manifest" in problems


@needs_git
def test_verify_frozen_with_the_real_probe_rejects_a_foreign_interpreter(frozen) -> None:
    tree, data_root, pins = frozen
    (tree / ".venv/bin").mkdir(parents=True)
    (tree / ".venv/bin/python").symlink_to(sys.executable)
    found = pr.verify_frozen(tree, data_root, pins)
    assert not found["verified"]
    assert any("interpreter" in p or "outside the tree" in p for p in found["problems"])


@needs_git
def test_freeze_refuses_a_tree_at_another_commit_or_with_changes(tmp_path, monkeypatch) -> None:
    tree = synthetic_tree(tmp_path)
    with pytest.raises(pr.RuproError, match="pinned to"):
        pr.freeze(tree, OLD, tmp_path, probe_=fake_probe())
    monkeypatch.setitem(pr.EXPERIMENTS[OLD], "commit", None)
    (tree / "uv.lock").write_text("other\n")
    with pytest.raises(pr.RuproError, match="changes"):
        pr.freeze(tree, OLD, tmp_path, probe_=fake_probe())


def test_the_pins_are_those_of_0034_and_hold_in_this_tree() -> None:
    assert {k: v for k, v in pr.PINNED_FILES.items() if k != pr.CONFIRM_PATH} == pc.FROZEN_SHA256
    assert pr.CONFIRM_SHA256 in (REPO / pr.CONFIRM_DECISION).read_text(encoding="utf-8")
    for name, sha in pr.PINNED_FILES.items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == sha, name
    assert pr.EXPERIMENTS[OLD]["commit"] == "f37df5a395be79b8a52c105fa2ea891c99bcc052"


def test_frozen_state_refuses_a_tree_without_a_manifest_or_of_another_experiment(tmp_path):
    assert pr.frozen_state(tmp_path) == ([f"not a frozen tree: {pr.FROZEN_FILE} is missing"], {})
    (tmp_path / pr.FROZEN_FILE).write_text(json.dumps({"experiment": OLD}))
    problems, info = pr.frozen_state(tmp_path)
    assert info == {} and "not prosody-rupro-1" in problems[0]


def test_frozen_state_names_modules_loaded_from_outside_the_tree(tmp_path) -> None:
    manifest = {"experiment": pr.OUTPUT_DIR, "commit": "c" * 40,
                "modules": dict(pr.MODULES), "files": {}}  # fmt: skip
    (tmp_path / pr.FROZEN_FILE).write_text(json.dumps(manifest))
    problems, info = pr.frozen_state(tmp_path)
    assert info["commit"] == "c" * 40
    assert sum("loaded from outside the frozen tree" in p for p in problems) == len(pr.MODULES)
    assert "the interpreter is not the frozen tree's .venv" in problems


def test_this_tree_runs_no_step_of_the_run(tmp_path) -> None:
    settings = make_config(tmp_path)
    for step in (pr.prepare, pr.run_check_fresh, pr.run_sanity, pr.run_main):
        with pytest.raises(pr.RuproError, match="frozen tree"):
            step(settings, REPO / pr.DECISION_DOC, tree=tmp_path)
    assert not (settings.data_root / pr.OUTPUT_DIR).exists()


# --- the steps on a synthetic station ------------------------------------------------------


@needs_ffmpeg
def test_survey_counts_the_import_and_cuts_nothing(station) -> None:
    found = pr.survey(station, REPO / pr.DECISION_DOC)
    out = station.data_root / pr.OUTPUT_DIR
    assert found["kept"]["ips"] == 4 and found["kept"]["pairs"] == 12 and found["fresh"]
    assert found["excluded_by_reason"]["import_check"] == 1
    assert found["kept"]["speakers_with_pairs"] == 2 and found["table_tokens"] == 20
    assert found["unmatched_exb_events"] == {"hesitation": 5}
    assert not (out / pr.CLIPS_DIR).exists() and not (out / pr.PREPARE_FILE).exists()
    (survey,) = (out / pr.SURVEY_DIR).glob("*.json")
    record = json.loads(survey.read_text())
    word = record["ips"][0]["words"][1]
    assert word["marked"] == 1 and word["word_pa"] == "H*" and word["graph_word_pa"] == "H*"
    assert word["rows"] and word["token_id"]


@needs_ffmpeg
def test_prepare_sanity_and_run_give_pass_once(station, monkeypatch) -> None:
    decision = REPO / pr.DECISION_DOC
    prepared = pr.prepare(station, decision)
    out = station.data_root / pr.OUTPUT_DIR
    assert len(prepared["clips"]) == 4 and prepared["fresh_check"]["fresh"]
    assert prepared["frozen"] == FROZEN_INFO
    recorded = (out / pr.PREPARE_SHA_FILE).read_text().strip()
    assert recorded == pr.sha256_file(out / pr.PREPARE_FILE)
    with pytest.raises(pr.RuproError, match="once"):
        pr.prepare(station, decision)
    assert pr.run_check_fresh(station, decision)["files"] == "ok"
    with pytest.raises(pr.RuproError, match="sanity"):
        pr.run_main(station, decision, transcribe=run_transcriber)
    passed_sanity(monkeypatch)
    assert pr.run_sanity(station, decision)["passed"]
    result = pr.run_main(station, decision, transcribe=run_transcriber)
    assert result["verdict"] == pr.PASS, result
    assert result["auc"] == 1.0 and result["coverage"] == 1.0 and result["pairs"] == 12
    assert result["speakers"] == 2 and result["ips_with_pairs"] == 4
    with pytest.raises(pr.RuproError, match="verdict is out"):
        pr.run_main(station, decision, transcribe=run_transcriber)


@needs_ffmpeg
def test_run_gives_fail_when_the_accented_words_are_not_the_prominent_ones(
    station, monkeypatch
) -> None:
    sources = pr.sources_dir(station)
    shutil.rmtree(sources)
    sources.mkdir()
    station_sources(sources, marked=1)
    pin_sources(monkeypatch, sources)
    decision = REPO / pr.DECISION_DOC
    pr.prepare(station, decision)
    passed_sanity(monkeypatch)
    pr.run_sanity(station, decision)
    assert pr.run_main(station, decision, transcribe=run_transcriber)["verdict"] == pr.FAIL


@needs_ffmpeg
def test_a_changed_prepare_json_or_clip_makes_the_run_invalid(station, monkeypatch) -> None:
    decision = REPO / pr.DECISION_DOC
    pr.prepare(station, decision)
    passed_sanity(monkeypatch)
    pr.run_sanity(station, decision)
    out = station.data_root / pr.OUTPUT_DIR
    clip = next((out / pr.CLIPS_DIR).glob("*.wav"))
    clip.write_bytes(clip.read_bytes() + b"\0\0")
    result = pr.run_main(station, decision, transcribe=run_transcriber)
    assert result["verdict"] == pr.INVALID and "SHA-256 differs" in result["reason"]
    assert not (out / pr.MAIN_DIR).exists()


@needs_ffmpeg
def test_wrong_frozen_pins_make_prepare_invalid_before_any_clip(station, monkeypatch) -> None:
    monkeypatch.setattr(pr, "frozen_state", lambda tree=None: (["prosody.py: pin"], FROZEN_INFO))
    result = pr.prepare(station, REPO / pr.DECISION_DOC)
    out = station.data_root / pr.OUTPUT_DIR
    assert result["verdict"] == pr.INVALID and "frozen: prosody.py: pin" in result["reason"]
    assert not (out / pr.CLIPS_DIR).exists()
    with pytest.raises(pr.RuproError, match="verdict is out"):
        pr.run_main(station, REPO / pr.DECISION_DOC)


@needs_ffmpeg
def test_two_failed_sanity_attempts_give_invalid(station, monkeypatch) -> None:
    decision = REPO / pr.DECISION_DOC
    pr.prepare(station, decision)
    monkeypatch.setattr(pc, "sanity", lambda *_: {"passed": False})
    assert not pr.run_sanity(station, decision)["passed"]
    assert not pr.run_sanity(station, decision)["passed"]
    result = json.loads((station.data_root / pr.OUTPUT_DIR / pr.RESULT_FILE).read_text())
    assert result["verdict"] == pr.INVALID


@needs_ffmpeg
def test_a_main_run_left_without_a_result_is_inconclusive_and_not_rerun(station, monkeypatch):
    decision = REPO / pr.DECISION_DOC
    pr.prepare(station, decision)
    passed_sanity(monkeypatch)
    pr.run_sanity(station, decision)

    def broken(samples, run_dir):
        raise RuntimeError("boom")

    assert pr.run_main(station, decision, transcribe=broken)["verdict"] == pr.INVALID
    out = station.data_root / pr.OUTPUT_DIR
    (out / pr.RESULT_FILE).unlink()
    assert pr.run_main(station, decision)["verdict"] == pr.INCONCLUSIVE


@needs_ffmpeg
@pytest.mark.parametrize("killed", [1, 2])
def test_a_sanity_attempt_killed_before_its_record_counts_as_used(
    station, monkeypatch, killed
) -> None:
    decision = REPO / pr.DECISION_DOC
    pr.prepare(station, decision)
    out = station.data_root / pr.OUTPUT_DIR
    for number in range(1, killed + 1):  # killed: a work directory and no record
        (out / pr.SANITY_DIR / f"work-{number}").mkdir(parents=True)
    calls: list = []
    monkeypatch.setattr(pc, "sanity", lambda *_: calls.append(1) or {"passed": False})
    found = pr.run_sanity(station, decision)
    assert len(calls) == 2 - killed
    if killed == 1:
        assert found["attempt"] == 2 and not found["passed"]
    else:
        assert found["verdict"] == pr.INVALID and "2 попытки" in found["reason"]
    result = json.loads((out / pr.RESULT_FILE).read_text())
    assert result["verdict"] == pr.INVALID
    assert not (out / pr.SANITY_DIR / "attempt-1.json").exists()
    with pytest.raises(pr.RuproError, match="verdict is out"):
        pr.run_sanity(station, decision)


@needs_ffmpeg
def test_a_main_run_stopped_by_the_wall_time_ceiling_is_inconclusive(station, monkeypatch):
    decision = REPO / pr.DECISION_DOC
    pr.prepare(station, decision)
    passed_sanity(monkeypatch)
    pr.run_sanity(station, decision)
    calls: list = []

    def transcriber(samples, run_dir):
        calls.append(1)
        return run_transcriber(samples, run_dir)

    result = pr.run_main(station, decision, transcribe=transcriber, wall_seconds=-1)
    assert result["verdict"] == pr.INCONCLUSIVE and "потолком" in result["reason"]
    assert calls == [] and result["ips_done"] == 0


@needs_ffmpeg
def test_a_main_run_out_of_memory_is_inconclusive_and_not_rerun(station, monkeypatch) -> None:
    decision = REPO / pr.DECISION_DOC
    pr.prepare(station, decision)
    passed_sanity(monkeypatch)
    pr.run_sanity(station, decision)

    def broken(samples, run_dir):
        raise MemoryError

    result = pr.run_main(station, decision, transcribe=broken)
    assert result["verdict"] == pr.INCONCLUSIVE and "памяти" in result["reason"]
    with pytest.raises(pr.RuproError, match="verdict is out"):
        pr.run_main(station, decision, transcribe=run_transcriber)


def test_the_cli_refuses_fetch_without_the_network_and_prints_no_text(
    tmp_path, monkeypatch, capsys
) -> None:
    settings = make_config(tmp_path)
    monkeypatch.setattr(config, "load", lambda: settings)
    assert pr.main(["fetch"]) == 2
    assert "needs the network" in capsys.readouterr().err
    assert pr.main(["verify-frozen", str(tmp_path)]) == 2
    assert f"not a frozen tree: {pr.FROZEN_FILE} is missing" in capsys.readouterr().err


# --- stage-1 criteria of #199 --------------------------------------------------------------

REAL_FROZEN_STATE = pr.frozen_state


def rule_block() -> str:
    text = (REPO / pr.DECISION_DOC).read_text(encoding="utf-8")
    return text[text.index(pr.RULE_BEGIN) : text.index(pr.RULE_END)]


def test_criterion_1_the_rule_block_records_every_part_of_the_protocol() -> None:
    text = (REPO / pr.DECISION_DOC).read_text(encoding="utf-8")
    assert (
        "## 2. Целевая величина\n\nНаличие pitch accent на слове по ручной разметке RuPro" in text
    )
    assert text.index("## 2. Целевая величина") < text.index(pr.RULE_BEGIN)
    block = rule_block()
    parts = (
        "Целевая величина. Наличие pitch accent на слове по word_pa RuPro",
        "не воспринимаемая выделенность и не смысловой акцент",
        "HEAD дерева равен коммиту\n  манифеста",
        "SHA-256 каждого файла манифеста совпадает",
        "Правило 0/1 (word_pa).",
        "80 записей, 40 говорящих",  # sample
        "пар «слово 1 / слово 0 одной IP»",  # unit of comparison
        "Исключение IP",
        "Свежесть (check-fresh).",
        "AUC",
        "кластерный бутстреп по говорящим",
        "1000 повторов, seed 0",
        "coverage < 0,70, пар < 1 200",
        "IP с парами < 60 или говорящих с парами < 20",
        "PASS, если AUC >= 0,75 и нижняя граница CI >= 0,70",
        "FAIL — иначе",
        "1. INVALID, если:",
        "2. INCONCLUSIVE, если",
    )
    for part in parts:
        assert part in block, part
    for reason in pr.REASONS:
        assert f"  {reason} " in block, reason
    assert block.index("1. INVALID") < block.index("2. INCONCLUSIVE") < block.index("3. PASS")


def test_criterion_2_the_doc_calls_the_old_tree_a_preservation_not_a_reproduced_inference():
    text = (REPO / pr.DECISION_DOC).read_text(encoding="utf-8")
    assert "Это сохранение файлов и\nокружения. Это не воспроизведённый inference" in text
    assert pr.EXPERIMENTS[OLD]["commit"] in text
    for name, sha in pr.PINNED_FILES.items():
        assert f"| `{name}` | `{sha}`" in text, name


def linked_tree(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """A clean git tree whose module files and .venv are links to what this process loaded."""
    tree = tmp_path / "frozen-rupro"
    for name in pr.MODULES:
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).symlink_to(REPO / name)
    (tree / "uv.lock").write_text("lock\n")
    git(tree, "init", "-q")
    git(tree, "add", "src", "uv.lock")
    git(tree, "commit", "-q", "-m", "x")
    (tree / ".venv").symlink_to(Path(sys.prefix).resolve())
    shas = {name: pr.sha256_file(tree / name) for name in (*pr.MODULES, "uv.lock")}
    manifest = {"experiment": pr.OUTPUT_DIR, "commit": git(tree, "rev-parse", "HEAD").strip(),
                "modules": dict(pr.MODULES), "files": shas | (files or {})}  # fmt: skip
    (tree / pr.FROZEN_FILE).write_text(json.dumps(manifest))
    return tree


@needs_git
def test_criterion_6_frozen_state_accepts_the_loaded_tree_and_refuses_a_wrong_pin(
    tmp_path, monkeypatch
) -> None:
    assert sys.modules["sound_scout.asr"] and sys.modules["sound_scout.acoustics"]
    tree = linked_tree(tmp_path)
    problems, info = REAL_FROZEN_STATE(tree)
    assert problems == [] and info["commit"] == git(tree, "rev-parse", "HEAD").strip()
    monkeypatch.setitem(pr.PINNED_FILES, "src/sound_scout/asr.py", "0" * 64)
    problems, _ = REAL_FROZEN_STATE(tree)
    assert problems == ["src/sound_scout/asr.py: SHA-256 is not the 0034 pin"]
    monkeypatch.undo()
    other = linked_tree(tmp_path / "b", {"src/sound_scout/acoustics.py": "1" * 64})
    problems, _ = REAL_FROZEN_STATE(other)
    assert "sound_scout.acoustics: SHA-256 differs from the manifest" in problems
    assert "src/sound_scout/acoustics.py: SHA-256 is not the 0034 pin" in problems


@needs_git
def test_frozen_state_checks_the_commit_the_clean_tree_and_every_manifest_file(tmp_path) -> None:
    tree = linked_tree(tmp_path)
    (tree / "uv.lock").write_text("other\n")
    problems, _ = REAL_FROZEN_STATE(tree)
    assert "the frozen tree has changes to tracked files" in problems
    assert "uv.lock: SHA-256 differs from the manifest" in problems
    git(tree, "commit", "-q", "-am", "y")
    problems, _ = REAL_FROZEN_STATE(tree)
    assert "the frozen tree is not at the manifest commit" in problems
    assert "the frozen tree has changes to tracked files" not in problems


@needs_ffmpeg
@needs_git
def test_criterion_6_prepare_from_a_tree_with_a_wrong_pin_is_invalid_with_no_clip(
    station, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pr, "frozen_state", REAL_FROZEN_STATE)
    tree = linked_tree(tmp_path / "t", {"src/sound_scout/eval/prosody.py": "2" * 64})
    result = pr.prepare(station, REPO / pr.DECISION_DOC, tree=tree)
    out = station.data_root / pr.OUTPUT_DIR
    assert result["verdict"] == pr.INVALID
    assert "src/sound_scout/eval/prosody.py: SHA-256 is not the 0034 pin" in result["reason"]
    assert not (out / pr.CLIPS_DIR).exists() and not (out / pr.PREPARE_FILE).exists()


@needs_ffmpeg
def test_criterion_5_the_run_scores_with_the_series_3_pipeline_and_asr_word_times(
    station, monkeypatch
) -> None:
    decision = REPO / pr.DECISION_DOC
    prepared = pr.prepare(station, decision)
    passed_sanity(monkeypatch)
    pr.run_sanity(station, decision)
    tracked, scored, heard = [], [], []
    real_track, real_score = prosody.track, prosody.score_utterance

    def track(samples):
        tracked.append(len(samples))
        return real_track(samples)

    def score(track_, asr_words, words):
        scored.append((asr_words, words))
        return real_score(track_, asr_words, words)

    def transcriber(samples, run_dir):
        found = run_transcriber(samples, run_dir)
        heard.append((len(samples), found["words"]))
        return found

    monkeypatch.setattr(prosody, "track", track)
    monkeypatch.setattr(prosody, "score_utterance", score)
    monkeypatch.setattr(prosody, "local_transcriber", lambda _: transcriber)
    result = pr.run_main(station, decision)
    assert result["verdict"] == pr.PASS
    kept = [ip for ip in prepared["ips"] if not ip["reasons"]]
    assert len(scored) == len(tracked) == len(heard) == len(kept) == 4
    for ip, (asr_words, words), (length, said) in zip(kept, scored, heard, strict=True):
        start, end = ip["cut"]
        assert length == round(end * RATE) - round(start * RATE)
        assert asr_words == said  # word times come from ASR only
        assert all(set(w) == {"form", "marked"} for w in words)  # EXB/RuPro give no word time
