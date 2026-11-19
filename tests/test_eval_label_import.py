"""Import of a returned blind labeling package (#91): synthetic packages in tmp_path only."""

import hashlib
import json
from pathlib import Path

import pytest

from sound_scout.cli import main
from sound_scout.eval import label_import
from sound_scout.eval.label_import import ImportRefused, LabelImportError, import_submission

PRIVATE_MARKER = "PRIVATE-STATION-PATH-MARKER"
REVIEW_HEADER = "clip_id,audio_path,duration_s,decision,model_outputs_seen,annotator,date,note"
RECORDS = (
    ("A001", "pilot-1", "ctl-sil-01", "dev", 10.0),
    ("A002", "pilot-1", "h-pron-02", "heldout", 12.5),
    ("A003", "pilot-1", "d-stress-03", "dev", 8.0),
    ("D001", "extra-dev", "x-rep-01", "dev", 6.0),
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def alias_map() -> dict:
    return {
        "kind": label_import.ALIAS_MAP_KIND,
        "source_splits_unchanged": True,
        "no_heldout_approval": True,
        "records": [
            {
                "alias": alias,
                "source_corpus": corpus,
                "original_record": {
                    "id": original,
                    "path": f"/{PRIVATE_MARKER}/clips/{original}.wav",
                    "sha256": digest(original),
                    "duration_s": duration,
                    "split": split,
                    "source_group": "g1",
                },
                "source": f"/{PRIVATE_MARKER}/blind/{alias}.wav",
                "source_label": f"/{PRIVATE_MARKER}/labels/{original}.json",
                "pcm_sha256": digest(alias),
                "frames": int(duration * 16000),
            }
            for alias, corpus, original, split, duration in RECORDS
        ],
    }


def template(alias: str) -> dict:
    _, _, original, _, duration = next(r for r in RECORDS if r[0] == alias)
    return {
        "schema_version": "1.0",
        "clip_id": alias,
        "clip_sha256": digest(original),
        "clip_duration_s": duration,
        "annotator": "",
        "annotated_at": "",
        "model_outputs_seen": False,
        "events": [],
        "note": None,
    }


def signed(alias: str, events: list | None = None, **extra: object) -> dict:
    return {
        **template(alias),
        "annotator": "listener-a",
        "annotated_at": "2026-11-20",
        "events": events or [],
        "note": "проверено",
        **extra,
    }


def an_event(start: float = 1.0, end: float = 1.6) -> dict:
    return {
        "category": "pronunciation",
        "start": start,
        "end": end,
        "description": "синтетика: слышно искажение",
        "certainty": "clear",
        "comment": None,
        "second_opinion": None,
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    private = corpus / "pkg" / "operator-private"
    private.mkdir(parents=True)
    (private / "alias-map.json").write_text(json.dumps(alias_map(), indent=2))
    (corpus / "pkg" / "01-main-blind").mkdir()
    (corpus / "pilot-1").mkdir()
    manifest = {
        "schema_version": "1.0",
        "name": "pilot-1",
        "records": [
            {"id": original, "split": split, "sha256": digest(original)}
            for _, name, original, split, _ in RECORDS
            if name == "pilot-1"
        ],
    }
    (corpus / "pilot-1" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return corpus


def submission(tmp_path: Path, files: dict[str, object], review: str | None = None) -> Path:
    """`files` maps a file name under labels/ to a dict (JSON) or raw text."""
    root = tmp_path / f"submission-{len(list(tmp_path.glob('submission-*')))}"
    (root / "labels").mkdir(parents=True)
    for name, content in files.items():
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        (root / "labels" / name).write_text(text, encoding="utf-8")
    (root / "START.md").write_text("not read")
    if review is not None:
        (root / "review.csv").write_text(review, encoding="utf-8")
    return root


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def refuse(corpus: Path, sub: Path, capsys) -> str:
    before = snapshot(corpus)
    assert main(["eval", "import-labels", str(sub), "--package", str(corpus / "pkg")]) == 1
    assert snapshot(corpus) == before
    err = capsys.readouterr().err
    assert "nothing written" in err or "eval_error" in err
    assert PRIVATE_MARKER not in err
    return err


# --- success ----------------------------------------------------------------------------------


def test_import_copies_maps_and_keeps_split(corpus: Path, tmp_path: Path) -> None:
    with_event = signed("A002", [an_event()])
    files = {"A001.json": template("A001"), "A002.json": with_event, "D001.json": signed("D001")}
    review = f"{REVIEW_HEADER}\nA002,clips/A002.wav,12.5,use,no,listener-a,2026-11-20,\n"
    sub = submission(tmp_path, files, review)
    outside_before = {k: v for k, v in snapshot(corpus).items()}

    summary = import_submission(sub, corpus / "pkg")

    out = corpus / "pkg-returns" / "001"
    assert summary.output == "pkg-returns/001" and summary.path == out
    assert (summary.imported, summary.unsigned, summary.already_imported) == (2, 1, 0)
    assert snapshot(out / "submission") == {
        k: v for k, v in snapshot(sub).items() if k.startswith("labels/") or k == "review.csv"
    }
    mapped = sorted(
        p.relative_to(out / "mapped").as_posix() for p in (out / "mapped").rglob("*.json")
    )
    assert mapped == ["extra-dev/x-rep-01.json", "pilot-1/h-pron-02.json"]
    for alias, path in (("A002", "pilot-1/h-pron-02.json"), ("D001", "extra-dev/x-rep-01.json")):
        raw = json.loads((sub / "labels" / f"{alias}.json").read_text())
        copy = json.loads((out / "mapped" / path).read_text())
        assert copy == {**raw, "clip_id": Path(path).stem}
        assert copy["events"] == raw["events"]
        assert {k: v for k, v in copy.items() if k != "clip_id"} == {
            k: v for k, v in raw.items() if k != "clip_id"
        }
    receipt = json.loads((out / "import.json").read_text())
    assert receipt["kind"] == "label_import_receipt" and receipt["package"] == "pkg"
    assert (
        receipt["review_csv_sha256"]
        == hashlib.sha256((sub / "review.csv").read_bytes()).hexdigest()
    )
    by_alias = {r["alias"]: r for r in receipt["records"]}
    assert by_alias["A001"]["status"] == "unsigned" and by_alias["A001"]["mapped_path"] is None
    assert by_alias["A002"]["status"] == "imported"
    assert by_alias["A002"]["mapped_path"] == "mapped/pilot-1/h-pron-02.json"
    for alias, _, original, split, _ in RECORDS[:2] + RECORDS[3:]:
        assert by_alias[alias]["split"] == split and by_alias[alias]["original_id"] == original
    assert by_alias["A002"]["manifest_check"] == "matched"
    assert by_alias["D001"]["manifest_check"] == "absent"
    assert "/" + PRIVATE_MARKER not in (out / "import.json").read_text()
    assert str(tmp_path) not in (out / "import.json").read_text()
    # Nothing outside pkg-returns/ changed: the package, pilot-1 and its manifest stay as they were.
    after = snapshot(corpus)
    assert {k: v for k, v in after.items() if not k.startswith("pkg-returns/")} == outside_before
    assert not list((corpus / "pkg-returns").glob(".*"))


def test_unsigned_template_is_not_no_events(corpus: Path, tmp_path: Path) -> None:
    sub = submission(tmp_path, {"A001.json": template("A001"), "A003.json": signed("A003")})
    summary = import_submission(sub, corpus / "pkg")
    assert not (summary.path / "mapped" / "pilot-1" / "ctl-sil-01.json").exists()
    receipt = json.loads((summary.path / "import.json").read_text())
    unsigned = next(r for r in receipt["records"] if r["alias"] == "A001")
    assert unsigned["status"] == "unsigned" and unsigned["mapped_sha256"] is None


def test_only_templates_is_nothing_to_import(corpus: Path, tmp_path: Path, capsys) -> None:
    sub = submission(tmp_path, {"A001.json": template("A001"), "D001.json": template("D001")})
    err = refuse(corpus, sub, capsys)
    assert "nothing to import" in err
    assert not (corpus / "pkg-returns").exists()


def test_second_submission_goes_to_002_and_repeats_are_already_imported(
    corpus: Path, tmp_path: Path, capsys
) -> None:
    first = submission(tmp_path, {"A003.json": signed("A003")})
    import_submission(first, corpus / "pkg")
    first_snapshot = snapshot(corpus / "pkg-returns" / "001")

    second = submission(tmp_path, {"A003.json": signed("A003"), "D001.json": signed("D001")})
    summary = import_submission(second, corpus / "pkg")
    assert summary.output == "pkg-returns/002"
    assert (summary.imported, summary.already_imported) == (1, 1)
    assert snapshot(corpus / "pkg-returns" / "001") == first_snapshot
    assert not (summary.path / "mapped" / "pilot-1").exists()
    receipt = json.loads((summary.path / "import.json").read_text())
    assert {r["alias"]: r["status"] for r in receipt["records"]} == {
        "A003": "already_imported",
        "D001": "imported",
    }

    changed = submission(tmp_path, {"A003.json": signed("A003", note="другое")})
    err = refuse(corpus, changed, capsys)
    assert "refused: labels/A003.json: duplicate_id" in err

    same_again = submission(tmp_path, {"A003.json": signed("A003")})
    assert "nothing to import" in refuse(corpus, same_again, capsys)


def test_cli_prints_counts_and_relative_output(corpus: Path, tmp_path: Path, capsys) -> None:
    sub = submission(tmp_path, {"A001.json": template("A001"), "A003.json": signed("A003")})
    assert main(["eval", "import-labels", str(sub), "--package", str(corpus / "pkg")]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["imported: 1", "unsigned: 1", "already imported: 0", "output: pkg-returns/001"]


# --- refusal codes, one file each -------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "code", "name"),
    [
        ({"Z999.json": signed("A001")}, "unknown_id", "labels/Z999.json"),
        ({"a001.json": signed("A001")}, "unknown_id", "labels/a001.json"),
        ({"A001.json": "{not json"}, "not_json", "labels/A001.json"),
        ({"A001.json": "[1, 2]"}, "not_json", "labels/A001.json"),
        ({"A001.json": signed("A001", clip_id="A003")}, "id_mismatch", "labels/A001.json"),
        (
            {"A001.json": signed("A001", clip_sha256=digest("other"))},
            "changed_sha256",
            "labels/A001.json",
        ),
        (
            {"A001.json": signed("A001", clip_duration_s=10.5)},
            "changed_duration",
            "labels/A001.json",
        ),
        (
            {"A001.json": {**template("A001"), "events": [an_event()]}},
            "unsigned_with_events",
            "labels/A001.json",
        ),
        (
            {"A001.json": signed("A001", annotated_at="")},
            "invalid_labels",
            "labels/A001.json",
        ),
        (
            {"A001.json": signed("A001", [an_event(2.0, 1.0)])},
            "invalid_labels",
            "labels/A001.json",
        ),
        ({"A001.txt": "x"}, "unexpected_entry", "labels/A001.txt"),
    ],
)
def test_refusal_codes(corpus, tmp_path, capsys, files, code, name) -> None:
    sub = submission(tmp_path, {"A003.json": signed("A003"), **files})
    err = refuse(corpus, sub, capsys)
    assert f"refused: {name}: {code}" in err
    assert "import refused: 1 files, nothing written" in err.splitlines()[-1]
    assert not (corpus / "pkg-returns").exists()
    with pytest.raises(ImportRefused) as raised:
        import_submission(sub, corpus / "pkg")
    assert [r.code for r in raised.value.refusals] == [code]


def test_invalid_labels_lists_validator_errors_without_text(corpus, tmp_path, capsys) -> None:
    bad = an_event(2.0, 1.0)
    bad["description"] = "СЕКРЕТНОЕ ОПИСАНИЕ"
    sub = submission(tmp_path, {"A001.json": signed("A001", [bad])})
    err = refuse(corpus, sub, capsys)
    assert "  $.events[0]: start must be less than end" in err
    assert "СЕКРЕТНОЕ" not in err


def test_subdirectory_and_symlink_are_unexpected(corpus, tmp_path, capsys) -> None:
    sub = submission(tmp_path, {"A003.json": signed("A003")})
    (sub / "labels" / "nested.json").mkdir()
    (sub / "labels" / "A001.json").symlink_to(sub / "labels" / "A003.json")
    err = refuse(corpus, sub, capsys)
    assert "refused: labels/nested.json: unexpected_entry" in err
    assert "refused: labels/A001.json: unexpected_entry" in err


@pytest.mark.parametrize(
    ("record", "detail"),
    [
        ({"id": "ctl-sil-01", "split": "heldout", "sha256": digest("ctl-sil-01")}, "split"),
        ({"id": "ctl-sil-01", "split": "dev", "sha256": digest("x")}, "sha256"),
        ({"id": "something-else", "split": "dev", "sha256": digest("ctl-sil-01")}, "no record"),
    ],
)
def test_manifest_mismatch(corpus, tmp_path, capsys, record, detail) -> None:
    path = corpus / "pilot-1" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["records"] = [r for r in manifest["records"] if r["id"] != "ctl-sil-01"] + [record]
    path.write_text(json.dumps(manifest))
    sub = submission(tmp_path, {"A001.json": signed("A001")})
    err = refuse(corpus, sub, capsys)
    assert "refused: labels/A001.json: manifest_mismatch" in err
    assert detail in err


@pytest.mark.parametrize("content", ["{not json", '{"records": {}}', '{"kind": "x"}', "[]"])
def test_malformed_manifest_is_refused(corpus, tmp_path, capsys, content) -> None:
    (corpus / "pilot-1" / "manifest.json").write_text(content)
    sub = submission(tmp_path, {"A001.json": signed("A001")})
    with pytest.raises(LabelImportError, match="pilot-1/manifest.json: cannot be read"):
        import_submission(sub, corpus / "pkg")
    err = refuse(corpus, sub, capsys)
    assert "pilot-1/manifest.json: cannot be read as a manifest" in err
    assert "no record with this id" not in err


@pytest.mark.parametrize(("decision", "seen"), [("reject", "no"), ("use", "yes"), ("", "true")])
def test_review_conflict(corpus, tmp_path, capsys, decision, seen) -> None:
    review = f"{REVIEW_HEADER}\nA003,clips/A003.wav,8.0,{decision},{seen},listener-a,,\n"
    sub = submission(tmp_path, {"A003.json": signed("A003")}, review)
    err = refuse(corpus, sub, capsys)
    assert "refused: labels/A003.json: review_conflict" in err


def test_review_reject_of_unsigned_template_is_fine(corpus, tmp_path) -> None:
    review = f"{REVIEW_HEADER}\nA001,clips/A001.wav,10.0,reject,no,listener-a,,шум\n"
    sub = submission(tmp_path, {"A001.json": template("A001"), "A003.json": signed("A003")}, review)
    assert import_submission(sub, corpus / "pkg").imported == 1


@pytest.mark.parametrize(
    ("review", "code"),
    [
        ("clip_id,decision\nA003,use\n", "bad_review_csv"),
        (f"{REVIEW_HEADER}\nA003,,,maybe,no,,,\n", "bad_review_csv"),
        (f"{REVIEW_HEADER}\nA003,,,use,perhaps,,,\n", "bad_review_csv"),
        (f"{REVIEW_HEADER}\nA003,,,use\n", "bad_review_csv"),
        (f'{REVIEW_HEADER}\nA003,,,use,no,,,"open\nZ999,,,use,no,,,\n', "bad_review_csv"),
        (f"{REVIEW_HEADER}\nZ999,,,use,no,,,\n", "unknown_id"),
        (f"{REVIEW_HEADER}\nA003,,,use,no,,,\nA003,,,use,no,,,\n", "duplicate_id"),
    ],
)
def test_review_csv_refusals(corpus, tmp_path, capsys, review, code) -> None:
    sub = submission(tmp_path, {"A003.json": signed("A003")}, review)
    err = refuse(corpus, sub, capsys)
    assert f"refused: review.csv: {code}" in err


def test_two_refusals_name_both_files(corpus, tmp_path, capsys) -> None:
    files = {
        "A001.json": signed("A001", clip_sha256=digest("other")),
        "D001.json": signed("D001", clip_duration_s=6.01),
        "A003.json": signed("A003"),
    }
    err = refuse(corpus, submission(tmp_path, files), capsys)
    assert "refused: labels/A001.json: changed_sha256" in err
    assert "refused: labels/D001.json: changed_duration" in err
    assert "import refused: 2 files, nothing written" in err


def test_refusal_leaves_earlier_submission_untouched(corpus, tmp_path, capsys) -> None:
    import_submission(submission(tmp_path, {"A003.json": signed("A003")}), corpus / "pkg")
    bad = submission(tmp_path, {"D001.json": signed("D001", clip_duration_s=7.0)})
    refuse(corpus, bad, capsys)
    assert sorted(p.name for p in (corpus / "pkg-returns").iterdir()) == ["001"]


# --- refusals of the whole command ------------------------------------------------------------


def test_missing_alias_map(corpus, tmp_path, capsys) -> None:
    (corpus / "pkg" / "operator-private" / "alias-map.json").unlink()
    err = refuse(corpus, submission(tmp_path, {"A003.json": signed("A003")}), capsys)
    assert "alias-map.json: cannot be read" in err


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda doc: doc.update(kind="manifest"), "kind"),
        (lambda doc: doc["records"][0]["original_record"].update(split="test"), "split"),
        (lambda doc: doc["records"][0]["original_record"].update(sha256="ab"), "sha256"),
        (lambda doc: doc["records"][0]["original_record"].update(duration_s=0), "duration_s"),
        (lambda doc: doc["records"][1].update(alias="A001"), "records[1].alias"),
        (lambda doc: doc["records"][0].update(source_corpus="../x"), "source_corpus"),
        (
            lambda doc: doc["records"][1]["original_record"].update(id="ctl-sil-01"),
            "records[1].original_record.id",
        ),
    ],
)
def test_broken_alias_map(corpus, tmp_path, capsys, change, field) -> None:
    path = corpus / "pkg" / "operator-private" / "alias-map.json"
    doc = alias_map()
    change(doc)
    path.write_text(json.dumps(doc))
    err = refuse(corpus, submission(tmp_path, {"A003.json": signed("A003")}), capsys)
    assert field in err


def test_alias_map_not_json(corpus, tmp_path, capsys) -> None:
    (corpus / "pkg" / "operator-private" / "alias-map.json").write_text("{")
    err = refuse(corpus, submission(tmp_path, {"A003.json": signed("A003")}), capsys)
    assert "not a UTF-8 JSON document" in err


def test_missing_or_empty_labels_dir(corpus, tmp_path, capsys) -> None:
    empty = tmp_path / "empty-sub"
    empty.mkdir()
    assert "no labels/ directory" in refuse(corpus, empty, capsys)
    (empty / "labels").mkdir()
    assert "is empty" in refuse(corpus, empty, capsys)


def test_submission_inside_package_is_refused(corpus, tmp_path, capsys) -> None:
    inside = corpus / "pkg" / "01-main-blind"
    (inside / "labels").mkdir()
    (inside / "labels" / "A003.json").write_text(json.dumps(signed("A003")))
    with pytest.raises(LabelImportError, match="inside the package"):
        import_submission(inside, corpus / "pkg")
    assert "inside the package" in refuse(corpus, inside, capsys)


def test_failed_rename_leaves_no_staging(corpus, tmp_path, monkeypatch) -> None:
    def failing_rename(source, target):
        raise OSError("synthetic failure")

    monkeypatch.setattr(label_import.os, "rename", failing_rename)
    with pytest.raises(OSError, match="synthetic failure"):
        import_submission(submission(tmp_path, {"A003.json": signed("A003")}), corpus / "pkg")
    assert list((corpus / "pkg-returns").iterdir()) == []


def test_earlier_submission_without_receipt_is_refused(corpus, tmp_path) -> None:
    (corpus / "pkg-returns" / "001").mkdir(parents=True)
    with pytest.raises(LabelImportError, match="001/import.json"):
        import_submission(submission(tmp_path, {"A003.json": signed("A003")}), corpus / "pkg")


def test_import_never_writes_over_package_templates_pilot_or_earlier_submission(
    corpus: Path, tmp_path: Path, capsys
) -> None:
    blind_labels = corpus / "pkg" / "01-main-blind" / "labels"
    blind_labels.mkdir()
    for alias in ("A001", "A003", "D001"):
        (blind_labels / f"{alias}.json").write_text(json.dumps(template(alias)))
    first = submission(tmp_path, {"A003.json": signed("A003")})
    assert main(["eval", "import-labels", str(first), "--package", str(corpus / "pkg")]) == 0
    protected_before = {
        k: v for k, v in snapshot(corpus).items() if not k.startswith("pkg-returns/002")
    }

    second = submission(tmp_path, {"A001.json": signed("A001"), "D001.json": signed("D001")})
    assert main(["eval", "import-labels", str(second), "--package", str(corpus / "pkg")]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "output: pkg-returns/002"

    after = snapshot(corpus)
    assert {k: v for k, v in after.items() if not k.startswith("pkg-returns/002/")} == (
        protected_before
    )
    assert json.loads((blind_labels / "A001.json").read_text()) == template("A001")
