"""Corpus manifest v1 (issue #26): the schema, the record rules and the minimum volumes."""

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sound_scout import _schema_messages, asr, schema
from sound_scout._schema_messages import schema_errors
from sound_scout.eval import labels, manifest
from sound_scout.eval.manifest import (
    CATEGORIES,
    CONTROL_KINDS,
    MIN_CLIPS,
    MIN_NEGATIVE_CONTROLS,
    ManifestReport,
    format_report,
    main,
    validate,
    validation_report,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_SCHEMA = REPO_ROOT / "schema" / "corpus-manifest.v1.schema.json"
PACKAGED_SCHEMA = REPO_ROOT / "src" / "sound_scout" / "eval" / "corpus-manifest.v1.schema.json"
PROTOCOL = REPO_ROOT / "docs" / "eval" / "PROTOCOL.md"


def sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def record(
    id: str,
    split: str,
    categories: list[str],
    *,
    group: str | None = None,
    negative_control: bool = False,
    control_kind: str = "silence",
    pair_id: str | None = None,
) -> dict:
    doc = {
        "id": id,
        "path": f"clips/{id}.wav",
        "sha256": sha(id),
        "duration_s": 4.5,
        "split": split,
        "source_group": group or f"group-{id}",
        "categories": categories,
        "negative_control": negative_control,
    }
    if negative_control:
        doc["control_kind"] = control_kind
    if pair_id is not None:
        doc["pair_id"] = pair_id
    return doc


def qualifying_manifest() -> dict:
    """Synthetic: the minimum volumes exactly, with one clip carrying two categories."""
    heldout = [
        record("h-pron-1", "heldout", ["pronunciation"]),
        record("h-pron-2", "heldout", ["pronunciation", "repetition_self_correction"]),
        record("h-rep-1", "heldout", ["repetition_self_correction"]),
        record("h-rep-2", "heldout", ["repetition_self_correction"]),
        record("h-pause-1", "heldout", ["pause_phrasing"]),
        record("h-compl-1", "heldout", ["completeness"]),
        record("h-pair-1a", "heldout", ["semantic_stress"], group="pair-1", pair_id="pair-1"),
        record("h-pair-1b", "heldout", ["semantic_stress"], group="pair-1", pair_id="pair-1"),
        record("h-pair-2a", "heldout", ["semantic_stress"], group="pair-2", pair_id="pair-2"),
        record("h-pair-2b", "heldout", ["semantic_stress"], group="pair-2", pair_id="pair-2"),
    ]
    dev = [record(f"d-{i}", "dev", ["pronunciation", "semantic_stress"]) for i in range(5)]
    controls = [
        record(f"nc-{i}", split, [], negative_control=True, control_kind=kind)
        for i, (split, kind) in enumerate(
            [
                ("heldout", "silence"),
                ("heldout", "noise"),
                ("heldout", "corrupted_asr"),
                ("dev", "silence"),
                ("dev", "corrupted_asr"),
            ]
        )
    ]
    records = heldout + dev + controls
    assert len(records) == MIN_CLIPS
    return {"schema_version": "1.0", "name": "synthetic-pilot", "records": records}


@pytest.fixture
def doc() -> dict:
    return qualifying_manifest()


def write(tmp_path: Path, doc: object) -> Path:
    return write_text(tmp_path, json.dumps(doc))


def write_text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(text, encoding="utf-8")
    return path


def assert_rejected(doc: dict, fragment: str) -> ManifestReport:
    report = validation_report(doc)
    assert not report.ok
    assert any(fragment in error for error in report.errors), report.errors
    return report


def test_schema_is_valid_draft_2020_12_and_packaged_copy_is_byte_equal() -> None:
    Draft202012Validator.check_schema(manifest.SCHEMA)
    assert manifest.SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    root_bytes = ROOT_SCHEMA.read_bytes()
    assert root_bytes == PACKAGED_SCHEMA.read_bytes()
    assert root_bytes == manifest.SCHEMA_BYTES
    assert hashlib.sha256(root_bytes).hexdigest() == manifest.SCHEMA_SHA256
    assert manifest.SCHEMA["properties"]["schema_version"] == {"const": manifest.SCHEMA_VERSION}
    assert tuple(manifest.SCHEMA["$defs"]["category"]["enum"]) == CATEGORIES
    record_properties = manifest.SCHEMA["$defs"]["record"]["properties"]
    assert tuple(record_properties["control_kind"]["enum"]) == CONTROL_KINDS
    assert manifest.SCHEMA["$defs"]["record"]["properties"]["split"]["enum"] == list(
        manifest.SPLITS
    )


def test_record_requires_hash_duration_split_group_categories_and_control_flag() -> None:
    required = manifest.SCHEMA["$defs"]["record"]["required"]
    for key in ("sha256", "duration_s", "split", "source_group", "categories", "negative_control"):
        assert key in required, key


def test_valid_manifest_file_qualifies_and_reports_full_coverage(tmp_path: Path, doc: dict) -> None:
    report = validate(write(tmp_path, doc))
    assert report.ok and report.errors == []
    assert report.unrepresented == []
    assert report.heldout_coverage == {
        "pronunciation": 2,
        "repetition_self_correction": 3,
        "semantic_stress": 4,
        "pause_phrasing": 1,
        "completeness": 1,
    }
    assert report == validation_report(doc)


def test_source_group_in_both_splits_is_rejected(doc: dict) -> None:
    doc["records"].append(record("d-variant", "dev", ["pronunciation"], group="group-h-pron-1"))
    report = assert_rejected(doc, "'group-h-pron-1' is present in both dev and heldout")
    assert len(report.errors) == 1


def test_duplicate_sha256_and_duplicate_id_are_rejected(doc: dict) -> None:
    doc["records"][1]["sha256"] = doc["records"][0]["sha256"]
    assert_rejected(doc, "$.records[1].sha256: duplicate of $.records[0]")
    doc = qualifying_manifest()
    doc["records"][2]["id"] = doc["records"][0]["id"]
    assert_rejected(doc, "$.records[2].id: duplicate of $.records[0]")


def test_record_without_category_must_be_a_negative_control(doc: dict) -> None:
    doc["records"][4]["categories"] = []
    assert_rejected(doc, "$.records[4].categories: a record without categories must be a negative")
    doc = qualifying_manifest()
    doc["records"][-1]["categories"] = ["pronunciation"]
    assert_rejected(doc, "$.records[19].categories: a negative control carries no categories")


def test_negative_control_must_name_its_kind(doc: dict) -> None:
    del doc["records"][-1]["control_kind"]
    report = assert_rejected(
        doc,
        "$.records[19].control_kind: a negative control names its kind, one of "
        "silence, noise, corrupted_asr",
    )
    assert len(report.errors) == 1


def test_kind_on_a_record_that_is_not_a_control_is_rejected(doc: dict) -> None:
    doc["records"][0]["control_kind"] = "noise"
    report = assert_rejected(
        doc, "$.records[0].control_kind: only a negative control carries a kind"
    )
    assert len(report.errors) == 1


@pytest.mark.parametrize("kind", CONTROL_KINDS)
def test_every_control_kind_is_accepted(doc: dict, kind: str) -> None:
    doc["records"][-1]["control_kind"] = kind
    assert validation_report(doc).ok


def test_unknown_control_kind_is_a_named_schema_error(doc: dict) -> None:
    doc["records"][-1]["control_kind"] = "music"
    assert_rejected(
        doc, "$.records[19].control_kind: must be one of ['silence', 'noise', 'corrupted_asr']"
    )


def test_category_shortfall_in_heldout_names_the_category_and_the_group(doc: dict) -> None:
    doc["records"][4]["categories"] = ["pronunciation"]  # h-pause-1 loses pause_phrasing
    doc["records"][5]["categories"] = ["pronunciation"]  # h-compl-1 loses completeness
    report = assert_rejected(doc, "category 'pause_phrasing' is not represented in heldout")
    assert_rejected(doc, "category 'completeness' is not represented in heldout")
    assert report.unrepresented == ["pause_phrasing", "completeness"]
    assert report.heldout_coverage["pause_phrasing"] == 0
    assert any("4 prosody events in heldout" in error for error in report.errors), report.errors
    assert not any("diction events" in error for error in report.errors)
    assert "not represented in held-out: pause_phrasing, completeness" in format_report(report)


def test_dev_only_events_do_not_count_towards_heldout_coverage(doc: dict) -> None:
    for item in doc["records"]:
        if item["split"] == "heldout" and not item["negative_control"]:
            item["split"] = "dev"
    report = validation_report(doc)
    assert report.unrepresented == list(CATEGORIES)
    assert report.heldout_coverage == dict.fromkeys(CATEGORIES, 0)
    assert any("0 diction events in heldout" in e for e in report.errors)
    assert any("0 prosody events in heldout" in e for e in report.errors)
    assert any("0 same-words pairs in heldout" in e for e in report.errors)


def test_minimum_clip_and_negative_control_counts(doc: dict) -> None:
    doc["records"] = [r for r in doc["records"] if r["id"] != "d-0"]
    assert_rejected(doc, f"{MIN_CLIPS - 1} clips, the corpus needs at least {MIN_CLIPS}")
    doc = qualifying_manifest()
    doc["records"][-1]["negative_control"] = False
    doc["records"][-1]["categories"] = ["completeness"]
    del doc["records"][-1]["control_kind"]
    assert_rejected(
        doc,
        f"{MIN_NEGATIVE_CONTROLS - 1} negative controls, the corpus needs at least "
        f"{MIN_NEGATIVE_CONTROLS}",
    )


def test_same_words_pairs_are_complete_in_one_group_and_counted_in_heldout(doc: dict) -> None:
    doc["records"][7]["pair_id"] = None
    report = assert_rejected(doc, "pair 'pair-1' has a single record")
    assert any("1 same-words pairs in heldout" in e for e in report.errors)
    doc = qualifying_manifest()
    doc["records"][7]["source_group"] = "elsewhere"
    assert_rejected(doc, "pair 'pair-1' spans several source groups")
    doc = qualifying_manifest()
    doc["records"][0]["pair_id"] = "pair-3"
    doc["records"][2]["pair_id"] = "pair-3"
    doc["records"][2]["source_group"] = doc["records"][0]["source_group"]
    assert_rejected(doc, "$.records[0].pair_id: a member of a same-words pair carries")


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda r: r.update(split="test"), "$.records[0].split"),
        (lambda r: r.update(sha256="abc"), "$.records[0].sha256"),
        (lambda r: r.update(duration_s=0), "$.records[0].duration_s"),
        (lambda r: r.update(categories=["emotion"]), "$.records[0].categories[0]"),
        (lambda r: r.update(categories=["completeness", "completeness"]), "non-unique"),
        (lambda r: r.pop("negative_control"), "'negative_control' is a required property"),
        (lambda r: r.update(extra=1), "$.records[0]: Additional properties"),
        (lambda r: r.update(path="/abs/clip.wav"), "must be relative to the manifest directory"),
        (lambda r: r.update(path="../clip.wav"), "must not contain '..' segments"),
    ],
)
def test_record_shape_and_path_rules(doc: dict, mutate, fragment: str) -> None:
    mutate(doc["records"][0])
    assert_rejected(doc, fragment)


def test_schema_errors_stop_before_record_rules_and_coverage_is_zero(doc: dict) -> None:
    doc["records"][0]["split"] = "test"
    report = validation_report(doc)
    assert all(error.startswith("$.records[0].split") for error in report.errors)
    assert report.heldout_coverage == dict.fromkeys(CATEGORIES, 0)


def test_unreadable_json_is_an_error_not_an_exception(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    report = validate(path)
    assert not report.ok
    assert report.errors[0].startswith("$: not a UTF-8 JSON document")
    assert report.heldout_coverage == dict.fromkeys(CATEGORIES, 0)


def test_missing_or_unreadable_manifest_file_is_an_error_not_an_exception(tmp_path: Path) -> None:
    report = validate(tmp_path / "missing.json")
    assert not report.ok
    assert report.errors[0].startswith("$: cannot read manifest:")
    assert report.heldout_coverage == dict.fromkeys(CATEGORIES, 0)
    assert validate(tmp_path).errors[0].startswith("$: cannot read manifest:")


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_non_finite_number_under_an_undeclared_key_names_its_parent(
    doc: dict, value: float
) -> None:
    marker = "MARKER-a1b2c3"
    doc[marker] = [{"x": value}]
    errors = validation_report(doc).errors
    assert "$: NaN/Infinity under an undeclared key is not allowed" in errors, errors
    assert not any(marker in error for error in errors), errors


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_duration_is_rejected_before_the_schema(
    tmp_path: Path, doc: dict, raw: str
) -> None:
    text = json.dumps(doc).replace('"duration_s": 4.5', f'"duration_s": {raw}', 1)
    report = validate(write_text(tmp_path, text))
    assert not report.ok
    assert report.errors[0] == "$.records[0].duration_s: NaN/Infinity is not allowed, use null"
    assert report.heldout_coverage == dict.fromkeys(CATEGORIES, 0)


def test_main_prints_the_report_and_exits_non_zero_on_shortfall(tmp_path: Path, capsys) -> None:
    doc = qualifying_manifest()
    assert main([str(write(tmp_path, doc))]) == 0
    assert "manifest qualifies" in capsys.readouterr().out
    doc["records"][5]["categories"] = ["pause_phrasing"]  # h-compl-1 loses completeness
    assert main([str(write(tmp_path, doc))]) == 1
    out = capsys.readouterr().out
    assert "not represented in held-out: completeness" in out
    assert "manifest does not qualify" in out
    assert main([]) == 2


def test_protocol_states_the_volumes_the_pilot_caveat_and_the_proposal_status() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for fragment in (
        "precision",
        "0.80",
        "recall",
        "0.70",
        "малый пилот",
        "не статистическое доказательство",
        "предложение владельцу",
        "abstention",
        "отрицательн",
        f"{MIN_CLIPS} ",
        "dev",
        "held-out",
    ):
        assert fragment in text, fragment
    for name in CATEGORIES:
        assert f"`{name}`" in text, name


def test_record_missing_categories_key_is_rejected(doc: dict) -> None:
    del doc["records"][0]["categories"]
    assert_rejected(doc, "'categories' is a required property")


def test_protocol_covers_categories_matching_volumes_split_rule_abstention_and_controls() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for heading in (
        "## 2. Категории событий",
        "## 4. Минимальные объёмы",
        "## 5. Разбиение dev / held-out",
        "## 6. Сопоставление finding и размеченного события",
        "## 7. Метрики и пороги",
        "## 8. Abstention",
        "## 9. Отрицательные контроли и «уверенная выдумка»",
    ):
        assert heading in text, heading
    for fragment in (
        "дикция",
        "просодия",
        "Варианты одного источника не разносятся между dev и held-out",
        "Допустимый выход finding за границы события",
        "**согласоваться с наблюдением**",
        "false positive, а событие остаётся пропуском (false negative)",
        "пересечение ≥ 50 %",
        "| отдельных клипов | 20 |",
        "| отрицательных контролей | 5 |",
        "| 5 | held-out |",
        "| пар «одинаковые слова, разный акцент» | 2 | held-out |",
        "0 — категория не представлена, квалификация не проходит",
        "**Уверенная выдумка**",
        "Порог — **ноль**",
        "T06",
    ):
        assert fragment in text, fragment


MANIFEST_TEXT = "секрет-манифеста"


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (lambda d: d.update(name=MANIFEST_TEXT), "$.name"),
        (lambda d: d.update(schema_version=MANIFEST_TEXT), "$.schema_version"),
        (lambda d: d.update(note=[MANIFEST_TEXT]), "$.note"),
        (lambda d: d.update({MANIFEST_TEXT: "x"}), "$: Additional properties are not allowed"),
        (lambda d: d.update(records={"note": MANIFEST_TEXT}), "$.records"),
        (lambda d: d["records"].append(MANIFEST_TEXT), "$.records[20]"),
        (lambda d: d["records"][0].update(id=MANIFEST_TEXT), "$.records[0].id"),
        (lambda d: d["records"][0].update(source_group=MANIFEST_TEXT), "$.records[0].source_group"),
        (lambda d: d["records"][0].update(pair_id=MANIFEST_TEXT), "$.records[0].pair_id"),
        (lambda d: d["records"][0].update(split=MANIFEST_TEXT), "$.records[0].split"),
        (lambda d: d["records"][0].update(note={"text": MANIFEST_TEXT}), "$.records[0].note"),
        (lambda d: d["records"][0].update(path=[MANIFEST_TEXT]), "$.records[0].path"),
        (lambda d: d["records"][0].update(duration_s=MANIFEST_TEXT), "$.records[0].duration_s"),
        (
            lambda d: d["records"][0].update(categories=[MANIFEST_TEXT, MANIFEST_TEXT]),
            "$.records[0].categories",
        ),
    ],
)
def test_schema_errors_name_the_rule_without_echoing_manifest_values(
    tmp_path: Path, capsys, doc: dict, mutate, path: str
) -> None:
    mutate(doc)
    report = assert_rejected(doc, path)
    assert not any(MANIFEST_TEXT in error for error in report.errors), report.errors
    assert main([str(write(tmp_path, doc))]) == 1
    out = capsys.readouterr().out
    assert path in out
    assert MANIFEST_TEXT not in out


IDENTIFIER_PATTERN = "'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'"


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda d: d["records"][0].update(id="-bad"),
            f"$.records[0].id: must match pattern {IDENTIFIER_PATTERN}",
        ),
        (
            lambda d: d["records"][0].update(split="test"),
            "$.records[0].split: must be one of ['dev', 'heldout']",
        ),
        (
            lambda d: d["records"][0].update(negative_control="no"),
            "$.records[0].negative_control: must be of type boolean",
        ),
        (
            lambda d: d.update(note=1),
            "$.note: must be of type ['string', 'null']",
        ),
        (
            lambda d: d["records"][0].pop("sha256"),
            "$.records[0]: 'sha256' is a required property",
        ),
        (
            lambda d: d["records"][0].update(extra=1),
            "$.records[0]: Additional properties are not allowed, allowed: ['categories',",
        ),
        (lambda d: d.update(schema_version="2.0"), "$.schema_version: '1.0' was expected"),
        (
            lambda d: d["records"][0].update(duration_s=0),
            "$.records[0].duration_s: must be greater than 0",
        ),
        (
            lambda d: d["records"][0].update(pair_id="-bad"),
            "$.records[0].pair_id: must be null or a valid identifier",
        ),
        (
            lambda d: d["records"][0].update(categories=["completeness", "completeness"]),
            "$.records[0].categories: has non-unique items, every item must be unique",
        ),
        (
            lambda d: d["records"][0].update(path=""),
            "$.records[0].path: must be at least 1 characters long",
        ),
    ],
)
def test_each_manifest_schema_rule_is_named_with_its_path(
    tmp_path: Path, doc: dict, mutate, error: str
) -> None:
    mutate(doc)
    report = validate(write(tmp_path, doc))
    assert any(e.startswith(error) for e in report.errors), report.errors
    assert not any("violates schema rule" in e for e in report.errors), report.errors


def test_result_corrections_manifest_and_labels_build_schema_messages_in_one_place() -> None:
    for module in (schema, asr, manifest, labels):
        assert module.schema_errors is _schema_messages.schema_errors, module
        assert not hasattr(module, "_schema_message"), module
        assert not hasattr(module, "_json_path"), module
        assert not hasattr(module, "Draft202012Validator"), module


def test_rule_without_its_own_message_is_named_without_the_value() -> None:
    errors = schema_errors({"properties": {"name": {"maxItems": 1}}}, {"name": [MANIFEST_TEXT] * 2})
    assert errors == ["$.name: violates schema rule 'maxItems'"]


# --- draft mode (#78): label-dependent shortfalls apart from structural errors -------------


def draft_manifest() -> dict:
    """Sound structure, enough clips and controls, but held-out coverage only for prosody."""
    doc = qualifying_manifest()
    for item in doc["records"]:
        if item["id"].startswith(("h-pron", "h-rep")):
            item["categories"] = ["pause_phrasing"]
        item.pop("pair_id", None)
    return doc


def test_label_dependent_errors_are_a_subset_of_errors(doc: dict) -> None:
    report = validation_report(draft_manifest())
    assert not report.ok and report.draft_ok
    assert report.label_dependent and set(report.label_dependent) <= set(report.errors)
    assert any("'pronunciation' is not represented" in e for e in report.label_dependent)
    assert any("same-words pairs" in e for e in report.label_dependent)
    assert validation_report(doc).label_dependent == []


def test_structural_and_count_errors_are_not_label_dependent() -> None:
    doc = draft_manifest()
    doc["records"][0]["source_group"] = doc["records"][10]["source_group"]
    doc["records"] = doc["records"][:-1]
    report = validation_report(doc)
    assert not report.draft_ok
    structural = [e for e in report.errors if e not in report.label_dependent]
    assert any("present in both dev and heldout" in e for e in structural)
    assert any("clips, the corpus needs" in e for e in structural)


def test_draft_report_puts_label_dependent_errors_in_their_own_block() -> None:
    report = validation_report(draft_manifest())
    text = format_report(report, draft=True)
    head, block = text.split(manifest.DRAFT_WAITING + ":\n")
    assert "same-words pairs" not in head and "same-words pairs" in block
    assert text.splitlines()[-1] == manifest.DRAFT_SOUND
    assert "same-words pairs" in format_report(report)
