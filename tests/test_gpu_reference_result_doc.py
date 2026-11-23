"""The #149 rental result in 0028, 0023, 0024, gpu-timing.md and CLAUDE.md is consistent."""

import re
from pathlib import Path

import pytest

from sound_scout.eval import gate0, gpu_ext, vast

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs/decisions/0028-gpu-reference-gate0.md"
RESULT_HEADING = "\n## Результат (2026-11-23)\n"
ATTEMPT = "20261123T054739Z-0ae97bef"
ITEMS = [f"{variant}({clip})" for clip in gate0.CONTROLS for variant in ("K1", "K2", "P", "A")]
FAILED_GPU = {
    "qwen25omni": ["P(S0)", "A(S0)", "A(N1)", "A(T1)"],
    "qwen3omni": ["P(T1)"],
}
FAILED_CPU = {
    "qwen25omni": ["P(S0)", "A(S0)", "P(N1)", "A(N1)", "A(T1)"],
    "qwen3omni": ["P(S0)", "P(T1)"],
}
LOCAL_DOC = {
    "qwen25omni": REPO / "docs/decisions/0024-gate0-qwen25-omni.md",
    "qwen3omni": REPO / "docs/decisions/0023-gate0-qwen3-omni.md",
}
TABLE_TITLE = {
    "qwen25omni": "**Qwen2.5-Omni-7B, `GATE0-RULE v1`.**",
    "qwen3omni": "**Qwen3-Omni-30B-A3B-Instruct, `GATE0-RULE v2`.**",
}


def text() -> str:
    return DOC.read_text(encoding="utf-8")


def result() -> str:
    return text().split(RESULT_HEADING, 1)[1]


def reason(failed: list[str]) -> str:
    """The reason line `gate0.evaluate` writes for a C failure."""
    return f"C: не прошли {len(failed)} из 12 пунктов контролей: {failed}"


def rows(section: str) -> dict[str, list[str]]:
    found = {}
    for line in section.splitlines():
        if line.startswith("| ") and not line.startswith("|---"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            found[cells[0]] = cells[1:]
    return found


def comparison(model: str) -> dict[str, list[str]]:
    section = result().split("### Р.4.", 1)[1].split("### Р.5.", 1)[0]
    table = section.split(TABLE_TITLE[model], 1)[1].split("Вне вердикта:", 1)[0]
    return rows(table)


def number(cell: str) -> float:
    return float(cell.replace(",", "."))


def test_the_result_is_one_dated_section_after_the_whole_preregistration() -> None:
    document = text()
    assert document.count(RESULT_HEADING) == 1
    before = document.split(RESULT_HEADING, 1)[0]
    headings = [line for line in before.splitlines() if line.startswith("#")]
    assert headings[-1] == "### Д.10. Запуск"
    assert "Статус: предрегистрация. Записана на `main` до аренды. Дата: 2026-11-22." in before
    assert "\n## " not in result()
    assert ATTEMPT in result() and "`62003019`" in result() and "`62104153`" in result()


@pytest.mark.parametrize("model", ["qwen25omni", "qwen3omni"])
def test_each_gpu_verdict_is_the_reason_line_gate0_writes(model: str) -> None:
    assert f"`{reason(FAILED_GPU[model])}`" in result()
    assert reason(FAILED_CPU[model]) in LOCAL_DOC[model].read_text(encoding="utf-8")


def mismatches(table: dict[str, list[str]], model: str) -> list[str]:
    """Items whose marks or "same" cell disagree with the CPU and GPU verdict reasons."""
    wrong = []
    for key in ITEMS:
        cpu, gpu, same = table[key]
        if (
            cpu[0] not in "✓✗"
            or gpu[0] not in "✓✗"
            or (key in FAILED_CPU[model]) != (cpu[0] == "✗")
            or (key in FAILED_GPU[model]) != (gpu[0] == "✗")
            or same != ("да" if cpu[0] == gpu[0] else "нет")
        ):
            wrong.append(key)
    return wrong


@pytest.mark.parametrize("model", ["qwen25omni", "qwen3omni"])
def test_the_per_criterion_table_covers_all_twelve_items_and_marks_the_verdict_failures(
    model: str,
) -> None:
    table = comparison(model)
    assert [key for key in table if key in ITEMS] == ITEMS
    assert mismatches(table, model) == []
    cpu_passed, gpu_passed, _ = table["C, прошло пунктов"]
    assert cpu_passed == f"{12 - len(FAILED_CPU[model])} из 12"
    assert gpu_passed == f"{12 - len(FAILED_GPU[model])} из 12"
    assert table["Вердикт"] == ["NOT READY", "NOT READY", "да"]
    assert [table[key][2] for key in ("P1", "P2", "P3")] == ["да"] * 3


def test_a_flipped_mark_in_the_table_is_caught() -> None:
    table = comparison("qwen3omni")
    cpu, gpu, same = table["P(T1)"]
    table["P(T1)"] = [cpu, "✓" + gpu[1:], same]
    assert mismatches(table, "qwen3omni") == ["P(T1)"]


def test_the_extended_conclusions_follow_the_preregistered_rules_from_the_table() -> None:
    section = result().split("### Р.5.", 1)[1].split("### Р.6.", 1)[0]
    table = rows(section)
    for column in (0, 1):
        utterances = int(table["R: реплик с обоими P по аудио"][column].split()[0])
        r1 = re.search(r"\[([\d,]+); ([\d,]+)\]", table["R1, balanced accuracy [95 % CI]"][column])
        mcnemar = re.search(
            r"b = (\d+), c = (\d+), p = ([\d,]+)", table["R2, McNemar аудио против текста"][column]
        )
        b, c, p = int(mcnemar[1]), int(mcnemar[2]), number(mcnemar[3])
        assert round(gpu_ext.mcnemar_exact(b, c), 3) == p
        shown = (
            utterances >= gpu_ext.MIN_UTTERANCES
            and number(r1[1]) > 0.5
            and p < gpu_ext.ALPHA
            and b > c
        )
        assert table["**Вывод R**"][column] == ("**показано**" if shown else "**не показано**")
        s1 = re.search(
            r"(\d+) из (\d+).*p = ([\d,]+)", table["S1, пар «верно → да, дефект → нет»"][column]
        )
        right, pairs, p_s1 = int(s1[1]), int(s1[2]), number(s1[3])
        assert round(gpu_ext.binomial_greater(right, pairs, gpu_ext.CHANCE_PAIR), 3) == p_s1
        above = pairs >= gpu_ext.MIN_PAIRS and p_s1 < gpu_ext.ALPHA
        assert table["**Вывод S1**"][column] == (
            "**выше случайного**" if above else "**не выше случайного**"
        )


def test_the_cache_record_names_the_flags_the_stack_starts_vllm_with() -> None:
    section = result().split("### Р.6.", 1)[1].split("### Р.7.", 1)[0]
    argv = vast.vllm_argv("/workspace/models/qwen25omni", "qwen25omni")
    assert "--no-enable-prefix-caching" in argv
    assert argv[argv.index("--mm-processor-cache-gb") + 1] == "0"
    assert "`'enable_prefix_caching': False`" in section
    assert "`'mm_processor_cache_gb': 0.0`" in section
    assert "Полей `Prefix cache hit rate` и `MM cache hit rate` нет ни в одной строке" in section
    assert "#161" in section


@pytest.mark.parametrize("model", ["qwen25omni", "qwen3omni"])
def test_the_local_decisions_carry_a_dated_gpu_note_after_their_sections(model: str) -> None:
    document = LOCAL_DOC[model].read_text(encoding="utf-8")
    note = document.split("\n## 10. Эталон GPU bf16 (2026-11-23)\n", 1)[1]
    assert document.index("\n## 9. Происхождение\n") < document.index("\n## 10. ")
    assert reason(FAILED_GPU[model]) in note and "`0028`" in note
    assert f"Прошли {12 - len(FAILED_GPU[model])} из 12" in note


def test_the_qwen3_note_does_not_claim_no_truncated_answer() -> None:
    note = LOCAL_DOC["qwen3omni"].read_text(encoding="utf-8").split("\n## 10. ", 1)[1]
    assert "`finish_reason: length`" in note and "обрезанный — один, P(`T1`)" in note


def test_claude_md_states_both_gpu_results_next_to_the_local_ones() -> None:
    claude = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    stack = claude.split("- **Стек:**", 1)[1].split("\n- **", 1)[0]
    flat = " ".join(stack.split())
    assert (
        "эталон в bf16 на vLLM (#149, `0028`) — тоже NOT READY, не прошёл 1 из 12: P(`T1`)" in flat
    )
    assert "эталон в bf16 на vLLM (#149, `0028`) — тоже NOT READY, не прошли 4 из 12" in flat
    assert "прошли 7 из 12" in flat


def test_the_timing_document_records_every_step_of_the_149_rental() -> None:
    timing = (REPO / "docs/gpu-timing.md").read_text(encoding="utf-8")
    section = timing.split("## Эталонный прогон #149 (2026-11-23)", 1)[1].split("\n## ", 1)[0]
    measured = {}
    for line in section.splitlines():
        if line.startswith("| `"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            measured[cells[0]] = cells[2]
    assert measured == {
        "`arm`": "0,04",
        "`create`": "2,3",
        "`ready`": "294,5",
        "`preflight`": "30,5",
        "`weights` M1": "196,3",
        "`smoke` M1": "47,2",
        "`run` M1": "85,5",
        "`extended` M1": "1234,6",
        "`weights` M2": "42,1",
        "`smoke` M2": "50,8",
        "`run` M2": "260,2",
        "`extended` M2": "1603,9",
        "`download`": "5,9",
        "`destroy`": "3,7",
    }
    assert ATTEMPT in section and "3877,3 с (64,6 мин)" in section
    assert "3877,3" in result() and "294,5" in result()


def test_the_new_records_hold_no_personal_paths_and_link_existing_decisions() -> None:
    for written in (result(), (REPO / "docs/gpu-timing.md").read_text(encoding="utf-8")):
        assert not re.search(r"/(home|Users)/|[A-Za-z]:\\", written)
    for name in set(re.findall(r"`(00\d\d)`", result())):
        assert len(list((REPO / "docs/decisions").glob(f"{name}-*.md"))) == 1, name


def test_the_money_record_states_the_section_9_bound_is_exceeded_by_the_seen_delta() -> None:
    section = result().split("### Р.2.", 1)[1].split("### Р.3.", 1)[0]
    flat = " ".join(section.split())
    assert "5. Кредит: сразу после `closed` и через 24 ч. Дельта ≤ 1,20 USD." in text()
    assert "**Граница § 9 п. 5 превышена.**" in flat
    assert "`delta_usd` 1,2118" in flat and "(1,2228)" in flat
    assert "не заменяет границу § 9 п. 5" in flat
    assert "0 инстансов. Это выполняет § 9 п. 4." in flat


def test_the_qwen3_records_cite_the_correction_of_the_rental_summary() -> None:
    correction = "issuecomment-7094549589"
    assert correction in result()
    note = LOCAL_DOC["qwen3omni"].read_text(encoding="utf-8").split("\n## 10. ", 1)[1]
    assert correction in note and "`тон` на CPU, `тишина` на GPU" in note
    assert "«Совпадает» — совпал итог пункта (✓ / ✗), не текст ответа." in result()
