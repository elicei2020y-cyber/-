"""Регрессия на две дыры, найденные фаззингом PDF-порчи (найдены и
починены после review): линейный разрыв внутри числа и склейка колонок.

scripts/run_pdf_corruption_fuzz.py прогоняет фаззинг по всем семи видам
порчи и печатает таблицу для отчёта — это разведка, не регрессия.
Здесь — ровно те два вида, что оказались опасными (доля 'неверное
значение прошло охранника' была 93% и 86% до починки), закреплены как
постоянный тест: если починка (C5 в guard.py, независимая от найденного
кандидата граница в extract._claim_from_loose) тихо развалится, тест
упадёт, а не останется незамеченным в одноразовом скрипте.

Единственный опасный исход — тот же, что и в скрипте: конвейер извлёк
НЕВЕРНОЕ значение и охранник его пропустил. Не найти утверждение или
честно отказать — безопасно.
"""

import pytest

from corpus_real import CORPUS_REAL
from pdf_corrupt import CORRUPTIONS, column_splicing, ground_truth, resolve_after_corruption

RECORDS = [r for r in CORPUS_REAL if r.get("has_t_test")]


def _outcome(rec, corrupted):
    if corrupted is None:
        return "n/a"
    claim, admitted, values = resolve_after_corruption(corrupted, rec["t"], rec["df"])
    if values is None:
        return "not_found"
    if not admitted:
        return "rejected"
    truth = {"t": rec["t"], "df": rec["df"], "p": rec["p"]}
    wrong = any(abs(values[n] - truth[n]) > 1e-9 for n in ("t", "df", "p"))
    return "wrong_admitted" if wrong else "correct_admitted"


@pytest.mark.parametrize("rec", RECORDS, ids=lambda r: r["id"])
def test_linebreak_in_number_never_passes_guard_wrong(rec):
    """ДЫРА 1 (C5, guard.py): разрыв строки внутри числа обязан либо не
    примениться (нет цифр после точки), либо дать честный отказ —
    никогда не должен пройти охранника с усечённым значением."""
    gt = ground_truth(rec["text"])
    assert gt is not None
    corrupted = CORRUPTIONS["linebreak_in_number"](rec["text"], gt)
    outcome = _outcome(rec, corrupted)
    assert outcome != "wrong_admitted", (
        f"{rec['id']}: усечённое переносом значение прошло охранника")


@pytest.mark.parametrize("rec", RECORDS, ids=lambda r: r["id"])
def test_column_splicing_never_passes_guard_wrong(rec):
    """ДЫРА 2 (независимая от кандидата граница, extract._claim_from_loose):
    вклинившийся фрагмент соседней колонки (с собственным t(...) внутри)
    обязан либо не примениться, либо дать честный отказ — никогда не
    должен подменить p исходного утверждения чужим значением."""
    idx = RECORDS.index(rec)
    foreign = RECORDS[(idx + 1) % len(RECORDS)]["text"]
    gt = ground_truth(rec["text"])
    assert gt is not None
    corrupted = column_splicing(rec["text"], gt, foreign)
    outcome = _outcome(rec, corrupted)
    assert outcome != "wrong_admitted", (
        f"{rec['id']}: p из вклинившейся колонки подменил исходное значение")
