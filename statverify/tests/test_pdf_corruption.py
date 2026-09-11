"""Регрессия на две дыры, найденные фаззингом PDF-порчи.

scripts/run_pdf_corruption_fuzz.py прогоняет фаззинг по всем семи видам
порчи и печатает таблицу для отчёта — это разведка, не регрессия. Здесь
— ровно те два вида, что оказались опасными, закреплены как постоянный
тест: если починка тихо развалится, тест упадёт, а не останется
незамеченным в одноразовом скрипте.

Единственный опасный исход — тот же, что и в скрипте: конвейер извлёк
НЕВЕРНОЕ значение и охранник его пропустил. Не найти утверждение или
честно отказать — безопасно.

ИСТОРИЯ ДЫРЫ 2 (склейка колонок), ЧЕСТНО, ПОТОМУ ЧТО ПЕРВАЯ ПОЧИНКА НЕ
СРАБОТАЛА. Первая версия column_splicing (и первая версия починки —
обрубание окна поиска p на следующем t(...)) не воспроизводила
реальный артефакт: не вставляла перенос строки вовсе и вклинившийся
фрагмент всегда содержал чужой t(...). Отчёт показывал 86% -> 0%, но
проверка на трёх РЕАЛИСТИЧНЫХ видах вклинивания (ни один не содержит
t(, все три приходят со своим переносом строки — так, как вытекание
колонки происходит на самом деле) нашла: дыра оставалась открыта на
100% и до, и после той "починки". Настоящая причина была ДВОЙНОЙ:
extract._claim_from_loose продвигал единственного кандидата без
проверки физического разрыва строки между t и кандидатом, а TIGHT ещё
и умел молча ПОГЛОЩАТЬ этот перенос через [\\s,;]* (\\s включает '\\n')
до того, как добраться до собственной защиты _GAP — то есть склейка
без точек внутри вклинившегося текста проходила прямо через TIGHT,
минуя _claim_from_loose целиком. Обе причины закрыты раздельно: TIGHT
получил разделитель _SEP без \\n, _claim_from_loose продвигает
единственного кандидата только при отсутствии переноса строки между
t-образцом и кандидатом.
"""

import pytest

from corpus_real import CORPUS_REAL
from pdf_corrupt import (CORRUPTIONS, COLUMN_SPLICE_KINDS, column_splicing,
                          ground_truth, resolve_after_corruption)

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


@pytest.mark.parametrize("kind", sorted(COLUMN_SPLICE_KINDS))
@pytest.mark.parametrize("rec", RECORDS, ids=lambda r: r["id"])
def test_column_splicing_never_passes_guard_wrong(rec, kind):
    """ДЫРА 2 (TIGHT._SEP + newline-gate в extract._claim_from_loose):
    вклинившийся фрагмент соседней колонки/таблицы/подписи — реалистичный
    (со своим переносом строки, БЕЗ собственного t(...)) — обязан либо
    не примениться, либо дать честный отказ, никогда не подменить p
    исходного утверждения чужим значением."""
    gt = ground_truth(rec["text"])
    assert gt is not None
    corrupted = column_splicing(rec["text"], gt, kind)
    outcome = _outcome(rec, corrupted)
    assert outcome != "wrong_admitted", (
        f"{rec['id']}/{kind}: p из вклинившейся колонки подменил исходное значение")
