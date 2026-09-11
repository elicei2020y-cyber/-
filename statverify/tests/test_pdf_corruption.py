"""Регрессия на две дыры, найденные фаззингом PDF-порчи.

scripts/run_pdf_corruption_fuzz.py прогоняет фаззинг по всем семи видам
порчи и печатает таблицу для отчёта — это разведка, не регрессия. Здесь
— ровно те два вида, что оказались опасными, закреплены как постоянный
тест: если починка тихо развалится, тест упадёт, а не останется
незамеченным в одноразовом скрипте.

Единственный опасный исход — тот же, что и в скрипте: конвейер извлёк
НЕВЕРНОЕ значение и охранник его пропустил. Не найти утверждение или
честно отказать — безопасно.

ИСТОРИЯ ДЫРЫ 2 (склейка колонок), ЧЕСТНО, В ТРИ РАУНДА — ДВА ПЕРВЫХ НЕ
СРАБОТАЛИ, КАЖДЫЙ БЫЛ ПОПРАВЛЕН СЛЕДУЮЩИМ РЕВЬЮ.

Раунд 1 (обрубание окна поиска p на следующем t(...)): ловил только ту
форму склейки, где вклинившийся фрагмент сам содержит t(...) —
реалистичные виды (строка таблицы, обрывок предложения, подпись к
рисунку) t( не содержат, дыра оставалась на 100%. Прежняя (первая)
версия column_splicing тоже была нереалистична — не вставляла перенос
строки вовсе, поэтому не могла ни найти уязвимость, ни подтвердить
починку.

Раунд 2 (запрет продвижения единственного кандидата при ЛЮБОМ переносе
строки между t-образцом и им): ловил все реалистичные виды склейки
(0%), но ценой, которую сам раунд не измерил — обычный перенос строки
ВНУТРИ ОДНОГО предложения (в PDF происходит постоянно) неотличим по
этому признаку от вытекания колонки. corpus_real.py (получен из HTML)
физически не может об этом свидетельствовать — переносов строк в нём
нет вовсе, поэтому "потерь на корпусе нет" не проверяло то, что нужно.
Настоящая потеря была измерена на родной ловушке tests/corpus.py
(id='line_break'): раньше извлекалась верно, после раунда 2 стала
AMBIGUOUS_PARSE.

Раунд 3 (действующий): различает случаи ЧИСЛОМ кандидатов, а не
наличием переноса. Обычное окно (CANDIDATE_WINDOW) остаётся первым
приближением всегда; если оно дало РОВНО одного кандидата, отделённого
переносом строки, поиск повторяется в окне WIDE_CANDIDATE_WINDOW.
При вытекании колонки настоящий p остаётся в тексте ДАЛЬШЕ
вклинившегося фрагмента — расширенный поиск находит его вторым
кандидатом, и AMBIGUOUS_PARSE возникает от числа найденных значений
(двух), а не от специального признака. При обычном переносе внутри
предложения второго кандидата там нет, находится всё тот же один —
line_break вернулся к исходному ожиданию (успешное извлечение).
WIDE_CANDIDATE_WINDOW подобран измерением двух кривых
(scripts/run_window_w_sweep.py), не назначен — и там же честно
сказано, что обе кривые синтетические: TIGHT ещё и умел молча
ПОГЛОЩАТЬ перенос строки через [\\s,;]* (\\s включает '\\n') до того,
как добраться до защиты самого _GAP — починено отдельно, разделителем
_SEP без \\n.
"""

import pytest

from corpus_real import CORPUS_REAL
from pdf_corrupt import (CORRUPTIONS, COLUMN_SPLICE_KINDS, column_splicing,
                          ground_truth, legit_linebreak, resolve_after_corruption)

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


@pytest.mark.parametrize("rec", RECORDS, ids=lambda r: r["id"])
def test_legit_linebreak_still_resolves_correctly(rec):
    """ЦЕНА ПОЧИНКИ ДЫРЫ 2, ЗАКРЕПЛЁННАЯ КАК ПОСТОЯННЫЙ ТЕСТ ПОЛНОТЫ, А
    НЕ ТОЛЬКО КАК ОТСУТСТВИЕ ОПАСНОСТИ. Обычный перенос строки внутри
    ОДНОГО предложения (без второго, вклинившегося p дальше по тексту)
    обязан по-прежнему извлекаться верно — раунд 2 починки дыры 2 (запрет
    продвижения при любом переносе) это ломал, раунд 3 (расширение окна,
    см. tests/pdf_corrupt.py::legit_linebreak) чинит. Если это перестанет
    выполняться на всех 43 записях corpus_real.py, а не только на
    единственном примере из tests/corpus.py::line_break, — здесь это
    будет видно."""
    gt = ground_truth(rec["text"])
    assert gt is not None
    corrupted = legit_linebreak(rec["text"], gt)
    if corrupted is None:
        pytest.skip(f"{rec['id']}: нет маркера p с пробелом перед ним")
    outcome = _outcome(rec, corrupted)
    assert outcome == "correct_admitted", (
        f"{rec['id']}: легитимный перенос строки внутри предложения "
        f"потерял извлечение (исход {outcome!r}), хотя второго кандидата нет")
