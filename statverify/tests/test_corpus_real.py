"""Конвенция нижнего порога на реальных утверждениях из tests/corpus_real.py.

Каждый тест здесь закреплён за конкретным утверждением из настоящей
статьи (id, DOI, дословная цитата в докстроке), а не за абстрактным
случаем — так же, как остальной проект требует прослеживать числа до
прогона, который их породил (CLAUDE.md).

Про происхождение корпуса и то, что в этой сессии не перепроверено
(сетевой доступ к journals.plos.org заблокирован политикой организации),
см. docs/CORPUS_REAL_NOTES.md.
"""

import pytest

from statverify.extract import extract
from statverify.verify import verify_claim

from corpus_real import CORPUS_REAL

_BY_ID = {r["id"]: r for r in CORPUS_REAL}


def _claim_for(cid):
    r = _BY_ID[cid]
    doc = extract(r["text"])
    claim = next(c for c in doc.claims if {"t", "df", "p"} <= set(c.slots))
    return r, claim


# --- Пять случаев печати нижнего порога -----------------------------------
#
# SPSS и часть журналов печатают 'p = .001' (или другой круглый порог)
# для всего, что меньше — это способ печати, не ошибка авторов. Все пять
# — из 10.1371/journal.pone.0267297 и 10.1371/journal.pone.0058546,
# https://journals.plos.org/plosone/article?id=<DOI>.

def test_floor_A03():
    """DOI 10.1371/journal.pone.0267297. Цитата: 'children: M = .40,
    SD = .17, t(27) = 6.17, p = .001, α = .05, d = 1.16'.
    Вычислено: p_two_tailed(6.17, 27) ≈ 1.35e-06, на порядки меньше
    напечатанного .001."""
    r, claim = _claim_for("A03")
    v = verify_claim(claim)
    assert v.status.startswith("INCONSISTENT")
    assert v.status != "CONSISTENT"
    assert v.floor_convention is True
    assert v.floor_threshold is not None and abs(v.floor_threshold - 0.001) < 1e-12


def test_floor_A05():
    """DOI 10.1371/journal.pone.0267297. Цитата: 'children: t(30) = 7.07,
    p = .001, α = .016, d = 1.27'. Вычислено: p_two_tailed(7.07, 30) ≈ 7e-08."""
    r, claim = _claim_for("A05")
    v = verify_claim(claim)
    assert v.status != "CONSISTENT"
    assert v.floor_convention is True
    assert v.floor_threshold is not None and abs(v.floor_threshold - 0.001) < 1e-12


def test_floor_A06():
    """DOI 10.1371/journal.pone.0267297. Цитата: 'adults: t(32) = 7.83,
    p = .001, α = .016, d = 1.36'. Вычислено: p_two_tailed(7.83, 32) ≈ 1e-08."""
    r, claim = _claim_for("A06")
    v = verify_claim(claim)
    assert v.status != "CONSISTENT"
    assert v.floor_convention is True
    assert v.floor_threshold is not None and abs(v.floor_threshold - 0.001) < 1e-12


def test_floor_A07():
    """DOI 10.1371/journal.pone.0267297. Цитата: 'children looked to the
    SCR at a rate above chance (t(30) = 5.37, p = .001, α = .05,
    d = 0.964)'. Вычислено: p_two_tailed(5.37, 30) ≈ 8.19e-06."""
    r, claim = _claim_for("A07")
    v = verify_claim(claim)
    assert v.status != "CONSISTENT"
    assert v.floor_convention is True
    assert v.floor_threshold is not None and abs(v.floor_threshold - 0.001) < 1e-12


def test_floor_B10():
    """DOI 10.1371/journal.pone.0058546. Цитата: '[t(14) = 4.92, p = .001,
    d = 1.81]'. Вычислено: p_two_tailed(4.92, 14) ≈ 2.26e-04 — тоже
    меньше .001, хотя и не на такой порядок, как остальные четыре: это
    ровно тот случай, где печать порога и мелкая транскрипционная ошибка
    выглядели бы неотличимо, если бы условие проверялось не строго."""
    r, claim = _claim_for("B10")
    v = verify_claim(claim)
    assert v.status != "CONSISTENT"
    assert v.floor_convention is True
    assert v.floor_threshold is not None and abs(v.floor_threshold - 0.001) < 1e-12


# --- Три настоящих расхождения, не объяснимых печатью порога --------------
#
# Все три — DOI 10.1371/journal.pone.0267297, тот же журнал; отчётность
# именно в этой статье в среднем небрежнее, чем в остальных трёх (4 из 5
# случаев печати порога — тоже оттуда).

def test_genuine_A04():
    """DOI 10.1371/journal.pone.0267297. Цитата: 'adults: M = .37,
    SD = .20, t(29) = 4.67, p = .002, α = .05, d = 0.85'.
    Вычислено: p_two_tailed(4.67, 29) ≈ 6.34e-05 — расходится с
    напечатанным .002 примерно в 30 раз. .002 не входит в типовые пороги
    печати (.05/.01/.001/.0001), поэтому конвенция здесь неприменима
    по построению: floor_convention обязан остаться False, а не
    сработать на любом расхождении, которое выглядит похожим."""
    r, claim = _claim_for("A04")
    v = verify_claim(claim)
    assert v.status != "CONSISTENT"
    assert v.floor_convention is False
    assert v.floor_threshold is None


def test_genuine_A08():
    """DOI 10.1371/journal.pone.0267297. Цитата: 'children: t(30) = 2.13,
    p = .045, α = 0.025, d = 0.38'. Вычислено: p_two_tailed(2.13, 30) ≈
    .0415 — расходится в третьем знаке, типичная транскрипционная
    ошибка, не печать порога (.045 не в списке типовых порогов)."""
    r, claim = _claim_for("A08")
    v = verify_claim(claim)
    assert v.status != "CONSISTENT"
    assert v.floor_convention is False
    assert v.floor_threshold is None


def test_genuine_A10():
    """DOI 10.1371/journal.pone.0267297. Цитата: 'looked at a rate above
    chance to the previously mentioned objects (M = .25, SD = .11,
    t(32) = 2.499, p = .020, α = .05, d = 0.43)'. Вычислено:
    p_two_tailed(2.499, 32) ≈ .0178 — расхождение в третьем знаке."""
    r, claim = _claim_for("A10")
    v = verify_claim(claim)
    assert v.status != "CONSISTENT"
    assert v.floor_convention is False
    assert v.floor_threshold is None


# --- Отрицательный контроль: согласованные утверждения не помечаются ------

@pytest.mark.parametrize("cid", [
    "A01", "A02", "A09",
    "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B13",
    "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21", "B22", "B23",
    "B24", "B25", "B26", "B27", "B28", "B29", "C01", "C02", "C03", "C04",
])
def test_consistent_cases_stay_consistent_and_unflagged(cid):
    """Все согласованные реальные утверждения — CONSISTENT, и признак
    печати порога не срабатывает на них ложно (он определён только
    внутри INCONSISTENT*-веток, но проверяем явно, а не полагаемся на
    значение по умолчанию)."""
    r, claim = _claim_for(cid)
    v = verify_claim(claim)
    assert v.status == "CONSISTENT", f"{cid}: {v.line()}"
    assert v.floor_convention is False


def test_floor_convention_never_becomes_consistent():
    """Требование задачи: признак печати порога уточняет вердикт, не
    смягчает его. Ни один из восьми расходящихся реальных случаев не
    должен превратиться в CONSISTENT из-за floor_convention."""
    floor_ids = [r["id"] for r in CORPUS_REAL if r.get("floor_convention")]
    assert len(floor_ids) == 5, floor_ids
    for cid in floor_ids:
        r, claim = _claim_for(cid)
        v = verify_claim(claim)
        assert v.status != "CONSISTENT", f"{cid} стал CONSISTENT — запрещено заданием"
        assert v.status.startswith("INCONSISTENT")
