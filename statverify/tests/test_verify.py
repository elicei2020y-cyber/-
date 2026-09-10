"""Решатель: интервалы, отношение R2a, локализация, абстенция.

Разметка порождена арифметикой (tests/synth.py), а не рукой.
"""

import pytest

from statverify.extract import extract
from statverify.guard import admit
from statverify.tstats import p_two_tailed
from statverify.verify import verify_claim

from synth import build_cases, make_claim, round_half_up, slot
from statverify.model import Claim

CONSISTENT, INCONSISTENT = build_cases()


@pytest.mark.parametrize("case", CONSISTENT,
                         ids=[f"t{t}_df{df}_p{p}" for t, df, p, _, _ in CONSISTENT])
def test_no_false_positive(case):
    """Ложное срабатывание — то, что убивает такие инструменты."""
    t, df, p, dec, true_p = case
    v = verify_claim(make_claim(t, df, p, dec))
    assert v.status == "CONSISTENT", (
        f"t={t} df={df} p={p} (истинный {true_p:.6f}) -> {v.line()}")


@pytest.mark.parametrize("case", INCONSISTENT,
                         ids=[f"t{t}_df{df}_p{p}" for t, df, p, _, _ in INCONSISTENT])
def test_detects_inconsistency(case):
    t, df, p, dec, true_p = case
    v = verify_claim(make_claim(t, df, p, dec))
    assert v.status.startswith("INCONSISTENT"), (
        f"t={t} df={df} p={p} (истинный {true_p:.6f}) не обнаружено")


def test_localization_is_currently_ambiguous():
    """Зафиксировано известное ограничение, а не желаемое поведение.

    Три переменные, одно отношение — задача недоопределена, виновной
    может быть любая. Нормальный предел оправдывает df примерно в 95%
    случаев, сужая круг с трёх до двух, но не до одного.

    Этот тест обязан упасть, когда будут добавлены R5 и R4 (задача 1
    в docs/TASK_FOR_CLAUDE_CODE.md) — падение будет означать, что
    локализация заработала, и тест надо заменить на проверку того,
    что названа верная переменная.
    """
    statuses = {verify_claim(make_claim(t, df, p, dec)).status
                for t, df, p, dec, _ in INCONSISTENT}
    assert statuses == {"INCONSISTENT_AMBIGUOUS"}, (
        f"локализация изменилась: {statuses} — обнови тест по существу")


def test_df_exonerated_by_normal_limit():
    """Нормальный предел обязан снимать подозрение с df в большинстве
    случаев: при сообщённом p ниже 2(1-Ф(|t|)) никакое df не согласует
    тройку."""
    cleared = sum(
        any(h.var == "df" and not h.admissible
            for h in verify_claim(make_claim(t, df, p, dec)).hypotheses)
        for t, df, p, dec, _ in INCONSISTENT)
    assert cleared / len(INCONSISTENT) > 0.9


@pytest.mark.parametrize("t,df,p,dec", [
    (2.5, 99, 0.013, 3), (2.5, 99, 0.015, 3),
    (2.1, 49, 0.037, 3), (3.2, 24, 0.003, 3),
])
def test_interval_beats_point_comparison(t, df, p, dec):
    """Грубо записанный t — там, где интервалы отличаются от точечной сверки.

    При 't(99) = 2.5' истинная статистика занимает [2.45, 2.55), которому
    отвечает p от .0122 до .0160 — почти треть порядка. Точечная сверка
    берёт t = 2.5 буквально и объявляет ошибкой всё, кроме .014.
    Каждый случай здесь — ложная тревога точечного подхода.
    """
    naive_flags = round_half_up(p_two_tailed(t, df), dec) != p
    assert naive_flags, "случай перестал быть расхождением, пересмотри тест"
    assert verify_claim(make_claim(t, df, p, dec, t_dec=1)).status == "CONSISTENT"


def test_abstains_without_p():
    claim = Claim((0, 1), "two_tailed_t",
                  {"t": slot("t", 2.45, "2.45"), "df": slot("df", 99, "99")})
    v = verify_claim(claim)
    assert v.status == "UNVERIFIABLE" and "p" in v.reason


def test_abstains_on_unknown_design():
    """Дизайн вне каталога не проверяется наугад.

    Правило, верное «обычно», в верификаторе хуже отсутствующего:
    его ложные срабатывания неотличимы от находок.
    """
    claim = Claim((0, 1), "paired_t", {
        "t": slot("t", 2.45, "2.45"), "df": slot("df", 99, "99"),
        "p": slot("p", 0.016, ".016")})
    assert verify_claim(claim).status == "UNVERIFIABLE"


def test_ambiguous_parse_is_not_resolved_silently():
    claim = Claim((0, 1), "two_tailed_t",
                  {"t": slot("t", 2.45, "2.45"), "df": slot("df", 99, "99")},
                  candidates={"p": [slot("p", 0.016, ".016"),
                                    slot("p", 0.023, ".023")]})
    assert verify_claim(claim).status == "AMBIGUOUS_PARSE"


@pytest.mark.parametrize("text,expect", [
    ("Scores differed, t(99) = 2.45, p = .016.", "CONSISTENT"),
    ("Scores differed, t(99) = 2.45, p = .001.", "INCONSISTENT_AMBIGUOUS"),
    ("The effect held, t(24) = 3.10, p = .500.", "INCONSISTENT_AMBIGUOUS"),
])
def test_end_to_end(text, expect):
    """Сквозная сборка L0 -> L1 -> L2 -> L4."""
    doc = extract(text)
    claim = doc.claims[0]
    ok, _ = admit(claim, doc.source)
    assert ok, "охранник отказал на честном извлечении"
    assert verify_claim(claim).status == expect
