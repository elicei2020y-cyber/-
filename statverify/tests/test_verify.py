"""Решатель: интервалы, отношение R2a, локализация, абстенция.

Разметка порождена арифметикой (tests/synth.py), а не рукой.
"""

import pytest

from statverify.extract import extract
from statverify.guard import admit
from statverify.relations import R2A, R4_CENTER, R4_WIDTH, R5
from statverify.tstats import p_two_tailed
from statverify.verify import reported_intervals, verify_claim

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


def test_localization_is_ambiguous_without_r5_r4():
    """Три переменные, одно отношение — задача недоопределена без R5/R4.

    Раньше это было единственное поведение локализации (задача 1 из
    docs/TASK_FOR_CLAUDE_CODE.md ещё не сделана — CATALOG состоял из
    R2a и пустого R1a). Теперь R5 и R4 в каталоге есть (relations.py),
    но НЕ применимы к этим конкретным случаям: synth.build_cases()
    строит только t, df, p, а Relation.solve требует все свои
    переменные — без est/se/ci_lo/ci_hi R5 и R4 просто не входят в
    `usable`, и остаётся ровно R2a, та же тройка, что и раньше.

    Это не регресс и не обход задачи 1: test_localization_with_r5_r4
    ниже показывает тот же граф на случаях, где все семь величин
    присутствуют, — и там локализация действительно называет виновника.
    """
    statuses = {verify_claim(make_claim(t, df, p, dec)).status
                for t, df, p, dec, _ in INCONSISTENT}
    assert statuses == {"INCONSISTENT_AMBIGUOUS"}, (
        f"локализация изменилась: {statuses} — обнови тест по существу")


def test_localization_with_r5_r4():
    """Задача 1, критерий приёмки: с графом R2a+R5+R4:center+R4:width
    локализация называет виновника для t/p/est/se/ci_lo/ci_hi — всегда,
    для df — в 15/33 (45.5%; 15/23=65.2% среди детектируемых, см.
    test_solve_is_sound_on_consistent_grid и run_step3.py §A2 про
    оставшиеся 10/33 генуинно неразличимых).

    df — единственное честное исключение, не послабление теста: df
    входит в R2a (через t, p) и в R4:width (через ci_lo, ci_hi, se)
    только как ОБРАЩЁННОЕ решение df_from, инвертирующее функцию с
    горизонтальной асимптотой (t_crit(df,.05) -> 1.9600 и симметрично
    нормальный предел для p — обе никогда не достигаются ни при каком
    конечном df). Оба решателя поэтому дают ШИРОКИЕ (но состоятельные —
    см. инвариантный тест) интервалы при df >= 25, которые реже
    пересекаются пусто друг с другом, — честная неопределённость самой
    статистики, не брак кода. (Раньше здесь стоял компромисс, где 2 из
    33 случаев порчи p давали AMBIGUOUS вместо точного 'p' — это было
    побочным эффектом временного удаления решателя df у R4:width, самого
    по себе несостоятельного из-за бага в Relation.solve; после починки
    solve() решатель восстановлен, и компромисс исчез — p снова 33/33.)

    Разметка — из tests/synth7.py, того же генератора, что и у
    scripts/run_step3.py, поэтому числа здесь и там совпадают дословно.
    """
    from synth7 import VARS, build_grid, corrupt_case

    cases = list(build_grid())
    per_var = {v: [0, 0] for v in VARS}          # [верно, всего]
    for df, est, se in cases:
        for var in VARS:
            v = verify_claim(corrupt_case(df, est, se, var))
            per_var[var][1] += 1
            if v.status == "INCONSISTENT" and v.var == var:
                per_var[var][0] += 1

    expected_min_rate = {"df": 0.40}
    for var in VARS:
        correct, total = per_var[var]
        rate = correct / total
        need = expected_min_rate.get(var, 1.0)
        assert rate >= need, (
            f"{var}: {correct}/{total} = {rate:.1%} — хуже наблюдавшегося "
            f"({need:.0%}); если это регресс, а не улучшение, разберись "
            f"прежде, чем поднимать порог")


def test_no_false_positive_seven_values():
    """R5 и R4 не вносят ложных срабатываний на согласованных случаях —
    то же требование правила 1.5, что и для R2a, теперь на всех семи
    величинах разом."""
    from synth7 import build_grid, gold, make_claim as make_claim7

    for df, est, se in build_grid():
        v = verify_claim(make_claim7(gold(df, est, se)))
        assert v.status == "CONSISTENT", f"df={df} est={est} se={se} -> {v.line()}"


def test_solve_is_sound_on_consistent_grid():
    """Инвариант состоятельности Relation.solve — не про локализацию,
    должен держаться всегда.

    Для согласованной (не испорченной) семёрки, для КАЖДОГО отношения и
    КАЖДОЙ его решаемой переменной, solve() обязан вернуть интервал,
    содержащий истинное значение этой переменной. Иначе распространение
    интервалов несостоятельно: узкий интервал, не содержащий истину,
    способен дать INCONSISTENT на согласованном утверждении — то самое
    ложное срабатывание, ради предотвращения которого построена вся
    интервальная арифметика (правило 5).

    Обнаружено внешней проверкой: solve() раньше молча отбрасывал углы
    бокса, где решатель вернул None, и брал min/max только по успешным
    углам. Для решателей, инвертирующих df_from (R2a: 'df' через t,p) —
    функцию с горизонтальной асимптотой (t_crit(df,alpha) -> const при
    df -> inf, никогда её не достигая) — истинная комбинация входов
    после округления может лежать так, что часть углов проваливается ЗА
    порог (None) и отбрасывается, а уцелевшие углы систематически смещены
    в сторону, не содержащую истинное df. На реальной сетке синуса это
    было не гипотетикой: 7 из 33 случаев R2a->df нарушали инвариант
    ДО починки (см. историю коммита). Починка (relations.Relation.solve):
    если решатель определён не на всех углах бокса, интервал расширяется
    до полной области определения цели, а не до диапазона одних лишь
    успешных углов — единственный способ остаться консервативным, не
    зная заранее, с какой стороны решатель не определён.
    """
    from synth7 import build_grid, gold, make_claim as make_claim7

    checked = 0
    for df, est, se in build_grid():
        true = gold(df, est, se)
        claim = make_claim7(true)
        rep = reported_intervals(claim)
        for rel in (R2A, R5, R4_CENTER, R4_WIDTH):
            for target in rel.solvers:
                others = {k: v for k, v in rep.items()
                          if k != target and k in rel.vars}
                d = rel.solve(target, others)
                true_val = true[target] if target != "alpha" else claim.ci_alpha
                checked += 1
                assert d is not None and d.contains(true_val), (
                    f"{rel.key}.solve({target}) = {d} не содержит истинное "
                    f"{true_val} при df={df} est={est} se={se} — "
                    f"несостоятельное распространение интервалов")
    assert checked > 300, f"подозрительно мало проверок: {checked}"


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
