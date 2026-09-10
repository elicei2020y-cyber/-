"""Шаг 3 — прогон L3 + L4 + L5 на графе из R2a, R5, R4:center, R4:width.

Задача 1 (docs/TASK_FOR_CLAUDE_CODE.md, §2): до R5/R4 локализация была
принципиально AMBIGUOUS — три переменные (t, df, p), одно отношение,
виновной могла быть любая. R5 (t = est/SE) и R4 (доверительный интервал,
разложенный на центр и полуширину) добавляют вторую и третью связи,
пересекающиеся с R2a и друг с другом по переменным t, est, se, df,
ci_lo, ci_hi. Испорченная переменная теперь ломает ВСЕ отношения, где
она участвует, и оставляет согласованными все отношения, где её нет —
это и даёт локализацию (verify.localize).

Разметка (tests/synth7.py) порождена арифметикой, как и в run_step2.py:
задаются (df, est, se), из них по формулам самого каталога вычисляются
t, p и границы CI. "Порча" — исходное сообщённое значение одной величины,
сдвинутое настолько, чтобы гарантированно выйти за её собственный
интервал. Числа здесь и в tests/test_verify.py::test_localization_with_r5_r4
происходят из одного и того же генератора, не из двух похожих.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from collections import Counter

from statverify.relations import R2A, R4_WIDTH
from statverify.tstats import t_from_p
from statverify.verify import reported_intervals, verify_claim

from synth7 import DF_GRID, VARS, build_grid, corrupt_case, gold, make_claim


def section_a():
    print("=" * 74)
    print("A. СОГЛАСОВАННЫЕ СЛУЧАИ  (ложные срабатывания на всех семи величинах)")
    print("=" * 74)
    cases = list(build_grid())
    bad = []
    for df, est, se in cases:
        v = verify_claim(make_claim(gold(df, est, se)))
        if v.status != "CONSISTENT":
            bad.append((df, est, se, v))
    n = len(cases)
    print(f"  случаев                     {n}")
    print(f"  помечено CONSISTENT         {n - len(bad)}  = {(n-len(bad))/n:.1%}")
    print(f"  ЛОЖНЫХ СРАБАТЫВАНИЙ         {len(bad)}  = {len(bad)/n:.1%}")
    for df, est, se, v in bad[:6]:
        print(f"    df={df} est={est} se={se} -> {v.line()}")
    return len(bad) / n, cases


def section_b(cases):
    print()
    print("=" * 74)
    print("B. ЛОКАЛИЗАЦИЯ  (порча ровно одной величины из семи)")
    print("=" * 74)
    per_var_total = Counter()
    per_var_correct = Counter()
    statuses = Counter()
    wrong_var: list = []
    undetected: list = []          # CONSISTENT при заведомой порче
    for df, est, se in cases:
        for var in VARS:
            claim = corrupt_case(df, est, se, var)
            v = verify_claim(claim)
            per_var_total[var] += 1
            statuses[v.status] += 1
            if v.status == "INCONSISTENT" and v.var == var:
                per_var_correct[var] += 1
            elif v.status == "CONSISTENT":
                undetected.append((df, est, se, var))
            elif v.status.startswith("INCONSISTENT") and len(wrong_var) < 6:
                wrong_var.append((df, est, se, var, v))

    total = sum(per_var_total.values())
    correct = sum(per_var_correct.values())
    print(f"  {'величина':<10}{'случаев':>10}{'верно локализовано':>22}{'доля':>10}")
    for var in VARS:
        n = per_var_total[var]
        c = per_var_correct[var]
        print(f"  {var:<10}{n:>10}{c:>22}{c/n:>9.1%}")
    print(f"  {'ИТОГО':<10}{total:>10}{correct:>22}{correct/total:>9.1%}")
    print()
    print("  распределение вердиктов по всем испорченным случаям:")
    for status, c in statuses.most_common():
        print(f"    {status:<28}{c:>5}  = {c/total:.1%}")
    if wrong_var:
        print()
        print("  образцы неверной/неполной локализации:")
        for df, est, se, var, v in wrong_var:
            print(f"    испорчено {var:<6} df={df:<4} est={est} se={se} -> {v.line()}")
    return correct / total, per_var_total, per_var_correct, undetected


def section_a2(undetected):
    """Знаменатель локализации, пересчитанный на детектируемые случаи.

    CONSISTENT при заведомой порче — не обязательно промах: если из
    ОСТАЛЬНЫХ (не испорченных) величин порченное значение выводится
    как допустимое, противоречия действительно нет, и CONSISTENT —
    правильный ответ, а не дыра в обнаружении. Проверяется через
    ту же самую пару прямых (не инвертированных) проверок, которыми
    verify_claim реально принимает решение: derived-p из R2a и
    derived-se из R4:width, взятые на ЗАЯВЛЕННОЙ (испорченной) величине.
    Направление через Relation.solve(target=испорченная_величина, ...)
    сюда сознательно не годится: именно оно оказалось ненадёжным для df
    (см. раздел C) — из-за асимптотики оно может дать заведомо узкий и
    не содержащий истину интервал, и использовать его как критерий
    'было ли различимо' значило бы объяснять баг тем же самым багом.
    """
    print()
    print("=" * 74)
    print("A2. ЗНАМЕНАТЕЛЬ ЛОКАЛИЗАЦИИ  (пересчёт без недетектируемых случаев)")
    print("=" * 74)
    by_var = Counter(var for *_, var in undetected)
    print(f"  случаев CONSISTENT при заведомой порче: {len(undetected)}")
    print(f"  из них по величинам: {dict(by_var)}")
    print()
    genuinely_blind = 0
    for df, est, se, var in undetected:
        claim = corrupt_case(df, est, se, var)
        rep = reported_intervals(claim)
        checks = []
        if var != "p" and "p" in rep:
            others = {k: v for k, v in rep.items() if k != "p" and k in R2A.vars}
            d = R2A.solve("p", others)
            checks.append(("R2a->p", d is not None and d.intersects(rep["p"])))
        if var != "se" and all(k in rep for k in R4_WIDTH.vars):
            others = {k: v for k, v in rep.items() if k != "se" and k in R4_WIDTH.vars}
            d = R4_WIDTH.solve("se", others)
            checks.append(("R4w->se", d is not None and d.intersects(rep["se"])))
        blind = all(ok for _, ok in checks) if checks else True
        genuinely_blind += blind
        label = ", ".join(f"{name}={'держит' if ok else 'НЕ держит'}" for name, ok in checks)
        print(f"    df={df:<4} est={est:<4} se={se:<5} испорчено={var:<6} {label}"
              f"  ->  {'генуинно неразличимо' if blind else 'ДЫРА В ОБНАРУЖЕНИИ'}")
    print()
    print(f"  генуинно неразличимых: {genuinely_blind}/{len(undetected)}")
    return genuinely_blind, len(undetected) - genuinely_blind


def section_c(per_var_total, per_var_correct, n_undetected):
    print()
    print("=" * 74)
    print("C. ПОЧЕМУ df ЛОКАЛИЗУЕТСЯ ХУЖЕ ОСТАЛЬНЫХ — ЧЕСТНЫЙ РАЗБОР")
    print("=" * 74)
    print("  Третья версия этого раздела. Первая объясняла слабую локализацию")
    print("  df 'уплощением t_crit к 1.96' — не подтвердилось. Вторая чинила")
    print("  это удалением df из решателей R4:width — устраняла симптом, а не")
    print("  причину: Relation.solve сам был несостоятелен (молча отбрасывал")
    print("  углы бокса, где решатель возвращал None, из-за чего интервал мог")
    print("  не содержать истинное значение — найдено внешней проверкой и")
    print("  подтверждено: 7 из 396 проверок инварианта состоятельности на")
    print("  R2a->df нарушались ДО этой правки, то есть баг был живой и в")
    print("  оставшемся после второй версии коде). Починка — в Relation.solve:")
    print("  если решатель определён не на всех углах бокса, интервал")
    print("  расширяется до полной области определения цели, а не до диапазона")
    print("  одних лишь успешных углов. После этого решатель df у R4:width")
    print("  снова состоятелен и восстановлен в каталоге.")
    print()
    print("  Что осталось верным. df входит в R2a (через t, p) и в R4:width")
    print("  (через ci_lo, ci_hi, se) КАК ОБРАЩЁННОЕ решение — 'каким должен")
    print("  быть df, чтобы согласовать остальное'. Обращение df_from(t, p)")
    print("  инвертирует функцию с горизонтальной асимптотой: t_crit(df,.05)")
    print("  стремится к 1.9600 и никогда её не достигает ни при каком")
    print("  конечном df, и симметрично — двусторонний p при фиксированном t")
    print("  стремится к своему нормальному пределу и никогда его не достигает.")
    print("  Из-за этого ОБА решателя (R2a->df и R4:width->df) на практике дают")
    print("  ШИРОКИЕ интервалы при df, близком к асимптотике (типично уже")
    print("  df >= 25) — это состоятельно (содержит истину), но малоинформативно:")
    print("  широкий интервал редко пересекается пусто с ДРУГИМ широким")
    print("  интервалом, поэтому меньше случаев дают однозначное 'виновата")
    print("  только df', и больше остаются честно AMBIGUOUS. Это ожидаемое")
    print("  свойство самой статистики, а не брак локализации.")
    print()
    print(f"  {'df в сетке':<12}{'t_crit(df,.05)':>16}")
    for df in DF_GRID:
        print(f"  {df:<12}{t_from_p(0.05, df):>16.4f}")
    print()
    c, n = per_var_correct["df"], per_var_total["df"]
    others_c = sum(per_var_correct[v] for v in per_var_total if v != "df")
    others_n = sum(per_var_total[v] for v in per_var_total if v != "df")
    print(f"  df: верно локализовано {c}/{n} = {c/n:.1%}  (знаменатель = все случаи)")
    n_detected_df = n - n_undetected
    if n_detected_df:
        print(f"  df: верно локализовано {c}/{n_detected_df} = {c/n_detected_df:.1%}"
              f"  (знаменатель = только обнаруженные — {n_undetected} генуинно")
        print(f"       неразличимых исключены, см. раздел A2)")
    print(f"  остальные шесть величин: {others_c}/{others_n} = {others_c/others_n:.1%}")


if __name__ == "__main__":
    fp, cases = section_a()
    loc, per_var_total, per_var_correct, undetected = section_b(cases)
    n_blind, n_hole = section_a2(undetected)
    n_undetected_df = sum(1 for *_, var in undetected if var == "df")
    section_c(per_var_total, per_var_correct, n_undetected_df)

    total = sum(per_var_total.values())
    correct = sum(per_var_correct.values())
    loc_corrected = correct / (total - len(undetected)) if total > len(undetected) else float("nan")

    print()
    print("=" * 74)
    print("ВОРОТА")
    print("=" * 74)
    checks = [("ложные срабатывания", fp, 0.0, "<="),
              ("верная локализация (знаменатель = все случаи)", loc, 0.90, ">="),
              ("верная локализация (знаменатель = только обнаруженные)",
               loc_corrected, 0.90, ">=")]
    ok = True
    for label, got, need, op in checks:
        good = got >= need if op == ">=" else got <= need
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {label:<52}{got:>7.1%}  нужно {op} {need:.0%}")
    print()
    print(f"  Оба числа локализации — законные метрики РАЗНОГО: первое —")
    print(f"  сквозная точность включая честную абстенцию на недетектируемом;")
    print(f"  второе — точность виновника РОВНО там, где противоречие вообще")
    print(f"  нашлось. Второе точнее отвечает на 'верно ли называет виновника,")
    print(f"  когда вообще что-то заметил' — {n_blind} из {len(undetected)}")
    print(f"  случаев CONSISTENT честны (раздел A2){'; ' + str(n_hole) + ' — дыра в обнаружении' if n_hole else ', дыр в обнаружении нет'}.")
    print()
    if ok:
        print("  ВЕРДИКТ: задача 1 (R5, R4) выполнена по заданному критерию")
    else:
        print("  ВЕРДИКТ: критерий НЕ пройден — см. раздел C выше за причиной,")
        print("  а не за оправданием; порог не менялся под результат.")
