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

from statverify.tstats import t_from_p
from statverify.verify import verify_claim

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
    for df, est, se in cases:
        for var in VARS:
            claim = corrupt_case(df, est, se, var)
            v = verify_claim(claim)
            per_var_total[var] += 1
            statuses[v.status] += 1
            if v.status == "INCONSISTENT" and v.var == var:
                per_var_correct[var] += 1
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
    return correct / total, per_var_total, per_var_correct


def section_c(per_var_total, per_var_correct):
    print()
    print("=" * 74)
    print("C. ПОЧЕМУ df ЛОКАЛИЗУЕТСЯ ХУЖЕ ОСТАЛЬНЫХ — ЧЕСТНЫЙ РАЗБОР")
    print("=" * 74)
    print("  t_crit(df, .05) = t_from_p(.05, df) сходится к 1.96 при росте df")
    print("  и меняется всё медленнее. R4:width сравнивает half-width с")
    print("  t_crit(df)·SE: при df, уже близком к асимптотике, ни один сдвиг")
    print("  df — сколь угодно большой — не выводит half-width за пределы")
    print("  интервала, отведённого точностью записи SE и CI. То же с R2a:")
    print("  двусторонний p при фиксированном t почти не зависит от df здесь.")
    print("  Это не баг локализации, а то же самое явление, что уже")
    print("  зафиксировано в CLAUDE.md как 'нормальный предел даёт бесплатное")
    print("  оправдание df' — только видно оно теперь с обеих сторон графа.")
    print()
    print(f"  {'df в сетке':<12}{'t_crit(df,.05)':>16}")
    for df in DF_GRID:
        print(f"  {df:<12}{t_from_p(0.05, df):>16.4f}")
    print()
    c, n = per_var_correct["df"], per_var_total["df"]
    others_c = sum(per_var_correct[v] for v in per_var_total if v != "df")
    others_n = sum(per_var_total[v] for v in per_var_total if v != "df")
    print(f"  df: верно локализовано {c}/{n} = {c/n:.1%}")
    print(f"  остальные шесть величин: {others_c}/{others_n} = {others_c/others_n:.1%}")


if __name__ == "__main__":
    fp, cases = section_a()
    loc, per_var_total, per_var_correct = section_b(cases)
    section_c(per_var_total, per_var_correct)

    print()
    print("=" * 74)
    print("ВОРОТА")
    print("=" * 74)
    checks = [("ложные срабатывания", fp, 0.0, "<="),
              ("верная локализация (все семь величин)", loc, 0.90, ">=")]
    ok = True
    for label, got, need, op in checks:
        good = got >= need if op == ">=" else got <= need
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {label:<42}{got:>7.1%}  нужно {op} {need:.0%}")
    print()
    if ok:
        print("  ВЕРДИКТ: задача 1 (R5, R4) выполнена по заданному критерию")
    else:
        print("  ВЕРДИКТ: критерий НЕ пройден — см. раздел C выше за причиной,")
        print("  а не за оправданием; порог не менялся под результат.")
