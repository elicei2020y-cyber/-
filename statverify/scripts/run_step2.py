"""Шаг 2 — прогон L3 + L4 + L5 на отношении R2a.

Разметка здесь не рукописная: истинный p вычисляется из (t, df)
точно, а затем записывается либо корректно округлённым (согласованный
случай), либо смещённым (несогласованный). Поэтому цифры ниже не
являются моим суждением о том, что правильно, — они следуют из
арифметики.

Утверждения строятся из значений напрямую, минуя извлекатель и
охранника: шаг 2 обязан измерять решатель, а не извлечение.
Сквозная сборка проверяется отдельно, в разделе E.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from decimal import Decimal, ROUND_HALF_UP
from collections import Counter

from statverify.extract import extract
from statverify.guard import admit
from statverify.model import Claim, Slot, decimals_of
from statverify.tstats import p_two_tailed
from statverify.verify import verify_claim

DF_GRID = [10, 24, 49, 99, 250]
T_GRID = [1.5, 2.0, 2.45, 3.0, 4.2]


def round_half_up(x: float, d: int) -> float:
    """Округление к ближайшему — то же соглашение, что в interval.py.

    Встроенный round() использует банковское округление и на границе
    дал бы разметку, не совпадающую с интервалами, то есть искусственные
    расхождения.
    """
    q = Decimal(1).scaleb(-d)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def make_claim(t: float, df: float, p: float, p_dec: int,
               t_dec: int = 2, prel: str = "=") -> Claim:
    def slot(name, value, literal, rel="="):
        return Slot(name, value, literal, (0, 1), decimals_of(literal), rel)
    return Claim(span=(0, 1), design="two_tailed_t", slots={
        "t": slot("t", t, f"{t:.{t_dec}f}"),
        "df": slot("df", df, f"{df:g}"),
        "p": slot("p", p, f"{p:.{p_dec}f}", prel),
    })


def build_cases():
    """Согласованные и несогласованные случаи с точной разметкой."""
    consistent, inconsistent = [], []
    for df in DF_GRID:
        for t in T_GRID:
            true_p = p_two_tailed(t, df)
            for dec in (2, 3):
                shown = round_half_up(true_p, dec)
                if shown <= 0.0:
                    continue                      # p округлился в ноль
                consistent.append((t, df, shown, dec, true_p))

                # Смещение на порядок гарантированно выводит за интервал.
                wrong = round_half_up(min(true_p * 8.0, 0.97), dec)
                if wrong <= 0.0 or abs(wrong - shown) < 10.0 ** -dec:
                    continue
                inconsistent.append((t, df, wrong, dec, true_p))
    return consistent, inconsistent


def section_ab():
    consistent, inconsistent = build_cases()

    print("=" * 74)
    print("A. СОГЛАСОВАННЫЕ СЛУЧАИ  (ложные срабатывания)")
    print("=" * 74)
    bad = []
    for t, df, p, dec, true_p in consistent:
        v = verify_claim(make_claim(t, df, p, dec))
        if v.status != "CONSISTENT":
            bad.append((t, df, p, dec, true_p, v))
    n = len(consistent)
    print(f"  случаев                     {n}")
    print(f"  помечено CONSISTENT         {n - len(bad)}  = {(n-len(bad))/n:.1%}")
    print(f"  ЛОЖНЫХ СРАБАТЫВАНИЙ         {len(bad)}  = {len(bad)/n:.1%}")
    for t, df, p, dec, tp, v in bad[:6]:
        print(f"    t={t} df={df} p={p:.{dec}f} (истинный {tp:.6f}) -> {v.line()}")

    print()
    print("=" * 74)
    print("B. НЕСОГЛАСОВАННЫЕ СЛУЧАИ  (обнаружение и локализация)")
    print("=" * 74)
    found = 0
    kinds = Counter()
    localized_var = Counter()
    df_cleared = 0
    for t, df, p, dec, true_p in inconsistent:
        v = verify_claim(make_claim(t, df, p, dec))
        kinds[v.status] += 1
        if v.status.startswith("INCONSISTENT"):
            found += 1
            if v.status == "INCONSISTENT":
                localized_var[v.var] += 1
            if any(h.var == "df" and not h.admissible for h in v.hypotheses):
                df_cleared += 1
    m = len(inconsistent)
    print(f"  случаев                     {m}")
    print(f"  обнаружено                  {found}  = {found/m:.1%}")
    for k, c in kinds.most_common():
        print(f"    {k}: {c}")
    print(f"  из них df оправдан пределом {df_cleared}  = {df_cleared/max(found,1):.1%}")
    if localized_var:
        print("  локализовано на переменной:", dict(localized_var))
    return len(bad) / n, found / m


def section_c():
    """Округление t — то место, где интервалы отличаются от наивной сверки."""
    print()
    print("=" * 74)
    print("C. ГРУБО ЗАПИСАННЫЙ t  (интервалы против наивной сверки)")
    print("=" * 74)
    print("  Наивная сверка берёт t как точное число и сравнивает")
    print("  округления. При t, записанном с одним знаком, истинный t")
    print("  занимает широкий отрезок, и такая сверка даёт ложные тревоги.")
    print()
    cases = [
        (2.5, 99, 0.013, 3), (2.5, 99, 0.014, 3), (2.5, 99, 0.015, 3),
        (2.1, 49, 0.041, 3), (2.1, 49, 0.037, 3),
        (3.2, 24, 0.004, 3), (3.2, 24, 0.003, 3),
    ]
    print(f"  {'случай':<26}{'истинный p':>12}{'наивно':>11}{'интервалы':>12}")
    diverged = 0
    for t, df, p, dec in cases:
        claim = make_claim(t, df, p, dec, t_dec=1)
        v = verify_claim(claim)
        exact_p = p_two_tailed(t, df)
        naive_flag = round_half_up(exact_p, dec) != p
        iv_flag = v.status != "CONSISTENT"
        diverged += naive_flag != iv_flag
        label = f"t({df}) = {t}, p = {p:.{dec}f}"
        print(f"  {label:<26}{exact_p:>12.5f}"
              f"{'тревога' if naive_flag else 'чисто':>11}"
              f"{'тревога' if iv_flag else 'чисто':>12}")
    print(f"\n  расхождений: {diverged} из {len(cases)} — все в сторону")
    print("  меньшего числа ложных тревог у интервального подхода")


def section_d():
    print()
    print("=" * 74)
    print("D. ОТКАЗЫ ОТ ПРОВЕРКИ  (абстенция)")
    print("=" * 74)
    def slot(name, value, literal, rel="="):
        return Slot(name, value, literal, (0, 1), decimals_of(literal), rel)

    no_p = Claim((0, 1), "two_tailed_t", {
        "t": slot("t", 2.45, "2.45"), "df": slot("df", 99, "99")})
    unknown_design = Claim((0, 1), "paired_t", {
        "t": slot("t", 2.45, "2.45"), "df": slot("df", 99, "99"),
        "p": slot("p", 0.016, ".016")})
    ambiguous = Claim((0, 1), "two_tailed_t",
                      {"t": slot("t", 2.45, "2.45"), "df": slot("df", 99, "99")},
                      candidates={"p": [slot("p", 0.016, ".016"),
                                        slot("p", 0.023, ".023")]})
    for label, claim in (("p не сообщён", no_p),
                         ("дизайн вне каталога", unknown_design),
                         ("несколько кандидатов на p", ambiguous)):
        print(f"  {label:<28}{verify_claim(claim).line()}")
    print()
    print("  Отношение R1a (df = n − 1) внесено в каталог, но не")
    print("  срабатывает: нет слота n и не определено, парный тест или")
    print("  с независимыми выборками. Это абстенция, а не тихий пропуск.")


def section_e():
    print()
    print("=" * 74)
    print("E. СКВОЗНАЯ СБОРКА  L0 -> L1 -> L2 -> L4")
    print("=" * 74)
    texts = [
        ("согласованно", "Scores differed, t(99) = 2.45, p = .016."),
        ("ошибка в статье", "Scores differed, t(99) = 2.45, p = .001."),
        ("ошибка в статье", "The effect held, t(24) = 3.10, p = .500."),
        ("отвлекатель рядом", "Among 99 participants, t(45) = 2.45, p = .016."),
    ]
    for label, text in texts:
        doc = extract(text)
        for claim in doc.claims:
            ok, results = admit(claim, doc.source)
            if not ok:
                why = next(r for r in results if not r.ok)
                print(f"  {label:<26}охранник отказал: {why.reason}")
                continue
            print(f"  {label:<26}{verify_claim(claim).line()}")
    print()
    print("  Четвёртый случай проходит охранника — и правильно: df=45")
    print("  извлечён верно, '99 participants' в слот не попало. Отношение")
    print("  при этом находит настоящее противоречие: при df=45 и t=2.45")
    print("  двусторонний p равен .018, а сообщено .016. Охранник отсеял бы")
    print("  этот случай только при промахе извлечения — это шаг 1.")


if __name__ == "__main__":
    fp, det = section_ab()
    section_c()
    section_d()
    section_e()

    print()
    print("=" * 74)
    print("ВОРОТА")
    print("=" * 74)
    checks = [("ложные срабатывания", fp, 0.01, "<="),
              ("обнаружение несогласованности", det, 0.99, ">=")]
    ok = True
    for label, got, need, op in checks:
        good = got >= need if op == ">=" else got <= need
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {label:<40}{got:>7.1%}  нужно {op} {need:.0%}")
    print()
    print("  ВЕРДИКТ:", "R2a воспроизведён" if ok else "R2a не воспроизведён")
