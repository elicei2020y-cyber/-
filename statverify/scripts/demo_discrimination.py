"""Проверка несущего утверждения архитектуры.

Охранник обязан различать две причины, по которым утверждение
выглядит неправильным:

  (A) в статье ошибка   — извлечение верное, охранник ПРОПУСКАЕТ,
                          дальше отношение L4 находит противоречие
                          и это находка, а не баг;
  (B) извлекли не то    — охранник ОТКАЗЫВАЕТ, утверждение уходит
                          на переизвлечение и наружу не попадает.

Если охранник ведёт себя в этих двух случаях одинаково, петля
переизвлечения становится опасной: она либо подгоняет значения
под инвариант, отмывая настоящую ошибку статьи, либо крутится
вхолостую на верных извлечениях.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from dataclasses import replace

from statverify.extract import extract
from statverify.guard import check_claim
from statverify.model import Slot

# (A) В статье настоящая ошибка: парный тест на N = 100 обязан иметь
# df = 99, а сообщено 98. Извлечение при этом безупречно.
PAPER_ERROR = ("Participants (N = 100) completed both blocks. "
               "Scores differed across conditions, t(98) = 2.45, p = .016.")

# (B) Тот же документ, но модель взяла df из второго теста.
MIS_SOURCED = ("Accuracy rose, t(99) = 2.45, p = .016; "
               "speed fell, t(49) = 1.20, p = .236.")


def report(title, claim, source, note):
    results = check_claim(claim, source)
    ok = all(r.ok for r in results)
    print(f"\n{title}")
    print(f"  извлечено: df={claim.slots['df'].value:g}, "
          f"t={claim.slots['t'].value:g}, p={claim.slots['p'].value:g}")
    print(f"  охранник:  {'ПРОПУСТИЛ' if ok else 'ОТКАЗАЛ'}")
    for r in results:
        if not r.ok:
            print(f"    [{r.slot}] {r.reason}: {r.detail}")
    print(f"  вывод:     {note}")
    return ok


print("=" * 72)
print("РАЗЛИЧЕНИЕ ПРИЧИН ОТКАЗА")
print("=" * 72)

doc_a = extract(PAPER_ERROR)
a_ok = report(
    "(A) ошибка в статье, извлечение верное",
    doc_a.claims[0], doc_a.source,
    "df=98 действительно написано в статье → противоречие с N=100 "
    "принадлежит статье, это находка для L4/L5",
)

doc_b = extract(MIS_SOURCED)
first = doc_b.claims[0]
stolen = doc_b.claims[1].slots["df"]           # df из второго теста
broken = replace(first, slots={**first.slots, "df": stolen})
b_ok = report(
    "(B) в статье всё согласовано, извлечение промахнулось",
    broken, doc_b.source,
    "df взят за границей утверждения → переизвлечь, наружу не выдавать",
)

print()
print("=" * 72)
ok = a_ok and not b_ok
print("РЕЗУЛЬТАТ:", "различение работает" if ok else "РАЗЛИЧЕНИЕ НЕ РАБОТАЕТ")
print()
print("  Обрати внимание: в случае (B) литерал '49' лежит в настоящих")
print("  скобках после настоящего t, поэтому сравнение литерала (C1),")
print("  значения (C2) и маркера роли (C3) его пропускают. Ловит только")
print("  граница утверждения (C0) — ради этого и нужен слой L0.")
