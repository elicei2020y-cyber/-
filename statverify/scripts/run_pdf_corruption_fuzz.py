"""Порча текста, имитирующая извлечение из PDF — задача отдельная от
step1-5: там мерилась чувствительность охранника к порче УЖЕ ИЗВЛЕЧЁННОГО
Claim (tests/corrupt.py). Здесь портится исходный ТЕКСТ до извлечения, и
гоняется весь конвейер L1 (extract) + L2 (guard) заново на испорченном
тексте.

ЕДИНСТВЕННАЯ опасная метрика — доля случаев, где конвейер извлёк
НЕВЕРНОЕ значение (t, df или p, отличное от истинного), и охранник его
ПРОПУСТИЛ. Не найти утверждение в испорченном тексте — безопасно (просто
абстенция, отношение решает 'не проверено', а не 'ложь'). Опасно только
пропустить неверное значение как согласованное с исходником.

Корпус — tests/corpus_real.py, только записи has_t_test=True (43 из 49):
только они дают literal-спаны t/df/p, нужные корупциям для прицельной
порчи. У корпуса нет полей est/se/ci_lo/ci_hi (проверено отдельно),
поэтому фуззинг ограничен тройкой (t, df, p), как и весь текущий охранник
на этом корпусе.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from collections import defaultdict

from corpus_real import CORPUS_REAL
from pdf_corrupt import (CORRUPTIONS, COLUMN_SPLICE_KINDS, column_splicing,
                          ground_truth, resolve_after_corruption)

RECORDS = [r for r in CORPUS_REAL if r.get("has_t_test")]


def run_one(rec, corrupt_fn):
    gt = ground_truth(rec["text"])
    assert gt is not None, f"{rec['id']}: истина не извлекается из чистого текста"
    corrupted = corrupt_fn(rec["text"], gt)
    if corrupted is None:
        return "n/a"  # порча неприменима к этому утверждению (нет 'fi'/'fl' и т.п.)
    claim, admitted, values = resolve_after_corruption(corrupted, rec["t"], rec["df"])
    if values is None:
        return "not_found"
    if not admitted:
        return "rejected"
    truth = {"t": rec["t"], "df": rec["df"], "p": rec["p"]}
    wrong = any(abs(values[n] - truth[n]) > 1e-9 for n in ("t", "df", "p"))
    return "wrong_admitted" if wrong else "correct_admitted"


def main() -> int:
    all_corruptions = dict(CORRUPTIONS)
    for kind in COLUMN_SPLICE_KINDS:
        # три реалистичных вида вытекания колонки — каждый своя строка
        # отчёта, не одна усреднённая 'column_splicing': прошлая версия
        # порчи (одна неправдоподобная форма) показала 0%, скрыв то,
        # что реалистичные формы держались на 100% несмотря на "починку".
        all_corruptions[f"column_splicing:{kind}"] = (
            lambda t, g, k=kind: column_splicing(t, g, k))

    outcomes = defaultdict(lambda: defaultdict(int))
    dangerous_cases = defaultdict(list)

    for name, fn in all_corruptions.items():
        for rec in RECORDS:
            outcome = run_one(rec, fn)
            outcomes[name][outcome] += 1
            if outcome == "wrong_admitted":
                dangerous_cases[name].append(rec["id"])

    order = (["ligatures", "soft_hyphen", "linebreak_in_number", "lost_space",
              "footnote_glued"]
             + [f"column_splicing:{k}" for k in sorted(COLUMN_SPLICE_KINDS)]
             + ["ocr_l1_o0"])
    n = len(RECORDS)

    NAME_W = 36
    print("=" * (NAME_W + 64))
    print(f"ПОРЧА ИЗВЛЕЧЕНИЯ ИЗ PDF — {n} утверждений корпуса (has_t_test=True)")
    print("=" * (NAME_W + 64))
    header = (f"{'вид порчи':<{NAME_W}}{'применимо':>10}{'не найдено':>12}"
              f"{'отклонено':>11}{'верно':>8}{'НЕВЕРНО+ПРОШЛО':>16}"
              f"{'доля опасных':>14}")
    print(header)
    print("-" * (NAME_W + 64))
    total_applicable = 0
    total_wrong = 0
    for name in order:
        o = outcomes[name]
        applicable = n - o.get("n/a", 0)
        wrong = o.get("wrong_admitted", 0)
        total_applicable += applicable
        total_wrong += wrong
        frac = f"{wrong}/{n} = {100.0*wrong/n:.1f}%"
        print(f"{name:<{NAME_W}}{applicable:>10}{o.get('not_found', 0):>12}"
              f"{o.get('rejected', 0):>11}{o.get('correct_admitted', 0):>8}"
              f"{wrong:>16} {frac:>14}")
    print("-" * (NAME_W + 64))
    total_slots = n * len(order)
    print(f"суммарно по всем {len(order)} видам порчи: {total_wrong} опасных "
          f"случаев из {total_slots} (43 утверждения x {len(order)} видов) = "
          f"{100.0*total_wrong/total_slots:.1f}%")

    if any(dangerous_cases.values()):
        print("\nОПАСНЫЕ СЛУЧАИ (неверное значение прошло охранника), по id:")
        for name, ids in dangerous_cases.items():
            if ids:
                print(f"  {name}: {ids}")
    else:
        print(f"\nНи одного случая 'неверное значение прошло охранника' не "
              f"найдено ни на одном виде порчи из {len(order)}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
