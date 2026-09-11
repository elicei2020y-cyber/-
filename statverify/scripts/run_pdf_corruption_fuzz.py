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
from pdf_corrupt import CORRUPTIONS, column_splicing, ground_truth, resolve_after_corruption

RECORDS = [r for r in CORPUS_REAL if r.get("has_t_test")]


def run_one(rec, corruption_name, corrupt_fn):
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
    all_corruptions["column_splicing"] = None  # обрабатывается отдельно ниже

    outcomes = defaultdict(lambda: defaultdict(int))
    dangerous_cases = defaultdict(list)

    for name in all_corruptions:
        for i, rec in enumerate(RECORDS):
            if name == "column_splicing":
                gt = ground_truth(rec["text"])
                assert gt is not None
                foreign = RECORDS[(i + 1) % len(RECORDS)]["text"]
                outcome = run_one(rec, name,
                                   lambda t, g, f=foreign: column_splicing(t, g, f))
            else:
                outcome = run_one(rec, name, CORRUPTIONS[name])
            outcomes[name][outcome] += 1
            if outcome == "wrong_admitted":
                dangerous_cases[name].append(rec["id"])

    order = ["ligatures", "soft_hyphen", "linebreak_in_number", "lost_space",
             "footnote_glued", "column_splicing", "ocr_l1_o0"]
    n = len(RECORDS)

    print("=" * 100)
    print(f"ПОРЧА ИЗВЛЕЧЕНИЯ ИЗ PDF — {n} утверждений корпуса (has_t_test=True)")
    print("=" * 100)
    header = (f"{'вид порчи':<22}{'применимо':>10}{'не найдено':>12}"
              f"{'отклонено':>11}{'верно':>8}{'НЕВЕРНО+ПРОШЛО':>16}"
              f"{'доля опасных':>14}")
    print(header)
    print("-" * 100)
    total_applicable = 0
    total_wrong = 0
    for name in order:
        o = outcomes[name]
        applicable = n - o.get("n/a", 0)
        wrong = o.get("wrong_admitted", 0)
        total_applicable += applicable
        total_wrong += wrong
        frac = f"{wrong}/{n} = {100.0*wrong/n:.1f}%"
        print(f"{name:<22}{applicable:>10}{o.get('not_found', 0):>12}"
              f"{o.get('rejected', 0):>11}{o.get('correct_admitted', 0):>8}"
              f"{wrong:>16}{frac:>14}")
    print("-" * 100)
    total_slots = n * len(order)
    print(f"суммарно по всем 7 видам порчи: {total_wrong} опасных случаев из "
          f"{total_slots} (43 утверждения x 7 видов) = "
          f"{100.0*total_wrong/total_slots:.1f}%")

    if any(dangerous_cases.values()):
        print("\nОПАСНЫЕ СЛУЧАИ (неверное значение прошло охранника), по id:")
        for name, ids in dangerous_cases.items():
            if ids:
                print(f"  {name}: {ids}")
    else:
        print("\nНи одного случая 'неверное значение прошло охранника' не найдено "
              "ни на одном виде порчи из семи.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
