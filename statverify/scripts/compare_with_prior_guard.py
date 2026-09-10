"""Сравнение с прежним охранником проекта claim_audit.

Прежний охранник (verify_extracted_number) проверяет, что число
встречается где-то в тексте после нормализации формата, — это проверка
ПРИСУТСТВИЯ значения. Здешний проверяет ПРОВЕНАНС: то же число по
указанному смещению, внутри границ утверждения, с маркером роли
вплотную перед ним.

Разница не косметическая: выдумывание ловят оба, а взятие числа не
оттуда — только второй. И это доминирующий режим отказа у statcheck
и у любого модельного извлекателя, потому что модель обычно берёт
настоящее число из настоящего места, просто не из того.

Прежний охранник в этот репозиторий не входит. Укажи путь к каталогу
extensions проекта claim_audit:

    python3 scripts/compare_with_prior_guard.py /path/to/claim_audit/extensions

Измеренный результат на приведённых ниже случаях: прежний 4 верных
вердикта из 7, здешний 7 из 7.
"""

import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from statverify.extract import extract
from statverify.guard import check_slot

# (текст, что якобы извлечено как df, честно ли извлечение, подпись)
CASES = [
    ("Among 99 participants, t(45) = 2.45, p = .016.", 45.0, True,
     "df взят верно"),
    ("Among 99 participants, t(45) = 2.45, p = .016.", 99.0, False,
     "df взят из числа участников"),
    ("Accuracy rose, t(99) = 2.45, p = .016; speed fell, t(49) = 1.20, p = .236.",
     99.0, True, "df первого теста"),
    ("Accuracy rose, t(99) = 2.45, p = .016; speed fell, t(49) = 1.20, p = .236.",
     49.0, False, "df взят из второго теста"),
    ("The contrast held, t(45) = 2.45, p = .045.", 45.0, True,
     "df взят верно"),
    ("All 30 trials counted, t(30) = 2.30, p = .030.", 30.0, False,
     "df взят из числа проб, значение совпало"),
    ("Scores differed, t(98) = 2.45, p = .016.", 77.0, False,
     "df выдуман, в тексте отсутствует"),
]


def new_guard_admits(text: str, claimed_df: float, honest: bool) -> bool:
    doc = extract(text)
    if not doc.claims:
        return False
    claim = doc.claims[0]
    true_df = claim.slots["df"]
    if honest:
        return check_slot(true_df, doc.source, claim).ok

    # Смоделировать промах: показать слотом на то же число в другом месте.
    lit, idx, start = f"{claimed_df:g}", -1, 0
    while True:
        j = doc.source.find(lit, start)
        if j == -1:
            break
        if (j, j + len(lit)) != true_df.span:
            idx = j
            break
        start = j + 1
    slot = (replace(true_df, value=claimed_df, literal=lit) if idx == -1
            else replace(true_df, value=claimed_df, literal=lit,
                         span=(idx, idx + len(lit))))
    return check_slot(slot, doc.source, claim).ok


def main() -> int:
    prior = None
    if len(sys.argv) > 1:
        sys.path.insert(0, sys.argv[1])
        try:
            from extraction_guard import verify_extracted_number as prior
        except ImportError:
            print(f"не найден extraction_guard.py в {sys.argv[1]}")
            return 1
    else:
        print(__doc__.strip().splitlines()[-4].strip())
        print("Путь к прежнему охраннику не указан — показан только здешний.\n")

    print("=" * 78)
    print("ПРЕЖНИЙ ОХРАННИК (присутствие значения)  vs  ЗДЕШНИЙ (провенанс)")
    print("=" * 78)
    print(f"  {'случай':<46}{'честно':>8}{'прежний':>10}{'здешний':>10}")
    print("-" * 78)

    old_right = new_right = 0
    for text, df, honest, label in CASES:
        new_pass = new_guard_admits(text, df, honest)
        new_right += new_pass == honest
        if prior:
            old_pass = prior(df, text)
            old_right += old_pass == honest
            old_cell = ("пропуск" if old_pass else "отказ") + ("" if old_pass == honest else " !")
        else:
            old_cell = "—"
        new_cell = ("пропуск" if new_pass else "отказ") + ("" if new_pass == honest else " !")
        print(f"  {label:<46}{'да' if honest else 'нет':>8}{old_cell:>10}{new_cell:>10}")

    n = len(CASES)
    print("-" * 78)
    old_cell = f"{old_right}/{n}" if prior else "—"
    print(f"  {'верных вердиктов':<46}{'':>8}{old_cell:>10}{f'{new_right}/{n}':>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
