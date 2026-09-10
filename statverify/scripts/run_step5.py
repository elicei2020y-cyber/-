"""Шаг 5 — прогон L0-L4 на корпусе реальных утверждений (49 из 4 статей
PLOS ONE, tests/corpus_real.py). Урезанная версия задачи 5 из
docs/TASK_FOR_CLAUDE_CODE.md.

Разметка в corpus_real.py вычислена независимо (scripts/build_corpus_real.py,
verify_claim при её построении не вызывался) — иначе получилась бы проверка
системы самой собой. Числа здесь печатаются РАЗДЕЛЬНО по dev/holdout:
dev — рабочая половина, по ней разрешено чинить код; holdout прогоняется
один раз и по нему не чинится ничего (docs/TASK_FOR_CLAUDE_CODE.md, §5).

ПРОИСХОЖДЕНИЕ КОРПУСА, честно. Файлы tests/corpus_real.py,
scripts/build_corpus_real.py и docs/CORPUS_REAL_NOTES.md получены как
готовый артефакт, не собраны в этой сессии: прямой сетевой доступ к
journals.plos.org, arxiv.org, PMC и другим источникам из этой сессии
заблокирован политикой организации (403 на CONNECT, проверено явно).
Поэтому дословность цитат и подлинность DOI в этой сессии НЕ
перепроверены против первоисточника. Перепроверено — арифметика
(python3 scripts/build_corpus_real.py воспроизводит tests/corpus_real.py
байт в байт по всем полям всех 49 записей) и то, что числа ниже
получены реальным прогоном текущего кода, а не переписаны из присланных
заметок.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from collections import Counter

from statverify.extract import extract
from statverify.guard import admit
from statverify.verify import verify_claim

from corpus_real import CORPUS_REAL, dev, holdout


def run(rows, label):
    print("=" * 74)
    print(f"{label}  ({len(rows)} утверждений)")
    print("=" * 74)

    with_triple = [r for r in rows if r["has_t_test"]]
    without_triple = [r for r in rows if not r["has_t_test"]]

    found = guard_ok = guard_fail = 0
    unverifiable = Counter()
    fp = []                       # ложные срабатывания на согласованных
    detected_genuine = detected_floor = 0
    n_consistent = n_genuine = n_floor = 0
    status_rows = []

    for r in with_triple:
        doc = extract(r["text"])
        claim = next((c for c in doc.claims
                      if {"t", "df", "p"} <= set(c.slots)), None)
        if claim is None:
            status_rows.append((r["id"], "НЕ ИЗВЛЕЧЕНО", None))
            continue
        found += 1

        ok, results = admit(claim, doc.source)
        if not ok:
            guard_fail += 1
            status_rows.append((r["id"], "ОХРАННИК ОТКАЗАЛ",
                                [x.reason for x in results if not x.ok]))
            continue
        guard_ok += 1

        v = verify_claim(claim)
        status_rows.append((r["id"], v.status, v.line()))
        if v.status == "UNVERIFIABLE":
            unverifiable[v.reason] += 1

        if r["floor_convention"]:
            n_floor += 1
        elif r["consistent"]:
            n_consistent += 1
        else:
            n_genuine += 1

        if r["consistent"] and v.status != "CONSISTENT":
            fp.append((r["id"], v.line()))
        if (not r["consistent"]) and v.status.startswith("INCONSISTENT"):
            if r["floor_convention"]:
                detected_floor += 1
            else:
                detected_genuine += 1

    n = len(with_triple)
    print(f"  утверждений с тройкой (t, df, p)     {n}")
    print(f"  без тестовой статистики (не в счёт)  {len(without_triple)}")
    print()
    print(f"  извлечено извлекателем               {found}/{n} = {found/n:.1%}")
    print(f"  прошло охранника                     {guard_ok}/{found if found else 1}"
          f" = {guard_ok/found if found else 0:.1%}")
    print(f"  ложных отказов охранника             {guard_fail}")
    print(f"  UNVERIFIABLE                         {sum(unverifiable.values())}")
    for reason, c in unverifiable.most_common():
        print(f"    {reason:<50}{c}")
    print()
    print(f"  независимая разметка:")
    print(f"    согласовано                        {n_consistent}")
    print(f"    расходится, объяснимо порогом       {n_floor}")
    print(f"    расходится, настоящие кандидаты     {n_genuine}")
    print()
    print(f"  ложных срабатываний на согласованных  {len(fp)}")
    for cid, line in fp:
        print(f"    {cid}: {line}")
    print(f"  обнаружено (INCONSISTENT*) из объяснимых порогом"
          f"   {detected_floor}/{n_floor if n_floor else 1}")
    print(f"  обнаружено (INCONSISTENT*) из настоящих кандидатов"
          f" {detected_genuine}/{n_genuine if n_genuine else 1}")
    print()
    print("  по каждому утверждению:")
    for cid, status, detail in status_rows:
        print(f"    {cid:4s} {status}")
    print()
    return dict(found=found, n=n, guard_ok=guard_ok, guard_fail=guard_fail,
                fp=len(fp), n_consistent=n_consistent, n_floor=n_floor,
                n_genuine=n_genuine, detected_floor=detected_floor,
                detected_genuine=detected_genuine)


if __name__ == "__main__":
    dev_stats = run(dev(), "DEV (рабочая половина)")
    hold_stats = run(holdout(), "HOLDOUT (отложенная половина, прогнано один раз)")

    print("=" * 74)
    print("СВОДКА")
    print("=" * 74)
    print(f"  {'':30s}{'dev':>10s}{'holdout':>10s}")
    print(f"  {'извлечено':30s}{dev_stats['found']:>6d}/{dev_stats['n']:<3d}"
          f"{hold_stats['found']:>6d}/{hold_stats['n']:<3d}")
    print(f"  {'ложных отказов охранника':30s}{dev_stats['guard_fail']:>10d}"
          f"{hold_stats['guard_fail']:>10d}")
    print(f"  {'ложных срабатываний':30s}{dev_stats['fp']:>10d}{hold_stats['fp']:>10d}")
    print(f"  {'согласовано':30s}{dev_stats['n_consistent']:>10d}"
          f"{hold_stats['n_consistent']:>10d}")
    print(f"  {'расходится (порог)':30s}{dev_stats['n_floor']:>10d}"
          f"{hold_stats['n_floor']:>10d}")
    print(f"  {'расходится (настоящее)':30s}{dev_stats['n_genuine']:>10d}"
          f"{hold_stats['n_genuine']:>10d}")
    print()
    print("  Статей просмотрено на 50 утверждений: 9 (4 пригодных, см.")
    print("  docs/CORPUS_REAL_NOTES.md) — плотность проверяемого в")
    print("  литературе для найденных статей примерно 4/9 ≈ 44%, не")
    print("  измерена для литературы в целом (см. ограничения в заметках).")
    print()
    ok = (dev_stats["fp"] == 0 and hold_stats["fp"] == 0
          and dev_stats["guard_fail"] == 0 and hold_stats["guard_fail"] == 0)
    print("  ВЕРДИКТ:", "ворота пройдены (0 ложных срабатываний на обеих половинах)"
          if ok else "ворота НЕ пройдены — см. построчный разбор выше")
