"""Шаг 1 — прогон L0 + L1 + L2. Отношения не проверяются.

Меряется охранник, и обязательно в обе стороны:
  A. качество извлечения (контекст, не цель шага);
  B. доля ложных отказов на честном извлечении;
  C. доля пойманных на искусственной порче, с разбором того,
     тот ли критерий сработал.

Если B высока — охранник слишком строг и задушит конвейер.
Если C низка — охранник декоративен и петля переизвлечения
будет небезопасна. Идти к шагу 2 можно только при B≈0 и C≈1.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from collections import Counter, defaultdict

from corpus import CORPUS
from corrupt import CORRUPTIONS, EXPECTED_CATCHER
from statverify.extract import extract
from statverify.guard import check_claim, check_slot


def _match(found, expected):
    """Сопоставить найденные утверждения с размеченными по паре (t, df)."""
    pairs, unmatched = [], list(found)
    for exp in expected:
        hit = next((c for c in unmatched
                    if "t" in c.slots and "df" in c.slots
                    and abs(c.slots["t"].value - exp["t"]) < 1e-9
                    and abs(c.slots["df"].value - exp["df"]) < 1e-9), None)
        if hit is not None:
            unmatched.remove(hit)
        pairs.append((exp, hit))
    return pairs, unmatched


def section_a():
    print("=" * 72)
    print("A. ИЗВЛЕЧЕНИЕ (L0 + L1)")
    print("=" * 72)
    exp_total = found_total = matched = 0
    spurious = 0
    p_correct = p_expected = 0
    problems = []

    for case in CORPUS:
        doc = extract(case["text"])
        exp = case["claims"]
        exp_total += len(exp)
        found_total += len(doc.claims)
        pairs, unmatched = _match(doc.claims, exp)
        spurious += len(unmatched)

        for e, got in pairs:
            if got is None:
                problems.append(f"  {case['id']}: не найдено t={e['t']}, df={e['df']}")
                continue
            matched += 1
            if e["p"] is not None:
                p_expected += 1
                slot = got.slots.get("p")
                if slot and abs(slot.value - e["p"]) < 1e-9 and slot.relation == e["prel"]:
                    p_correct += 1
                else:
                    have = f"{slot.value}{slot.relation}" if slot else "нет"
                    problems.append(f"  {case['id']}: p ожидалось {e['p']}{e['prel']}, получено {have}")
            else:
                if "p" in got.slots:
                    problems.append(f"  {case['id']}: p разрешён принудительно, ожидались кандидаты")
        for c in unmatched:
            t = c.slots.get("t")
            problems.append(f"  {case['id']}: лишнее утверждение t={t.value if t else '?'}")

    print(f"  утверждений размечено       {exp_total}")
    print(f"  утверждений найдено         {found_total}")
    print(f"  совпало по (t, df)          {matched}/{exp_total}  = {matched/exp_total:.1%}")
    print(f"  ложных утверждений          {spurious}")
    print(f"  p извлечён верно            {p_correct}/{p_expected}  = {p_correct/max(p_expected,1):.1%}")
    if problems:
        print("  расхождения:")
        for p in problems:
            print(p)
    return matched, exp_total


def section_b():
    print()
    print("=" * 72)
    print("B. ОХРАННИК НА ЧЕСТНОМ ИЗВЛЕЧЕНИИ  (ложные отказы)")
    print("=" * 72)
    total = passed = 0
    by_reason = Counter()
    detail = []

    for case in CORPUS:
        doc = extract(case["text"])
        for claim in doc.claims:
            for r in check_claim(claim, doc.source):
                total += 1
                if r.ok:
                    passed += 1
                else:
                    by_reason[r.reason] += 1
                    detail.append(f"  {case['id']} [{r.slot}] {r.reason}: {r.detail}")

    rate = passed / max(total, 1)
    print(f"  проверок слотов             {total}")
    print(f"  прошло                      {passed}  = {rate:.1%}")
    print(f"  ложных отказов              {total - passed}  = {1 - rate:.1%}")
    for reason, n in by_reason.most_common():
        print(f"    {reason}: {n}")
    for d in detail:
        print(d)
    return rate


def section_c():
    print()
    print("=" * 72)
    print("C. ОХРАННИК НА ПОРЧЕ  (чувствительность)")
    print("=" * 72)
    stats = defaultdict(lambda: {"n": 0, "caught": 0, "by": Counter()})

    for case in CORPUS:
        doc = extract(case["text"])
        for claim in doc.claims:
            for name, fn in CORRUPTIONS.items():
                out = fn(claim, doc)
                if out is None:
                    continue
                bad_claim, slot_name = out
                slot = bad_claim.slots[slot_name]
                res = check_slot(slot, doc.source, bad_claim)
                s = stats[name]
                s["n"] += 1
                if not res.ok:
                    s["caught"] += 1
                    s["by"][res.reason] += 1
                else:
                    s["by"]["ПРОПУЩЕНО"] += 1

    print(f"  {'порча':<22}{'проч':>5}{'поймано':>9}{'доля':>8}  критерий")
    total_n = total_caught = 0
    for name in CORRUPTIONS:
        s = stats[name]
        if s["n"] == 0:
            print(f"  {name:<22}{'-':>5}{'-':>9}{'-':>8}  неприменима к корпусу")
            continue
        total_n += s["n"]
        total_caught += s["caught"]
        frac = s["caught"] / s["n"]
        want = EXPECTED_CATCHER[name]
        by = ", ".join(f"{k}×{v}" for k, v in s["by"].most_common())
        flag = "" if set(s["by"]) == {want} else "   <- не тот критерий"
        print(f"  {name:<22}{s['n']:>5}{s['caught']:>9}{frac:>8.0%}  {by}{flag}")

    rate = total_caught / max(total_n, 1)
    print(f"\n  итого поймано               {total_caught}/{total_n}  = {rate:.1%}")
    return rate


if __name__ == "__main__":
    matched, exp_total = section_a()
    b = section_b()
    c = section_c()

    print()
    print("=" * 72)
    print("ВОРОТА ПЕРЕХОДА К ШАГУ 2")
    print("=" * 72)
    checks = [
        ("извлечение находит размеченные утверждения", matched / exp_total, 0.90, ">="),
        ("ложных отказов охранника", 1 - b, 0.02, "<="),
        ("порча поймана", c, 0.99, ">="),
    ]
    ok = True
    for label, got, need, op in checks:
        good = got >= need if op == ">=" else got <= need
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {label:<44}{got:>7.1%}  нужно {op} {need:.0%}")
    print()
    print("  ВЕРДИКТ:", "можно идти к шагу 2" if ok else "шаг 2 преждевременен")
