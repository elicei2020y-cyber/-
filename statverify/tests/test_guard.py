"""Охранник извлечения — обе стороны.

Охранник, который всё пропускает, бесполезен, поэтому мерить его надо
и на честном извлечении (ложные отказы), и на порче (чувствительность).

Про 100% прохождения на честном извлечении: при регексповом извлекателе
span берутся из позиций групп совпадения, поэтому C0, C1 и C2 проходят
по построению. Настоящее содержание — в тестах на порчу, где виды 5-8
сделаны правдоподобными: литерал настоящий и лежит по месту, поэтому
C1 и C2 на них слепы, и ловить обязаны C0 (границы) и C3 (роль).
"""

import pytest

from statverify.extract import extract
from statverify.guard import check_claim, check_slot
from statverify.model import Slot

from corpus import CORPUS
from corpus_real import CORPUS_REAL
from corrupt import CORRUPTIONS, EXPECTED_CATCHER


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c["id"])
def test_no_false_rejection(case):
    """Честное извлечение обязано проходить охранника целиком."""
    doc = extract(case["text"])
    for claim in doc.claims:
        for r in check_claim(claim, doc.source):
            assert r.ok, f"{case['id']} [{r.slot}] {r.reason}: {r.detail}"


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruption_is_caught(name):
    """Каждая порча ловится, и именно предназначенным критерием.

    Срабатывание чужого критерия — не победа: значит, критерии
    перекрываются не так, как задумано, и покрытие мнимое.
    """
    fn = CORRUPTIONS[name]
    fired = 0
    for case in CORPUS:
        doc = extract(case["text"])
        for claim in doc.claims:
            out = fn(claim, doc)
            if out is None:
                continue
            bad_claim, slot_name = out
            fired += 1
            res = check_slot(bad_claim.slots[slot_name], doc.source, bad_claim)
            assert not res.ok, f"{name} на {case['id']} ПРОПУЩЕНА"
            assert res.reason == EXPECTED_CATCHER[name], (
                f"{name} на {case['id']} поймана критерием {res.reason}, "
                f"ожидался {EXPECTED_CATCHER[name]}")
    assert fired > 0, f"порча {name} ни разу не применилась — покрытие мнимое"


def test_new_slots_pass_on_honest_extraction():
    """est/se/ci_lo/ci_hi из задачи 1 обязаны проходить охранника целиком,
    как и старые слоты — тот же C0-C4, применённый к новым ролям."""
    doc = extract("Scores differed, t(98) = 3.75, p = .001, b = 0.45, "
                  "SE = 0.12, 95% CI [0.21, 0.69].")
    claim = doc.claims[0]
    for r in check_claim(claim, doc.source):
        assert r.ok, f"[{r.slot}] {r.reason}: {r.detail}"


def test_est_se_role_confusion_is_caught():
    """Значение se записано в слот est — литерал настоящий, лежит по
    месту (C1, C2 слепы), ловить обязан C3 (маркер роли), в точности
    как role_confusion в corrupt.py для t/p."""
    from dataclasses import replace as _replace

    doc = extract("Scores differed, t(98) = 3.75, p = .001, b = 0.45, "
                  "SE = 0.12, 95% CI [0.21, 0.69].")
    claim = doc.claims[0]
    se = claim.slots["se"]
    fake_est = Slot("est", se.value, se.literal, se.span, se.decimals, "=")
    broken = _replace(claim, slots={**claim.slots, "est": fake_est})
    res = check_slot(broken.slots["est"], doc.source, broken)
    assert not res.ok
    assert res.reason == "role_missing"


def test_ci_bounds_swapped_role_confusion_is_caught():
    """ci_hi, записанный в слот ci_lo (например, если извлекатель перепутал
    границы) — литерал настоящий, роль не подтверждается: перед верхней
    границей стоит запятая/'to', а не 'CI ['."""
    from dataclasses import replace as _replace

    doc = extract("Scores differed, t(98) = 3.75, p = .001, "
                  "95% CI [0.21, 0.69].")
    claim = doc.claims[0]
    hi = claim.slots["ci_hi"]
    fake_lo = Slot("ci_lo", hi.value, hi.literal, hi.span, hi.decimals, "=")
    broken = _replace(claim, slots={**claim.slots, "ci_lo": fake_lo})
    res = check_slot(broken.slots["ci_lo"], doc.source, broken)
    assert not res.ok
    assert res.reason == "role_missing"


def test_discriminates_paper_error_from_mis_extraction():
    """Несущее утверждение архитектуры.

    (A) в статье ошибка, извлечение верное  -> охранник ПРОПУСКАЕТ,
        противоречие уходит дальше как находка;
    (B) статья согласована, извлечение промахнулось -> охранник ОТКАЗЫВАЕТ.

    Если поведение в этих случаях одинаково, петля переизвлечения
    становится опасной: она либо подгоняет значения под инвариант,
    отмывая настоящую ошибку статьи, либо крутится на верных извлечениях.
    """
    from dataclasses import replace

    a = extract("Participants (N = 100) completed both blocks. "
                "Scores differed, t(98) = 2.45, p = .016.")
    assert all(r.ok for r in check_claim(a.claims[0], a.source))

    b = extract("Accuracy rose, t(99) = 2.45, p = .016; "
                "speed fell, t(49) = 1.20, p = .236.")
    first = b.claims[0]
    stolen = b.claims[1].slots["df"]          # df из соседнего теста
    broken = replace(first, slots={**first.slots, "df": stolen})
    results = check_claim(broken, b.source)
    assert not all(r.ok for r in results)
    # Литерал настоящий и в настоящих скобках после настоящего t,
    # поэтому C1 и C3 слепы — ловит граница утверждения.
    assert any(r.reason == "not_contained" for r in results if not r.ok)


def test_c5_catches_linebreak_truncated_number():
    """ДЫРА 1 (найдена мутационным фаззингом PDF-порчи, реальный случай
    corpus_real.py id='A01', PLOS ONE 10.1371/journal.pone.0267297):
    'relatedness measures ... t(15) = 3.82, p = .002)' — настоящее
    p = .002. Разрыв строки внутри числа ('.002' -> '.00\\n2)') даёт
    извлекателю синтаксически валидный, но УСЕЧЁННЫЙ литерал '.00'
    (значение 0.0 вместо истинных 0.002). До C5 это проходило все
    четыре критерия: span на месте, значение разбирается из литерала,
    маркер роли 'p =' стоит вплотную, вложенность соблюдена — критерия
    ПОЛНОТЫ литерала не было вовсе. C5 обязан поймать именно это,
    отдельно и независимо от C0-C4.
    """
    rec = next(r for r in CORPUS_REAL if r["id"] == "A01")
    assert rec["p"] == 0.002 and "p = .002)" in rec["text"]
    corrupted_text = rec["text"].replace("p = .002)", "p = .00\n2)")
    assert corrupted_text != rec["text"]

    doc = extract(corrupted_text)
    claim = next(c for c in doc.claims if "t" in c.slots and "df" in c.slots)
    assert claim.slots["p"].value == 0.0        # усечённый литерал '.00'
    res = check_slot(claim.slots["p"], corrupted_text, claim)
    assert not res.ok, "усечённое переносом число прошло охранника необнаруженным"
    assert res.reason == "truncated_literal"


def test_c5_no_false_rejection_on_real_corpus():
    """C5 — новый критерий, и у него есть названный остаточный риск
    (guard.py, комментарий у _LINE_ARTIFACT): 'p = .016\\n3 participants'
    даёт ложный отказ. Здесь — эмпирическая проверка того, что на всех
    43 настоящих (has_t_test) утверждениях corpus_real.py этот риск не
    реализуется: подтверждено, что после каждого span в чистом тексте
    корпуса стоит '.', ',', ')' или пробел с буквой — ни разу цифра
    через перенос — поэтому C5 не даёт ни одного НОВОГО отказа."""
    for rec in CORPUS_REAL:
        if not rec.get("has_t_test"):
            continue
        doc = extract(rec["text"])
        claim = next(c for c in doc.claims
                     if {"t", "df", "p"} <= set(c.slots))
        for r in check_claim(claim, rec["text"]):
            assert r.ok, f"{rec['id']} [{r.slot}] {r.reason}: {r.detail}"
