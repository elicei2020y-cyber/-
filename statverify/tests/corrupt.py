"""Искусственная порча извлечения — тест чувствительности охранника.

Охранник, который всё пропускает, бесполезен, поэтому мерить его надо
в обе стороны: доля ложных отказов на честном извлечении и доля
пойманных на порче. Каждая порча воспроизводит конкретную ошибку,
которую делает модельный извлекатель.

Порчи 5-7 сделаны намеренно «правдоподобными»: литерал в них
действительно лежит по указанному смещению, поэтому C1 и C2 их
пропускают и ловить обязаны C0 (границы) и C3 (роль). Это и есть
настоящая проверка охранника.
"""

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

from statverify.model import Claim, Document, Slot

Corruption = Callable[[Claim, Document], Optional[Tuple[Claim, str]]]


def _swap(claim: Claim, slot: Slot) -> Claim:
    return replace(claim, slots={**claim.slots, slot.name: slot})


def fabricated_value(claim: Claim, doc: Document):
    """Модель приписала слоту число, которого в тексте нет. Ждём C2."""
    for name in ("p", "t", "df"):
        if name in claim.slots:
            s = claim.slots[name]
            return _swap(claim, replace(s, value=s.value + 1.0)), name
    return None


def fabricated_literal(claim: Claim, doc: Document):
    """Модель сообщила литерал, не совпадающий с исходником. Ждём C1."""
    for name in ("p", "t", "df"):
        if name in claim.slots:
            s = claim.slots[name]
            return _swap(claim, replace(s, literal=s.literal + "9")), name
    return None


def span_drift(claim: Claim, doc: Document):
    """Смещение уехало на пару символов — типично при сдвиге офсетов. Ждём C1.

    Слоты перебираются от середины утверждения к краю: сдвиг у крайнего
    слота вылетел бы за границы и был бы пойман C0, а проверить надо
    именно C1.
    """
    for name in ("df", "t", "p"):
        if name in claim.slots:
            s = claim.slots[name]
            lo, hi = s.span
            if hi + 2 <= claim.span[1]:
                return _swap(claim, replace(s, span=(lo + 2, hi + 2))), name
    return None


def relation_flip(claim: Claim, doc: Document):
    """'p < .001' записано как 'p = .001'. Ждём C4.

    Для L4 это не мелочь: интервалы (0, .001) и [.0005, .0015) не пересекаются.
    """
    if "p" not in claim.slots:
        return None
    s = claim.slots["p"]
    flipped = "<" if s.relation == "=" else "="
    return _swap(claim, replace(s, relation=flipped)), "p"


def role_confusion(claim: Claim, doc: Document):
    """Значение t записано в слот p. Литерал настоящий, лежит по месту.

    C1 и C2 проходят. Ловить обязан C3 — маркер роли перед числом.
    """
    if "p" not in claim.slots or "t" not in claim.slots:
        return None
    t = claim.slots["t"]
    fake_p = Slot("p", t.value, t.literal, t.span, t.decimals, "=")
    return _swap(claim, fake_p), "p"


def cross_claim(claim: Claim, doc: Document):
    """Слот взят из соседнего теста — структурно валидного.

    Ни C1, ни C2, ни C3 его не поймают: там настоящий df в настоящих
    скобках после настоящего t. Ловить обязан C0 — границы утверждения.
    Ради этого L0 и существует.
    """
    others = [c for c in doc.claims if c.span != claim.span and "df" in c.slots]
    if not others or "df" not in claim.slots:
        return None
    return _swap(claim, others[0].slots["df"]), "df"


def _occurrences(doc: Document, literal: str):
    out, start = [], 0
    while True:
        idx = doc.source.find(literal, start)
        if idx == -1:
            return out
        out.append((idx, idx + len(literal)))
        start = idx + 1


def _distractor(claim: Claim, doc: Document, inside: bool):
    if "df" not in claim.slots:
        return None
    s = claim.slots["df"]
    for span in _occurrences(doc, s.literal):
        if span == s.span:
            continue
        within = claim.span[0] <= span[0] and span[1] <= claim.span[1]
        if within == inside:
            return _swap(claim, replace(s, span=span)), "df"
    return None


def distractor_outside(claim: Claim, doc: Document):
    """df показывает на такое же число вне границ утверждения.

    Литерал и значение совпадают — C1 и C2 слепы, маркер роли может
    случайно совпасть. Ловить обязан C0.
    """
    return _distractor(claim, doc, inside=False)


def distractor_inside(claim: Claim, doc: Document):
    """df показывает на такие же цифры внутри утверждения — например,
    на '45' в составе '2.45'. C0 бессилен, ловить обязан C3.
    """
    return _distractor(claim, doc, inside=True)


CORRUPTIONS: Dict[str, Corruption] = {
    "fabricated_value": fabricated_value,
    "fabricated_literal": fabricated_literal,
    "span_drift": span_drift,
    "relation_flip": relation_flip,
    "role_confusion": role_confusion,
    "cross_claim": cross_claim,
    "distractor_outside": distractor_outside,
    "distractor_inside": distractor_inside,
}

# Какой критерий обязан поймать каждую порчу — если ловит другой,
# это не победа, а признак того, что критерии перекрываются не так,
# как задумано.
EXPECTED_CATCHER: Dict[str, str] = {
    "fabricated_value": "value_mismatch",
    "fabricated_literal": "span_mismatch",
    "span_drift": "span_mismatch",
    "relation_flip": "relation_mismatch",
    "role_confusion": "role_missing",
    "cross_claim": "not_contained",
    "distractor_outside": "not_contained",
    "distractor_inside": "role_missing",
}
