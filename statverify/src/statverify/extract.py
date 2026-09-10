"""L0 — сегментация на утверждения. L1 — типизированное извлечение.

Извлекатель здесь регексповый, а не модельный, и это осознанно:
на шаге 1 измеряется охранник, а регекс даёт заведомо честные span
(позиции групп совпадения), то есть эталон, относительно которого
видно, что именно охранник ловит, а что пропускает по своей природе.

Модельный извлекатель встанет на это же место позже, реализуя
тот же контракт: вернуть Claim со слотами, у каждого — span.

Шаг 1 покрывает тройку (t, df, p). Слот n отложен до шага 2:
он нужен только для проверки отношения df = n - 1, а на шаге 1
никакие отношения не проверяются.
"""

import re
from typing import List, Optional

from .model import Claim, Document, Slot

_NUM_SIGNED = r"[-−–‐+]?\d*\.?\d+"
_NUM_PLAIN = r"\d+(?:\.\d+)?"
_P_NUM = r"\d*\.\d+|\d+"
_REL = r"[<>=≤≥]{1,2}"

# Разрыв между t-значением и p: ограниченный по длине, без выхода за
# границу предложения и без вложенного второго t-теста внутри.
_GAP = r"(?:(?!t\s*\()[^.\n]){0,40}?"

TIGHT = re.compile(
    rf"""
    \bt\s*
    \(\s* (?:df\s*[=:]\s*)? (?P<df>{_NUM_PLAIN}) \s*\)
    \s* [=:] \s*
    (?P<t>{_NUM_SIGNED})
    [\s,;]* {_GAP} [\s,;(]*
    \b(?P<pmark>[pP])\s*
    (?P<prel>{_REL})\s*
    (?P<p>{_P_NUM})
    """,
    re.VERBOSE,
)

# t-тест без прилегающего p — источник кандидатов.
LOOSE = re.compile(
    rf"""
    \bt\s*
    \(\s* (?:df\s*[=:]\s*)? (?P<df>{_NUM_PLAIN}) \s*\)
    \s* [=:] \s*
    (?P<t>{_NUM_SIGNED})
    """,
    re.VERBOSE,
)

P_ANY = re.compile(rf"\b(?P<pmark>[pP])\s*(?P<prel>{_REL})\s*(?P<p>{_P_NUM})")

_REL_CANON = {"=": "=", "==": "=", "<": "<", ">": ">", "≤": "<=", "≥": ">=",
              "<=": "<=", ">=": ">="}

CANDIDATE_WINDOW = 60


def _rel(raw: str) -> str:
    return _REL_CANON.get(raw, raw)


def _claim_from_tight(source: str, m: re.Match) -> Claim:
    claim = Claim(span=m.span(), design="two_tailed_t")
    for name in ("df", "t"):
        claim = claim.with_slot(Slot.from_match(name, source, m.span(name)))
    claim = claim.with_slot(
        Slot.from_match("p", source, m.span("p"), relation=_rel(m.group("prel")))
    )
    return claim


def _claim_from_loose(source: str, m: re.Match) -> Claim:
    """Утверждение без прилегающего p: p ищется в окне и уходит в кандидаты.

    Слот p намеренно не разрешается принудительно. Отсечение кандидатов —
    работа решателя (L5b), а не извлекателя: извлекатель, выбирающий
    один вариант из нескольких, — это и есть та точка, где системы
    вроде statcheck теряют точность.
    """
    end = m.end()
    window = source[end:end + CANDIDATE_WINDOW]
    cands: List[Slot] = []
    for pm in P_ANY.finditer(window):
        s, e = pm.span("p")
        cands.append(
            Slot.from_match("p", source, (end + s, end + e), relation=_rel(pm.group("prel")))
        )
    span_end = max([c.span[1] for c in cands], default=m.end())
    claim = Claim(span=(m.start(), span_end), design="two_tailed_t")
    for name in ("df", "t"):
        claim = claim.with_slot(Slot.from_match(name, source, m.span(name)))
    if len(cands) == 1:
        claim = claim.with_slot(cands[0])
    elif cands:
        claim.candidates["p"] = cands
    return claim


def extract(source: str) -> Document:
    doc = Document(source=source)
    taken: List[range] = []

    for m in TIGHT.finditer(source):
        doc.claims.append(_claim_from_tight(source, m))
        taken.append(range(*m.span()))

    for m in LOOSE.finditer(source):
        if any(m.start() in r for r in taken):
            continue
        doc.claims.append(_claim_from_loose(source, m))

    doc.claims.sort(key=lambda c: c.span[0])
    return doc
