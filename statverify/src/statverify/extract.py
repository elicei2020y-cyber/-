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

# Разделитель по обе стороны _GAP — НЕ \s: \s включает перенос строки,
# и раньше именно через него TIGHT (а не через сам _GAP, у которого
# своя защита от '.' и вложенного t() уже есть) тихо пересекал
# физический разрыв строки — ровно тот артефакт, которым и является
# склейка колонок. _GAP не спасал: разрыв поглощался ДО него, этим же
# разделителем, который его окружает. Пробел и таб — нет, они никогда
# не разрывают одно утверждение на две физические строки.
_SEP = r"[ \t,;]*"

TIGHT = re.compile(
    rf"""
    \bt\s*
    \(\s* (?:df\s*[=:]\s*)? (?P<df>{_NUM_PLAIN}) \s*\)
    \s* [=:] \s*
    (?P<t>{_NUM_SIGNED})
    {_SEP} {_GAP} [ \t,;(]*
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

# Задача 1 (docs/TASK_FOR_CLAUDE_CODE.md, §2): b/β/M + SE/SD и доверительный
# интервал — величины для R5 и R4. Как и p, они ищутся в окне ПОСЛЕ уже
# найденного t()=... и присоединяются к тому же утверждению, а не образуют
# новое: это те же семь взаимно согласованных чисел одного теста.
#
# Известное ограничение, честно, а не молча: окно ищет только вперёд по
# тексту. Порядок 'b = .45, SE = .12, t(98) = 3.75, p = .001' (оценка
# перед t-тестом) этим не покрывается — расширение до двустороннего окна
# осталось бы за задачей 3 (модельный извлекатель), где решение о
# границах утверждения в любом случае должно стать надёжнее регекса.
EST_LABEL = r"[bβM]"
SE_LABEL = r"SE|SD"

EST_SE = re.compile(
    rf"""
    \b(?P<estlabel>{EST_LABEL})\s*[=:]\s*(?P<est>{_NUM_SIGNED})
    \s*[,;(]*\s*
    \b(?P<selabel>{SE_LABEL})\s*[=:]\s*(?P<se>{_NUM_SIGNED})
    \)?
    """,
    re.VERBOSE,
)

CI_BRACKETS = re.compile(
    rf"""
    (?:(?P<pct>\d{{1,2}})\s*%\s*)?
    CI\s*(?:[=:]\s*)?
    \[\s*(?P<lo>{_NUM_SIGNED})\s*,\s*(?P<hi>{_NUM_SIGNED})\s*\]
    """,
    re.VERBOSE,
)

CI_TO = re.compile(
    rf"""
    (?:(?P<pct>\d{{1,2}})\s*%\s*)?
    CI\s*[=:]?\s*
    (?P<lo>{_NUM_SIGNED})\s+to\s+(?P<hi>{_NUM_SIGNED})
    """,
    re.VERBOSE,
)

ESTCI_WINDOW = 90


def _attach_est_se(source: str, claim: Claim) -> Claim:
    end = claim.span[1]
    window = source[end:end + ESTCI_WINDOW]
    m = EST_SE.search(window)
    if m is None:
        return claim
    claim = claim.with_slot(Slot.from_match("est", source, (end + m.start("est"), end + m.end("est"))))
    claim = claim.with_slot(Slot.from_match("se", source, (end + m.start("se"), end + m.end("se"))))
    new_end = max(claim.span[1], end + m.end())
    claim.span = (claim.span[0], new_end)
    return claim


def _attach_ci(source: str, claim: Claim) -> Claim:
    end = claim.span[1]
    window = source[end:end + ESTCI_WINDOW]
    m = CI_BRACKETS.search(window) or CI_TO.search(window)
    if m is None:
        return claim
    claim = claim.with_slot(Slot.from_match("ci_lo", source, (end + m.start("lo"), end + m.end("lo"))))
    claim = claim.with_slot(Slot.from_match("ci_hi", source, (end + m.start("hi"), end + m.end("hi"))))
    pct = m.group("pct")
    if pct is not None:
        claim.ci_alpha = round(1.0 - int(pct) / 100.0, 6)
        claim.ci_alpha_explicit = True
    new_end = max(claim.span[1], end + m.end())
    claim.span = (claim.span[0], new_end)
    return claim


def _attach_est_ci(source: str, claim: Claim) -> Claim:
    claim = _attach_est_se(source, claim)
    claim = _attach_ci(source, claim)
    return claim


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

    Слот p намеренно не разрешается принудительно, КРОМЕ единственного
    кандидата без переноса строки между ним и t-образцом (см. ниже).
    Отсечение кандидатов при остальной неоднозначности — работа
    решателя (L5b), а не извлекателя: извлекатель, выбирающий один
    вариант из нескольких, — это и есть та точка, где системы вроде
    statcheck теряют точность.

    ИСТОРИЯ ПОЧИНКИ ЭТОЙ ФУНКЦИИ (два раунда, первый был неполным).
    Раунд 1 (span_end из найденного кандидата): span_end брался как
    max() позиций НАЙДЕННЫХ кандидатов, и C0 в guard.py ('кандидат
    внутри границ утверждения') была тождественно истинной по
    построению — граница подстраивалась под то же, что в неё
    проверялось. Раунд 2 (граница по next-t(...), ЭТОТ ПОДХОД
    ОТКЛОНЁН): обрубал окно поиска на начале следующего t(...) —
    ловил только ту форму склейки колонок, где вклинившийся фрагмент
    сам содержит t(...). Смоделированная проверка (три реалистичных
    вида вклинивания — строка таблицы, обрывок предложения, подпись к
    рисунку, НИ ОДИН не содержит t() показала: 100% случаев остаются
    опасными что до этой 'починки', что после — она не защищала вовсе,
    несмотря на 0% в отчёте по нерепрезентативной порче.

    ДЕЙСТВУЮЩАЯ ПОЧИНКА: единственный найденный кандидат продвигается
    в слот, ТОЛЬКО ЕСЛИ между концом t-образца и началом кандидата НЕТ
    переноса строки. Настоящая склейка колонок (вытекание соседней
    колонки/таблицы/подписи) приходит СВОИМ переносом строки ВСЕГДА —
    во извлечённом из PDF тексте иначе вклинившийся фрагмент был бы не
    вытеканием колонки, а просто соседним текстом на той же строке.
    Законное 't(99) = 2.45. This was reliable, p = .016.' переноса не
    содержит и продвигается как прежде — полнота на чистом тексте не
    теряется. Признак отличает эти два случая по СУЩЕСТВУ (наличие
    физического разрыва строки), а не по совпадающему с порчей узору
    вроде t(, поэтому и держит все три реалистичных вида вклинивания:
    0% опасных (было 100% и до, и после раунда 2) — при отсутствии
    единственного кандидата случай уходит в claim.candidates —
    AMBIGUOUS_PARSE, отказ, а не тихо неверное значение.

    ОСТАТОЧНЫЙ РИСК, названный прямо: вклинившийся фрагмент БЕЗ
    переноса строки перед собой (то есть текст, приклеенный к концу
    предыдущей строки без физического обрыва) неотличим от честного
    продолжения предложения — но это уже не 'склейка колонок' в
    физическом смысле, а другой класс дефекта (потерянный перенос
    строки), для которого нет отдельного признака.
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
    if len(cands) == 1 and "\n" not in source[end:cands[0].span[0]]:
        claim = claim.with_slot(cands[0])
    elif cands:
        claim.candidates["p"] = cands
    return claim


def extract(source: str) -> Document:
    doc = Document(source=source)
    taken: List[range] = []

    for m in TIGHT.finditer(source):
        claim = _attach_est_ci(source, _claim_from_tight(source, m))
        doc.claims.append(claim)
        taken.append(range(*m.span()))

    for m in LOOSE.finditer(source):
        if any(m.start() in r for r in taken):
            continue
        claim = _attach_est_ci(source, _claim_from_loose(source, m))
        doc.claims.append(claim)

    doc.claims.sort(key=lambda c: c.span[0])
    return doc
