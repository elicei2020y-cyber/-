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

# Используется, только когда обычное окно (CANDIDATE_WINDOW) дало РОВНО
# одного кандидата, отделённого от t-образца переносом строки — то есть
# именно тогда, когда нужно решить, законный ли это перенос внутри
# предложения или вытекание колонки.
#
# Подобрано измерением (scripts/run_window_w_sweep.py), не назначено:
# порог, на котором доля опасных исходов на трёх синтетических видах
# склейки колонок падает до 0.0%, — W=105; 130 — конец плато, на
# котором доля утраченных легитимных извлечений ещё та же минимальная
# (2.3%, из-за одного не связанного предложения со своим p дальше по
# тексту в синтетической модели), а не наименьшее достаточное значение,
# ради запаса на случай более длинных, чем смоделированные, фрагментов
# склейки. ЧЕСТНОЕ ОГРАНИЧЕНИЕ: обе кривые измерены на синтетической
# порче, не на живом PDF — corpus_real.py получен из HTML и не содержит
# переносов строк вовсе, поэтому откалибровать это число по реальному
# тексту нечем.
WIDE_CANDIDATE_WINDOW = 130

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


def _p_candidates(source: str, end: int, window_size: int) -> List[Slot]:
    window = source[end:end + window_size]
    cands: List[Slot] = []
    for pm in P_ANY.finditer(window):
        s, e = pm.span("p")
        cands.append(
            Slot.from_match("p", source, (end + s, end + e), relation=_rel(pm.group("prel")))
        )
    return cands


def _claim_from_loose(source: str, m: re.Match) -> Claim:
    """Утверждение без прилегающего p: p ищется в окне и уходит в кандидаты.

    Слот p намеренно не разрешается принудительно, если кандидатов не
    ровно один. Отсечение кандидатов при неоднозначности — работа
    решателя (L5b), а не извлекателя: извлекатель, выбирающий один
    вариант из нескольких, — это и есть та точка, где системы вроде
    statcheck теряют точность.

    ИСТОРИЯ ПОЧИНКИ ЭТОЙ ФУНКЦИИ (три раунда, первые два были неполными
    или излишне дорогими — каждый следующий правил находку рецензента
    на предыдущем).

    Раунд 1 (span_end из найденного кандидата): C0 в guard.py
    ('кандидат внутри границ утверждения') была тождественно истинной
    по построению — граница подстраивалась под то же, что в неё
    проверялось.

    Раунд 2 (граница по next-t(...), ОТКЛОНЁН): ловил только ту форму
    склейки колонок, где вклинившийся фрагмент сам содержит t(...) —
    реалистичные виды (строка таблицы, обрывок предложения, подпись к
    рисунку) t( не содержат, дыра оставалась на 100%.

    Раунд 3 (запрет продвижения через перенос строки, ТОЖЕ ОТКЛОНЁН
    рецензентом после раунда 2): продвигал единственного кандидата,
    только если между t-образцом и им нет '\\n'. Ловил все реалистичные
    виды склейки, но ценой, которую раунд 2 не измерил: перенос строки
    внутри ОДНОГО предложения — обычное дело в PDF, а не признак чужой
    колонки, и признак этот их не различает. corpus_real.py тут не
    судья: он получен из HTML и не содержит переносов вовсе, поэтому
    "потерь на корпусе нет" ничего не доказывает — родная ловушка
    corpus.py на этот случай (id='line_break') раньше проходила, после
    раунда 3 стала AMBIGUOUS_PARSE, и это была настоящая, измеримая
    потеря полноты, а не только формальная.

    ДЕЙСТВУЮЩАЯ ПОЧИНКА (раунд 4): различает эти два случая не запретом,
    а РАСШИРЕНИЕМ ОКНА ПОИСКА. При вытекании колонки настоящий p всегда
    остаётся в тексте ДАЛЬШЕ, за вклинившимся фрагментом, — заглянуть
    туда достаточно, чтобы кандидатов стало ДВА, а два кандидата уже
    безопасно уходят в AMBIGUOUS_PARSE сами по себе, без специального
    признака. При обычном переносе внутри предложения второго кандидата
    там нет, и находится всё тот же один. Обычное окно (CANDIDATE_WINDOW)
    остаётся первым приближением всегда; расширение до WIDE_CANDIDATE_WINDOW
    включается, только если обычное окно уже дало РОВНО одного кандидата,
    отделённого от t-образца переносом строки, — то есть именно тогда,
    когда нужно решить, какой это случай. Величина WIDE_CANDIDATE_WINDOW
    подобрана измерением обеих кривых (доли опасных исходов на реалистичной
    склейке и доли утраченных извлечений на синтетических легитимных
    переносах) — scripts/run_window_w_sweep.py, там же честно про то,
    что обе кривые построены на синтетической порче, не на живом PDF:
    в corpus_real.py переносов нет вовсе, и по нему это W не выбрать.

    ОСТАТОЧНЫЙ РИСК, названный прямо: слишком широкое окно способно
    зацепить p из следующего, не связанного предложения — тогда
    легитимный случай тоже уйдёт в AMBIGUOUS_PARSE, но это не тихая
    подмена значения, а лишняя, безопасная абстенция. Вклинившийся
    фрагмент БЕЗ физического переноса строки перед собой по-прежнему
    неотличим от честного продолжения — это уже другой класс дефекта
    (потерянный перенос), для которого нет отдельного признака.
    """
    end = m.end()
    cands = _p_candidates(source, end, CANDIDATE_WINDOW)
    if len(cands) == 1 and "\n" in source[end:cands[0].span[0]]:
        cands = _p_candidates(source, end, WIDE_CANDIDATE_WINDOW)
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
