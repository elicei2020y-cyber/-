"""Порча текста, имитирующая артефакты извлечения из PDF.

В отличие от corrupt.py (который портит уже извлечённый Claim, чтобы
измерить чувствительность охранника к конкретному критерию C0-C4), здесь
портится ИСХОДНЫЙ ТЕКСТ до извлечения — так, как его мог бы отдать
плохой конвертер PDF->текст. Цель другая: не "поймает ли охранник
подставленное значение", а "не пропустит ли весь конвейер (L1+L2)
НЕВЕРНОЕ значение, которое сам же и придумал из испорченного текста,
но которое выглядит для охранника совершенно законным" — то есть
единственный опасный исход из всех возможных.

Каждая функция принимает (text, gt) — исходный текст и Claim,
извлечённый из НЕИСПОРЧЕННОГО текста (источник истинных литералов и
их смещений) — и возвращает испорченный текст. Порча не знает про
охранник и не подстраивается под него: она имитирует физический
артефакт, а не атаку на конкретный критерий.
"""

import re
from typing import Callable, Dict, Optional, Tuple

from statverify.extract import extract
from statverify.guard import admit
from statverify.model import Claim

Corruption = Callable[[str, Claim], Optional[str]]


def ligatures(text: str, gt: Claim) -> Optional[str]:
    """Лигатуры fi/fl, типичные для шрифтового вывода PDF.

    Затрагивает только буквы в словах ('significant' -> 'siɡnifi‌cant'
    через U+FB01), числа не трогает вовсе — это заведомо инертная для
    извлечения чисел порча, включена для полноты картины.
    """
    if "fi" not in text and "fl" not in text:
        return None
    return text.replace("fi", "ﬁ").replace("fl", "ﬂ")


def soft_hyphen(text: str, gt: Claim) -> Optional[str]:
    """Мягкий перенос (U+00AD) внутри длинных слов — след обёртки строк
    в PDF, которую конвертер не убрал. Числа не трогает: перенос внутри
    буквенного слова, не внутри цифр (это отдельная порча ниже)."""
    def _break(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 6:
            return w
        cut = len(w) // 2
        return w[:cut] + "­" + w[cut:]
    new = re.sub(r"[A-Za-z]{6,}", _break, text, count=3)
    return new if new != text else None


def linebreak_in_number(text: str, gt: Claim) -> Optional[str]:
    """Разрыв строки внутри числа: колонка PDF обрывается ровно на цифре
    ('2.4\\n5'). Бьём literal p (у него обычно больше знаков после
    запятой, чем у t), если в нём есть хотя бы одна цифра после точки;
    иначе — literal t."""
    for name in ("p", "t"):
        s = gt.slots.get(name)
        if s is None:
            continue
        lit = s.literal
        digits_after_dot = len(lit.split(".", 1)[1]) if "." in lit else 0
        if digits_after_dot >= 1:
            cut = len(lit) - 1  # разрыв перед последней цифрой
            broken = lit[:cut] + "\n" + lit[cut:]
            lo, hi = s.span
            return text[:lo] + broken + text[hi:]
    return None


def lost_space(text: str, gt: Claim) -> Optional[str]:
    """Потерянный пробел перед маркером p: 't(29)=2.45p=.016'.

    Убирает пробел(ы) между концом literal t (или предшествующей
    запятой/скобкой) и маркером p — конвертер PDF из двух колонок
    нередко теряет межсловные пробелы именно на границе токенов."""
    p = gt.slots.get("p")
    if p is None:
        return None
    lo = p.span[0]
    before = text[:lo]
    # маркер 'p' (или 'P') с разделителем перед числом — ищем его САМЫЙ
    # ПРАВЫЙ край, вплотную перед p.value
    marker_m = re.search(r"[pP]\s*[<>=≤≥]{1,2}\s*$", before)
    if marker_m is None:
        return None
    ws_m = re.search(r"\s+$", before[:marker_m.start()])
    if ws_m is None:
        return None
    new_before = before[:ws_m.start()] + before[ws_m.end():marker_m.start()] + before[marker_m.start():]
    return new_before + text[lo:]


def footnote_glued(text: str, gt: Claim) -> Optional[str]:
    """Сноска, приклеенная к цифре без пробела: 'p = .002¹'.

    Надстрочный символ приклеен СРАЗУ после literal p, имитируя
    footnote marker, слитый со значением при выгрузке текста."""
    p = gt.slots.get("p")
    if p is None:
        return None
    hi = p.span[1]
    return text[:hi] + "¹" + text[hi:]


# Три реалистичных вида вытекания колонки — ни один не содержит 't(':
# настоящая склейка колонок происходит между СОВЕРШЕННО другими
# фрагментами документа, обнаружение общего 't(' в них было бы
# совпадением, а не правилом. Все три несут собственное 'p = ...',
# иначе вклинивание не создало бы ложного кандидата вовсе.
COLUMN_SPLICE_KINDS: Dict[str, str] = {
    "table_row": ("M = 2.10, SD = 0.45, p = {p}, N = 24 per condition, "
                  "age and sex controlled for as covariates"),
    "sentence_fragment": ("in line with prior reports, p = {p}, effects "
                          "varied across the three replication sites tested"),
    "figure_caption": ("Fig. 2. Error bars denote SEM, p = {p}, asterisks "
                       "mark significant pairwise comparisons"),
}


def column_splicing(text: str, gt: Claim, kind: str, foreign_p: str = ".731") -> Optional[str]:
    """Склейка колонок: между t(...) = ... и его собственным p = ...
    вклинивается фрагмент соседней колонки/таблицы/подписи.

    ВТОРАЯ ВЕРСИЯ этой функции, ПЕРВАЯ была неправдоподобной и не
    воспроизводила артефакт: не вставляла перенос строки вовсе (в
    извлечённом из PDF тексте вытекание колонки СВОИМ переносом строки
    приходит всегда — иначе это не вытекание колонки, а просто соседний
    текст на той же строке) и вклинившийся фрагмент всегда содержал
    собственный 't(', из-за чего 'обрубить окно поиска на next-t('
    выглядело как починка, а на деле ловило только ту форму склейки,
    которую сама порча гарантированно создавала. Ревью, смоделировавшее
    реальные варианты (строка таблицы, обрывок предложения, подпись к
    рисунку — ни один не содержит 't('), нашло, что дыра оставалась
    открыта на 86% несмотря на 0% в отчёте по старой порче.

    Три вида (kind), ни один не содержит 't(' — настоящая склейка
    колонок бывает именно между такими разнородными фрагментами:
      table_row         — строка таблицы со своим столбцом p
      sentence_fragment — обрывок соседнего предложения с p-значением
      figure_caption    — подпись к рисунку с указанием значимости
    """
    fragment = COLUMN_SPLICE_KINDS[kind].format(p=foreign_p)
    assert "t(" not in fragment, f"порча {kind!r} не должна содержать t("
    t = gt.slots.get("t")
    p = gt.slots.get("p")
    if t is None or p is None:
        return None
    cut = t.span[1]
    if cut >= p.span[0]:
        return None
    junk = "\n" + fragment + "\n"
    return text[:cut] + junk + text[cut:]


def legit_linebreak(text: str, gt: Claim) -> Optional[str]:
    """Легитимный перенос строки ВНУТРИ ОДНОГО предложения — обычное
    дело в PDF (строки рвутся постоянно, это не признак чужой колонки).
    Перенос вставлен на месте пробела перед маркером p — тот же узор,
    что tests/corpus.py::line_break. Не порча в смысле опасности: это
    ДОЛЖНО продолжать извлекаться верно, используется для измерения
    ПОТЕРЬ (не находок), а не находок."""
    p = gt.slots.get("p")
    if p is None:
        return None
    lo = p.span[0]
    before = text[:lo]
    marker_m = re.search(r"[pP]\s*[<>=≤≥]{1,2}\s*$", before)
    if marker_m is None:
        return None
    ws_m = re.search(r"\s+$", before[:marker_m.start()])
    if ws_m is None:
        return None
    new_before = before[:ws_m.start()] + "\n" + before[ws_m.end():]
    return new_before + text[lo:]


def legit_linebreak_with_trailing_p(text: str, gt: Claim,
                                     trailing_p: str = ".500",
                                     pad_chars: int = 0) -> Optional[str]:
    """То же самое, плюс СЛЕДУЮЩЕЕ, не связанное предложение со своим
    p дальше по тексту — модель именно того риска, ради которого
    подбирается WIDE_CANDIDATE_WINDOW измерением, а не назначением:
    окно, достаточно широкое, чтобы поймать настоящую склейку колонок,
    способно зацепить и честное продолжение той же статьи.

    pad_chars — необязательная прокладка ПЕРЕД дополнительным
    предложением: расстояние до следующего p в реальном тексте не
    фиксировано, и без разброса по расстоянию вся кривая потерь была бы
    единственной точкой обрыва (все 43 записи схлопываются в один и тот
    же зазор) вместо кривой, показывающей поведение при разных
    расстояниях."""
    base = legit_linebreak(text, gt)
    if base is None:
        return None
    padding = ("also " * (pad_chars // 5))[:pad_chars]
    trailer = (f" Additional {padding}exploratory analyses examined "
               f"secondary outcomes without correction, p = {trailing_p}.")
    return base + trailer


def ocr_l1_o0(text: str, gt: Claim) -> Optional[str]:
    """OCR-подмены: строчная 'l' распознана как цифра '1', заглавная
    'O' — как '0'. Классическая ошибка OCR-слоя, а не PDF-текстового
    слоя, но даёт тот же класс риска: буква превращается в цифру рядом
    с настоящими числами."""
    if "l" not in text and "O" not in text:
        return None
    return text.replace("l", "1").replace("O", "0")


CORRUPTIONS: Dict[str, Corruption] = {
    "ligatures": ligatures,
    "soft_hyphen": soft_hyphen,
    "linebreak_in_number": linebreak_in_number,
    "lost_space": lost_space,
    "footnote_glued": footnote_glued,
    # column_splicing взят отдельно в раннере: ему нужен foreign-фрагмент
    "ocr_l1_o0": ocr_l1_o0,
}


def ground_truth(text: str) -> Optional[Claim]:
    """Claim, извлечённый из НЕИСПОРЧЕННОГО текста — источник истинных
    литералов/смещений для прицельной порчи и истинных значений для
    сверки после неё."""
    doc = extract(text)
    for c in doc.claims:
        if {"t", "df", "p"} <= set(c.slots):
            return c
    return None


def resolve_after_corruption(
    text: str, true_t: float, true_df: float
) -> Tuple[Optional[Claim], bool, Optional[Dict[str, float]]]:
    """Извлечь t/df/p из (возможно испорченного) текста заново, прогнать
    через охранник, вернуть (claim_или_None, admitted, values_или_None).

    Ищем claim по СОВПАДЕНИЮ (t, df) с истинными значениями исходного
    утверждения, а не первый попавшийся с полной тройкой слотов: у
    вклинившегося (column_splicing) фрагмента может обнаружиться СВОЙ,
    отдельный claim с собственной, верной для НЕГО тройкой — взять
    первый без разбора значило бы сверять истину одного утверждения со
    значениями другого.

    values — только если t/df/p сидят как конкретные, однозначно
    разрешённые слоты (не кандидаты): если p ушёл в список кандидатов
    или пропал вовсе, это честная неоднозначность/абстенция (L5b), а
    не 'извлекатель ошибся молча' — такие случаи считаются 'не
    найдено', ровно по логике guard.py (правило 6 CLAUDE.md)."""
    doc = extract(text)
    own = [c for c in doc.claims
           if "t" in c.slots and "df" in c.slots
           and abs(c.slots["t"].value - true_t) < 1e-9
           and abs(c.slots["df"].value - true_df) < 1e-9]
    if not own:
        return None, False, None
    c = own[0]
    if "p" not in c.slots:
        return c, False, None
    ok, _results = admit(c, text)
    values = {n: c.slots[n].value for n in ("t", "df", "p")}
    return c, ok, values
