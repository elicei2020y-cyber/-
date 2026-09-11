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


def column_splicing(text: str, gt: Claim, foreign: str) -> Optional[str]:
    """Склейка колонок: между t(...) = ... и p = ... вклинивается
    хвост ДРУГОГО предложения из соседней колонки — реальный артефакт
    построчного считывания двухколоночной вёрстки поверх текстового
    потока. foreign — фрагмент из НЕСВЯЗАННОГО утверждения корпуса."""
    t = gt.slots.get("t")
    p = gt.slots.get("p")
    if t is None or p is None:
        return None
    cut = t.span[1]
    if cut >= p.span[0]:
        return None
    junk = " " + foreign.strip()[:70] + " "
    return text[:cut] + junk + text[cut:]


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
