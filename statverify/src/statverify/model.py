"""Типы данных слоёв L0-L1.

Ключевое свойство Slot: value и literal хранятся раздельно.
value  — разобранное число, которым будет пользоваться решатель.
literal — точная подстрока исходника.
Охранник (L2) проверяет, что одно действительно порождает другое,
и что literal действительно лежит по указанному span.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

Span = Tuple[int, int]

# Минусы, встречающиеся в PDF-извлечении: HYPHEN-MINUS, U+2212, EN DASH, U+2010
_MINUSES = "-−–‐"


def parse_number(literal: str) -> Optional[float]:
    """Разобрать число ровно так, как оно записано в статье.

    Допускает APA-форму без ведущего нуля ('.016') и юникодные минусы.
    Возвращает None, если строка не является числом, — охранник трактует
    это как отказ, а не как исключение.
    """
    s = literal.strip()
    if not s:
        return None
    sign = 1.0
    if s[0] in _MINUSES:
        sign, s = -1.0, s[1:]
    elif s[0] == "+":
        s = s[1:]
    if s.startswith("."):
        s = "0" + s
    try:
        return sign * float(s)
    except ValueError:
        return None


def decimals_of(literal: str) -> int:
    """Число знаков после запятой в том виде, как записано.

    Питает интервальную арифметику L4: '.016' -> 3 знака -> [.0155, .0165).
    """
    s = literal.strip().lstrip("+" + _MINUSES)
    return len(s.split(".", 1)[1]) if "." in s else 0


@dataclass(frozen=True)
class Slot:
    name: str            # "t" | "df" | "p" | "n"
    value: float
    literal: str
    span: Span
    decimals: int
    relation: str = "="  # "=" | "<" | ">" | "<=" | ">="

    @staticmethod
    def from_match(name: str, source: str, span: Span, relation: str = "=") -> "Slot":
        literal = source[span[0]:span[1]]
        value = parse_number(literal)
        if value is None:
            raise ValueError(f"нечисловой литерал для слота {name}: {literal!r}")
        return Slot(name, value, literal, span, decimals_of(literal), relation)


@dataclass
class Claim:
    """Одно самодостаточное статистическое утверждение (L0).

    span задаёт границу, за которую слоты не имеют права выходить.
    Это то, что не даёт взять df из соседнего теста.
    """
    span: Span
    design: str
    slots: Dict[str, Slot] = field(default_factory=dict)
    candidates: Dict[str, List[Slot]] = field(default_factory=dict)

    def with_slot(self, slot: Slot) -> "Claim":
        return replace(self, slots={**self.slots, slot.name: slot})


@dataclass
class Document:
    source: str
    claims: List[Claim] = field(default_factory=list)
