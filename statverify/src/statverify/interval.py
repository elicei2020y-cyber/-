"""Интервальная арифметика L4.

Ни одно сообщённое число не является точкой. 'p = .016' означает, что
истинное значение лежит в [.0155, .0165) — округление до трёх знаков.
Сравнение на равенство здесь порождало бы ложные срабатывания на
каждой второй статье, поэтому весь слой работает интервалами,
а решение о противоречии принимается через пустоту пересечения.

Открытые концы моделируются сжатием на 1e-12. Это на восемь порядков
меньше самой мелкой встречающейся точности записи (пять знаков после
запятой даёт полуширину 5e-6), поэтому подмена открытого конца
закрытым не может изменить ни один вердикт.

СОГЛАШЕНИЕ ОБ ОКРУГЛЕНИИ. Принято округление к ближайшему: '.016'
даёт [.0155, .0165). Часть журналов усекает, и тогда истинный
интервал был бы [.016, .017). Усечение здесь не поддерживается
сознательно: округление к ближайшему даёт интервал, накрывающий
усечённый лишь наполовину, поэтому при усекающем журнале система
даст ложное срабатывание. Это зафиксировано как известный режим
отказа, а не устранено, потому что определять соглашение журнала
должен слой классификации, которого пока нет.
"""

from dataclasses import dataclass
from typing import Optional

TINY = 1e-12


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"пустой интервал [{self.lo}, {self.hi}]")

    def contains(self, v: float) -> bool:
        return self.lo <= v <= self.hi

    def intersects(self, other: "Interval") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    def meet(self, other: "Interval") -> Optional["Interval"]:
        lo, hi = max(self.lo, other.lo), min(self.hi, other.hi)
        return Interval(lo, hi) if lo <= hi else None

    def abs(self) -> "Interval":
        if self.lo <= 0.0 <= self.hi:
            return Interval(0.0, max(abs(self.lo), abs(self.hi)))
        a, b = abs(self.lo), abs(self.hi)
        return Interval(min(a, b), max(a, b))

    @property
    def corners(self):
        return (self.lo, self.hi) if self.lo != self.hi else (self.lo,)

    def contains_integer(self) -> bool:
        import math
        return math.floor(self.hi) >= math.ceil(self.lo)

    def __repr__(self):
        return f"[{self.lo:.6g}, {self.hi:.6g}]"


def from_reported(value: float, decimals: int, relation: str = "=",
                  domain: Optional[Interval] = None) -> Interval:
    """Интервал, порождённый точностью записи и знаком отношения."""
    if relation == "=":
        half = 0.5 * (10.0 ** -decimals)
        iv = Interval(value - half, value + half - TINY)
    elif relation == "<":
        lo = domain.lo if domain else -1e300
        iv = Interval(lo + TINY, value - TINY)
    elif relation == "<=":
        lo = domain.lo if domain else -1e300
        iv = Interval(lo + TINY, value)
    elif relation == ">":
        hi = domain.hi if domain else 1e300
        iv = Interval(value + TINY, hi - TINY)
    elif relation == ">=":
        hi = domain.hi if domain else 1e300
        iv = Interval(value, hi - TINY)
    else:
        raise ValueError(f"неизвестное отношение {relation!r}")
    return iv.meet(domain) or iv if domain else iv


# Области определения переменных — жёсткие границы (класс Bound из L3).
DOMAIN = {
    "p": Interval(TINY, 1.0 - TINY),
    "df": Interval(TINY, 1e9),
    "t": Interval(-1e6, 1e6),
    "n": Interval(1.0, 1e9),
}
