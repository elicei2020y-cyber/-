"""Порождение размеченных случаев арифметикой, а не рукой.

Истинный p вычисляется из (t, df) точно, затем записывается либо
корректно округлённым, либо смещённым. Поэтому разметка не является
чьим-либо суждением о том, что правильно.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple

from statverify.model import Claim, Slot, decimals_of
from statverify.tstats import p_two_tailed

DF_GRID = [10, 24, 49, 99, 250]
T_GRID = [1.5, 2.0, 2.45, 3.0, 4.2]


def round_half_up(x: float, d: int) -> float:
    """Округление к ближайшему — то же соглашение, что в interval.py.

    Встроенный round() использует банковское округление и на границе
    дал бы разметку, не совпадающую с интервалами, то есть искусственные
    расхождения. Это реальная ловушка, а не теоретическая.
    """
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d),
                                           rounding=ROUND_HALF_UP))


def slot(name: str, value: float, literal: str, rel: str = "=") -> Slot:
    return Slot(name, value, literal, (0, 1), decimals_of(literal), rel)


def make_claim(t: float, df: float, p: float, p_dec: int,
               t_dec: int = 2, prel: str = "=") -> Claim:
    return Claim(span=(0, 1), design="two_tailed_t", slots={
        "t": slot("t", t, f"{t:.{t_dec}f}"),
        "df": slot("df", df, f"{df:g}"),
        "p": slot("p", p, f"{p:.{p_dec}f}", prel),
    })


Case = Tuple[float, float, float, int, float]


def build_cases() -> Tuple[List[Case], List[Case]]:
    """(согласованные, несогласованные) с точной разметкой."""
    consistent: List[Case] = []
    inconsistent: List[Case] = []
    for df in DF_GRID:
        for t in T_GRID:
            true_p = p_two_tailed(t, df)
            for dec in (2, 3):
                shown = round_half_up(true_p, dec)
                if shown <= 0.0:
                    continue                      # p округлился в ноль
                consistent.append((t, df, shown, dec, true_p))

                wrong = round_half_up(min(true_p * 8.0, 0.97), dec)
                if wrong <= 0.0 or abs(wrong - shown) < 10.0 ** -dec:
                    continue
                inconsistent.append((t, df, wrong, dec, true_p))
    return consistent, inconsistent
