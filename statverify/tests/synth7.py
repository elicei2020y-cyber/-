"""Порождение размеченных случаев на всех семи величинах: t, df, p, est,
se, ci_lo, ci_hi. То же соглашение, что в synth.py (арифметика, не рука,
округление к ближайшему), расширенное с тройки R2a на граф R5 + R4
(задача 1 в docs/TASK_FOR_CLAUDE_CODE.md).

Общий источник для scripts/run_step3.py и test_verify.py — числа,
которые видит пользователь в прогоне run_step3.py, и то, что проверяет
pytest, обязаны быть одной и той же арифметикой, а не двумя похожими.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from statverify.model import Claim, Slot, decimals_of
from statverify.tstats import p_two_tailed, t_from_p

DF_GRID = [10, 24, 49, 99, 250]
EST_GRID = [0.2, 0.5, 1.0]
SE_GRID = [0.1, 0.25, 0.5]

P_DEC, T_DEC, EST_DEC, SE_DEC, CI_DEC = 3, 2, 2, 2, 2

# Абсолютный сдвиг при порче — по той же логике, что ABS_BUMP в
# run_step2.py для p (сдвиг на порядок величины, а не на волосок):
# заведомо больше половины цены разряда, чтобы порча гарантированно
# вышла за интервал, а не балансировала на границе.
ABS_BUMP = {"t": 1.5, "est": 0.6, "se": 0.6, "ci_lo": -0.8, "ci_hi": 0.8}

VARS = ("t", "df", "p", "est", "se", "ci_lo", "ci_hi")


def round_half_up(x: float, d: int) -> float:
    q = Decimal(1).scaleb(-d)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def _slot(name: str, value: float, literal: str, rel: str = "=") -> Slot:
    return Slot(name, value, literal, (0, 1), decimals_of(literal), rel)


def gold(df: float, est: float, se: float) -> dict:
    """Семь взаимно согласованных величин, вычисленных арифметикой."""
    t = est / se
    p = p_two_tailed(t, df)
    tcrit = t_from_p(0.05, df)
    ci_lo, ci_hi = est - tcrit * se, est + tcrit * se
    return dict(t=t, df=df, p=p, est=est, se=se, ci_lo=ci_lo, ci_hi=ci_hi)


def make_claim(values: dict) -> Claim:
    v = values
    slots = {
        "t": _slot("t", v["t"], f"{round_half_up(v['t'], T_DEC):.{T_DEC}f}"),
        "df": _slot("df", v["df"], f"{v['df']:g}"),
        "p": _slot("p", v["p"], f"{round_half_up(v['p'], P_DEC):.{P_DEC}f}"),
        "est": _slot("est", v["est"], f"{round_half_up(v['est'], EST_DEC):.{EST_DEC}f}"),
        "se": _slot("se", v["se"], f"{round_half_up(v['se'], SE_DEC):.{SE_DEC}f}"),
        "ci_lo": _slot("ci_lo", v["ci_lo"], f"{round_half_up(v['ci_lo'], CI_DEC):.{CI_DEC}f}"),
        "ci_hi": _slot("ci_hi", v["ci_hi"], f"{round_half_up(v['ci_hi'], CI_DEC):.{CI_DEC}f}"),
    }
    return Claim(span=(0, 1), design="two_tailed_t", slots=slots,
                 ci_alpha=0.05, ci_alpha_explicit=True)


def corrupt_case(df: float, est: float, se: float, var: str) -> Claim:
    v = dict(gold(df, est, se))
    if var == "p":
        v["p"] = min(v["p"] * 8.0, 0.97)
    elif var == "df":
        v["df"] = v["df"] * 3.0 + 30.0
    else:
        v[var] = v[var] + ABS_BUMP[var]
    return make_claim(v)


def build_grid():
    """(df, est, se), где округление p до P_DEC не вырождается в ноль —
    тот же фильтр, что в synth.build_cases(): записи 'p = .000' не
    бывает, репортуют 'p < .001'. Без фильтра df_from на вырожденном p
    численно нестабилен и даёт случайные ложные срабатывания, не
    относящиеся к R5/R4 (см. run_step3.py, историю правки)."""
    for df in DF_GRID:
        for est in EST_GRID:
            for se in SE_GRID:
                t = est / se
                true_p = p_two_tailed(t, df)
                if round_half_up(true_p, P_DEC) <= 0.0:
                    continue
                yield df, est, se
