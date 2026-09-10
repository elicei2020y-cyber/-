"""Ядро t-распределения против scipy.

scipy здесь только оракул: рантайм от него не зависит. Смысл в том,
что две независимые реализации дают основание доверять числу, а одна,
взятая на веру, — нет. Именно эта сверка поймала расхождение в
префакторе неполной беты (docs/ARCHITECTURE.md, 8.3).
"""

import math
import random

import pytest

from statverify.tstats import df_from, p_normal_limit, p_two_tailed, t_from_p

sps = pytest.importorskip("scipy.stats", reason="scipy — оракул, нужен только тестам")

# Порог выведен из требования задачи, а не подогнан под результат.
# Самое узкое сравнение в L4 — интервал при четырёх знаках после
# запятой, полуширина 5e-5. Требуем тысячекратный запас.
TOLERANCE_P = 5e-5 / 1000


def test_p_two_tailed_matches_scipy():
    rng = random.Random(20260910)
    worst = 0.0
    for _ in range(4000):
        df = rng.choice([1, 2, 3, 5, 8, 13, 24, 45.3, 99, 250, 1000, 10000])
        t = rng.uniform(0.0, 8.0)
        ref = 2.0 * sps.t.sf(abs(t), df)
        worst = max(worst, abs(p_two_tailed(t, df) - ref) / max(ref, 1e-300))
    assert worst < TOLERANCE_P, f"расхождение с scipy {worst:.3e}"


def test_t_from_p_matches_scipy():
    rng = random.Random(4242)
    for _ in range(300):
        df = rng.choice([3, 10, 30, 99, 500])
        p = 10 ** rng.uniform(-6, -0.05)
        got, ref = t_from_p(p, df), sps.t.isf(p / 2.0, df)
        assert got is not None
        assert abs(got - ref) / ref < 1e-9


def test_t_crit_matches_scipy():
    """t_from_p(alpha, df) — критическое значение для доверительного
    интервала. На этом будет строиться отношение R4."""
    for df in (10, 24, 99, 250):
        assert abs(t_from_p(0.05, df) - sps.t.ppf(0.975, df)) < 1e-12


def test_df_from_is_inverse():
    for df in (5, 20, 99, 300):
        for t in (1.5, 2.45, 3.2, 5.0):
            p = p_two_tailed(t, df)
            back = df_from(t, p)
            assert back is not None
            assert abs(p_two_tailed(t, back) - p) / p < 1e-9


@pytest.mark.parametrize("t", [1.0, 1.96, 2.45, 3.0, 4.0])
def test_normal_limit_is_a_floor(t):
    """Двусторонний p не опускается ниже предела ни при каком df.

    На этом держится бесплатное оправдание df при локализации.
    """
    limit = p_normal_limit(t)
    for df in (1, 5, 50, 1000, 100000, 10_000_000):
        assert p_two_tailed(t, df) >= limit - 1e-15


@pytest.mark.parametrize("t", [1.0, 1.96, 2.45, 3.0, 4.0])
def test_df_from_rejects_below_normal_limit(t):
    """Цель ниже предела недостижима — ответ None, а не подогнанное df."""
    assert df_from(t, p_normal_limit(t) * 0.5) is None


@pytest.mark.parametrize("t,df,want", [(1.96, 1e7, 0.05), (2.0, 60, 0.0500),
                                       (2.45, 99, 0.01604)])
def test_reference_values(t, df, want):
    assert math.isclose(p_two_tailed(t, df), want, rel_tol=5e-3)
