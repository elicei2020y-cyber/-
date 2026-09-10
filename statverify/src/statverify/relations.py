"""L3 — каталог отношений и решение их в любую сторону.

Отношение здесь не предикат, а связь, решаемая относительно любой
своей переменной. Именно это превращает верификатор из сигнализации
в диагностику: вместо «p неверно» получается «из t и df следует
p в [.0158, .0163], сообщено .001».

Распространение интервалов ведётся по углам входного бокса. Это
корректно ровно потому, что каждая функция монотонна по каждому
аргументу по отдельности: двусторонний p убывает и по |t|, и по df.
Для немонотонной связи такой приём давал бы заниженный интервал
и, значит, ложные срабатывания, поэтому монотонность — не деталь
реализации, а условие внесения отношения в каталог.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Optional

from .interval import DOMAIN, Interval
from .tstats import df_from, p_two_tailed, t_from_p


@dataclass(frozen=True)
class Relation:
    key: str
    vars: tuple
    text: str
    designs: tuple          # дизайны, при которых отношение применимо
    solvers: Dict[str, Callable]
    # Переменные, которые вправе быть гипотезой «эта врёт», при локализации.
    # None означает «все vars» (обычный случай — R2a, R1a). Существует ради
    # псевдо-слотов вроде alpha в R4:width: alpha участвует в решении
    # (её значение нужно, чтобы получить t_crit), но она не заявлена в
    # статье и не может быть виновной — это допущение, а не число из текста.
    blamable: Optional[tuple] = None

    @property
    def blame_vars(self) -> tuple:
        return self.blamable if self.blamable is not None else self.vars

    def applies_to(self, design: str) -> bool:
        return design in self.designs

    def solve(self, target: str, known: Dict[str, Interval]) -> Optional[Interval]:
        """Интервал целевой переменной, выведенный из остальных."""
        fn = self.solvers.get(target)
        if fn is None:
            return None
        needed = [v for v in self.vars if v != target]
        if any(v not in known for v in needed):
            return None
        boxes = [known[v].corners for v in needed]
        vals = []
        for combo in product(*boxes):
            out = fn(dict(zip(needed, combo)))
            if out is not None:
                vals.append(out)
        if not vals:
            return None
        iv = Interval(min(vals), max(vals))
        dom = DOMAIN.get(target)
        if dom is None:
            return iv
        # Выход за область определения — содержательный отказ, а не повод
        # вернуть неурезанный интервал: None здесь означает «такого
        # значения переменной не существует», и локализация обязана
        # на этом основании отвергнуть гипотезу.
        return iv.meet(dom)


# --- R2a: двусторонний p как функция t и df -------------------------------

def _p_from(k):
    return p_two_tailed(k["t"], k["df"])


def _t_from(k):
    return t_from_p(k["p"], k["df"])


def _df_from(k):
    return df_from(k["t"], k["p"])


R2A = Relation(
    key="R2a",
    vars=("t", "df", "p"),
    text="p = 2·(1 − CDF_t(|t|, df))",
    designs=("two_tailed_t",),
    solvers={"p": _p_from, "t": _t_from, "df": _df_from},
)

# --- R1a: df = n − 1 ------------------------------------------------------
# Внесено в каталог, но на текущем этапе не срабатывает, и по двум
# независимым причинам сразу — обе честные, ни одна не обойдена:
#   1) слот n извлекается на уровне документа, а не утверждения,
#      и до появления механизма удалённых слотов не строится;
#   2) отношение применимо только к парному и одновыборочному тесту,
#      а для независимых выборок верно df = n1 + n2 − 2. Различить их
#      обязан слой классификации дизайна L1.5, которого пока нет.
# Пустой designs означает, что отношение не применится ни к чему,
# и утверждение получит UNVERIFIABLE вместо тихого пропуска проверки.

def _df_from_n(k):
    return k["n"] - 1.0


def _n_from_df(k):
    return k["df"] + 1.0


R1A = Relation(
    key="R1a",
    vars=("n", "df"),
    text="df = n − 1  (парный / одновыборочный)",
    designs=(),
    solvers={"df": _df_from_n, "n": _n_from_df},
)

# --- R5: t = оценка / SE --------------------------------------------------
# Пересекается с R2a по переменной t. Именно это пересечение и делает
# локализацию возможной: испорченный t ломает оба отношения сразу,
# а испорченный p или se ломает только одно — граф из двух отношений
# различает виновника там, где одно отношение не могло (задача 1).

def _t_from_est_se(k):
    se = k["se"]
    if se == 0.0:
        return None
    return k["est"] / se


def _est_from_t_se(k):
    return k["t"] * k["se"]


def _se_from_t_est(k):
    t = k["t"]
    if t == 0.0:
        return None
    return k["est"] / t


R5 = Relation(
    key="R5",
    vars=("est", "se", "t"),
    text="t = оценка / SE",
    # Формула верна для любого t-теста, но дизайн пока единственный
    # в каталоге; расширить designs, когда появится R2b.
    designs=("two_tailed_t",),
    solvers={"t": _t_from_est_se, "est": _est_from_t_se, "se": _se_from_t_est},
)


# --- R4: доверительный интервал --------------------------------------------
# Разложено на две связи, а не одну, ради монотонности (правило 1.4):
# центр и полуширина CI по отдельности монотонны по каждому аргументу,
# а «CI как единая формула от (est, se, df)» — нет, потому что интервал
# несимметричен относительно ошибок в ci_lo и ci_hi по отдельности.
#
# alpha не заявлена в тексте как число само по себе — это допущение
# (по умолчанию 0.05, т.е. 95% CI), которое verify.py обязан записать
# в вердикт, если явного уровня в статье не нашлось (см. Claim.ci_alpha).
# Поэтому alpha участвует в vars (нужна для решения), но не в blamable:
# «alpha врёт» — это утверждение о допущении верификатора, не о статье.

def _est_from_ci(k):
    return (k["ci_lo"] + k["ci_hi"]) / 2.0


def _cilo_from_est_cihi(k):
    return 2.0 * k["est"] - k["ci_hi"]


def _cihi_from_est_cilo(k):
    return 2.0 * k["est"] - k["ci_lo"]


R4_CENTER = Relation(
    key="R4:center",
    vars=("ci_lo", "ci_hi", "est"),
    text="(ci_lo + ci_hi) / 2 = оценка",
    designs=("two_tailed_t",),
    solvers={"est": _est_from_ci, "ci_lo": _cilo_from_est_cihi,
             "ci_hi": _cihi_from_est_cilo},
)


def _t_crit(df, alpha):
    """Критическое |t|: t_from_p уже инвертирует p_two_tailed(t, df),
    а t_crit(df, alpha) по определению — то t, при котором эта функция
    равна alpha. Отдельной численной процедуры не требуется."""
    return t_from_p(alpha, df)


def _se_from_ciw(k):
    tc = _t_crit(k["df"], k["alpha"])
    if tc is None or tc == 0.0:
        return None
    return (k["ci_hi"] - k["ci_lo"]) / (2.0 * tc)


def _cihi_from_ciw(k):
    tc = _t_crit(k["df"], k["alpha"])
    if tc is None:
        return None
    return k["ci_lo"] + 2.0 * tc * k["se"]


def _cilo_from_ciw(k):
    tc = _t_crit(k["df"], k["alpha"])
    if tc is None:
        return None
    return k["ci_hi"] - 2.0 * tc * k["se"]


# df НЕ входит в solvers ниже — сознательно, после того как выяснилось,
# что df_from(tc_needed, alpha) здесь небезопасно решать через обычное
# распространение по углам бокса, и вот почему.
#
# h(ci_lo, ci_hi, se) = df_from((ci_hi − ci_lo) / (2·se), alpha) монотонна
# по каждому аргументу (композиция монотонных функций), но t_crit(df,alpha)
# имеет горизонтальную асимптоту при df → ∞: как только (ci_hi−ci_lo)/(2se)
# опускается ниже этого предела, никакой df больше не подходит, и df_from
# возвращает None. Для df, близких к асимптотике (df ≳ 25 при alpha=.05),
# истинная комбинация (ci_lo, ci_hi, se) лежит РЯДОМ с этим порогом внутри
# бокса, заданного точностью записи, — а не снаружи. Угол бокса, где по
# направлению монотонности достигается теоретический максимум h, регулярно
# попадает ЗА порог (даёт None) и просто отбрасывается вместе со своим
# вкладом; уцелевшие углы, наоборот, смещены в сторону НИЗКИХ df. Итоговый
# интервал получается заведомо узким и не содержащим истинный df —
# проверено прогоном: при df=99, est=0.2, se=0.1 решение даёт [14.2, 32.4],
# не содержащее 99, и это не редкий случай, а систематика при df ≳ 25.
#
# Правило 1.4 («нужна немонотонная связь — сначала меняй способ
# распространения, потом вноси») здесь применяется в узком смысле: формула
# монотонна, но её область определения обрывается внутри бокса, и это
# ровно то, для чего стандартное распространение по углам не рассчитано.
# Правильный способ распространения для этого конкретного направления —
# не корневая интерполяция по 8 угла́м, а отдельная процедура, учитывающая
# асимптоту; её нет, и придумывать наспех — значит проносить непроверенную
# арифметику именно в то место, которому верификатор обязан доверять
# (тот же принцип, что уже объяснён в tstats.py про lgamma). Поэтому R4:width
# просто не предлагает направление на df, а не предлагает его неверно.
#
# se, ci_hi, ci_lo ниже эту асимптоту не задевают: t_crit(df, alpha) сам по
# себе гладкая ограниченная функция на всей области df > 0, и её прямое
# вычисление (а не обращение) не имеет такой проблемы.

R4_WIDTH = Relation(
    key="R4:width",
    vars=("alpha", "ci_lo", "ci_hi", "df", "se"),
    text="(ci_hi − ci_lo) / 2 = t_crit(df, alpha) · SE",
    designs=("two_tailed_t",),
    solvers={"se": _se_from_ciw, "ci_hi": _cihi_from_ciw,
             "ci_lo": _cilo_from_ciw},
    blamable=("ci_lo", "ci_hi", "se"),   # df тоже исключена — см. выше
)

CATALOG: List[Relation] = [R2A, R1A, R5, R4_CENTER, R4_WIDTH]


def for_design(design: str) -> List[Relation]:
    return [r for r in CATALOG if r.applies_to(design)]
