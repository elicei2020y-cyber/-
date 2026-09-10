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

CATALOG: List[Relation] = [R2A, R1A]


def for_design(design: str) -> List[Relation]:
    return [r for r in CATALOG if r.applies_to(design)]
