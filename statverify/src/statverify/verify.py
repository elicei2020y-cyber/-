"""L4 — распространение интервалов. L5 — локализация. L7 — сборка вердикта.

CONSISTENT здесь не означает «верно». Оно означает «внутренних
противоречий не обнаружено»: статья может быть безупречно согласованной
и при этом полностью неправильной. Формулировка в выдаче держится
именно такой намеренно.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .interval import DOMAIN, Interval, from_reported
from .model import Claim
from .relations import Relation, for_design

# Правдоподобие для отсева гипотез при локализации. Это не мягкие
# эвристики из L6, а жёсткие ограничения предметной области: t-статистика
# за сотню означает не «подозрительно», а «такого не сообщают».
T_PLAUSIBLE = Interval(-100.0, 100.0)


@dataclass
class Hypothesis:
    var: str
    derived: Optional[Interval]
    admissible: bool
    why: str


@dataclass
class Verdict:
    status: str                     # CONSISTENT | INCONSISTENT | ...
    relation: Optional[str] = None
    var: Optional[str] = None
    expected: Optional[Interval] = None
    reported: Optional[Interval] = None
    reason: str = ""
    hypotheses: List[Hypothesis] = field(default_factory=list)

    def line(self) -> str:
        if self.status == "CONSISTENT":
            return "CONSISTENT — внутренних противоречий не обнаружено"
        if self.status == "INCONSISTENT":
            return (f"INCONSISTENT [{self.relation}] — врёт {self.var}: "
                    f"следует {self.expected}, сообщено {self.reported}")
        if self.status == "INCONSISTENT_AMBIGUOUS":
            live = ", ".join(h.var for h in self.hypotheses if h.admissible)
            return (f"INCONSISTENT [{self.relation}] — противоречие есть, "
                    f"виновник не выделен; допустимы: {live}")
        if self.status == "INCONSISTENT_UNLOCALIZED":
            return f"INCONSISTENT [{self.relation}] — ни одна гипотеза не проходит"
        return f"{self.status} — {self.reason}"


def reported_intervals(claim: Claim) -> Dict[str, Interval]:
    out = {}
    for name, s in claim.slots.items():
        out[name] = from_reported(s.value, s.decimals, s.relation, DOMAIN.get(name))
    return out


def _plausible(var: str, iv: Optional[Interval]) -> tuple:
    if iv is None:
        return False, "решения не существует"
    if var == "df":
        if not iv.contains_integer():
            return False, f"{iv} не содержит целого df"
        return True, ""
    if var == "t":
        if not iv.intersects(T_PLAUSIBLE):
            return False, f"{iv} вне правдоподобного диапазона t"
        return True, ""
    return True, ""


def localize(rel: Relation, rep: Dict[str, Interval]) -> List[Hypothesis]:
    """Для каждой переменной отношения проверить гипотезу «врёт она».

    Гипотеза отвергается, если из остальных переменных значение для неё
    не выводится вовсе. Для df это происходит регулярно и бесплатно:
    двусторонний p ограничен снизу нормальным пределом 2(1−Φ(|t|)),
    поэтому при сообщённом p ниже этого предела никакое df тройку
    не согласует, и df оправдан.
    """
    out = []
    for v in rel.vars:
        if v not in rep:
            continue
        others = {k: iv for k, iv in rep.items() if k != v and k in rel.vars}
        derived = rel.solve(v, others)
        ok, why = _plausible(v, derived)
        out.append(Hypothesis(v, derived, ok, why))
    return out


def verify_claim(claim: Claim) -> Verdict:
    if claim.candidates:
        which = ", ".join(claim.candidates)
        return Verdict("AMBIGUOUS_PARSE",
                       reason=f"неразрешённые кандидаты: {which}")

    rels = for_design(claim.design)
    if not rels:
        return Verdict("UNVERIFIABLE",
                       reason=f"для дизайна {claim.design!r} нет применимых отношений")

    rep = reported_intervals(claim)

    usable = [r for r in rels if all(v in rep for v in r.vars)]
    if not usable:
        missing = sorted({v for r in rels for v in r.vars} - set(rep))
        return Verdict("UNVERIFIABLE",
                       reason=f"недостаёт слотов: {', '.join(missing)}")

    for rel in usable:
        # Обнаружение ведётся в одну сторону — этого достаточно:
        # если выведенный p не пересекается с сообщённым, то и любое
        # другое направление того же отношения не сойдётся.
        target = rel.vars[-1]
        derived = rel.solve(target, {k: v for k, v in rep.items() if k in rel.vars})
        if derived is None:
            return Verdict("UNVERIFIABLE", relation=rel.key,
                           reason=f"{target} не выводится из остальных")
        if derived.intersects(rep[target]):
            continue

        hyps = localize(rel, rep)
        live = [h for h in hyps if h.admissible]
        if len(live) == 1:
            h = live[0]
            return Verdict("INCONSISTENT", rel.key, h.var, h.derived,
                           rep[h.var], hypotheses=hyps)
        if not live:
            return Verdict("INCONSISTENT_UNLOCALIZED", rel.key, hypotheses=hyps)
        return Verdict("INCONSISTENT_AMBIGUOUS", rel.key, hypotheses=hyps)

    return Verdict("CONSISTENT")
