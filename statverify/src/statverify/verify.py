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
    # Допущения, использованные при выводе (например, alpha=0.05 для CI,
    # когда уровень не был указан в тексте явно) — правило 1.2/8.5:
    # ничего не проваливается в вердикт молча.
    assumptions: List[str] = field(default_factory=list)

    def _tail(self) -> str:
        return f" [{'; '.join(self.assumptions)}]" if self.assumptions else ""

    def line(self) -> str:
        if self.status == "CONSISTENT":
            return "CONSISTENT — внутренних противоречий не обнаружено" + self._tail()
        if self.status == "INCONSISTENT":
            return (f"INCONSISTENT [{self.relation}] — врёт {self.var}: "
                    f"следует {self.expected}, сообщено {self.reported}" + self._tail())
        if self.status == "INCONSISTENT_AMBIGUOUS":
            live = ", ".join(h.var for h in self.hypotheses if h.admissible)
            return (f"INCONSISTENT [{self.relation}] — противоречие есть, "
                    f"виновник не выделен; допустимы: {live}" + self._tail())
        if self.status == "INCONSISTENT_UNLOCALIZED":
            return (f"INCONSISTENT [{self.relation}] — ни одна гипотеза не проходит"
                    + self._tail())
        return f"{self.status} — {self.reason}" + self._tail()


def reported_intervals(claim: Claim) -> Dict[str, Interval]:
    out = {}
    for name, s in claim.slots.items():
        out[name] = from_reported(s.value, s.decimals, s.relation, DOMAIN.get(name))
    # alpha — не заявленное число, а допущение (см. model.Claim.ci_alpha):
    # вырожденный интервал в одну точку, всегда доступный отношению
    # R4:width, если у утверждения вообще есть слоты CI.
    out["alpha"] = Interval(claim.ci_alpha, claim.ci_alpha)
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


def _relation_holds(rel: Relation, rep: Dict[str, Interval]) -> bool:
    """Согласовано ли отношение само по себе, без гипотезы о виновном.

    Проверка ведётся в одну сторону (последняя переменная из
    blame_vars) — этого достаточно: если выведенное значение не
    пересекается с сообщённым, то и любое другое направление того же
    отношения не сойдётся, потому что это одно и то же равенство.
    """
    target = rel.blame_vars[-1] if rel.blame_vars else rel.vars[-1]
    if target not in rep:
        return True             # переменной нет — отношение неприменимо
    others = {k: v for k, v in rep.items() if k != target and k in rel.vars}
    derived = rel.solve(target, others)
    return derived is not None and derived.intersects(rep[target])


def localize(usable: List[Relation], rep: Dict[str, Interval]) -> List[Hypothesis]:
    """Кросс-отношенческая локализация виновной переменной.

    Одного отношения (например, только R2a: t, df, p) недостаточно,
    чтобы указать виновника — три переменные, одна связь, задача
    недоопределена, и это фиксирует test_localization_is_ambiguous_
    without_r5_r4. С добавлением R5 и R4 отношения образуют граф
    с общими переменными (t общая у R2a и R5, est/se — у R5 и R4,
    ci_lo/ci_hi/df — у R4:center и R4:width), и виновника можно
    выделить перекрёстной проверкой:

    Гипотеза «врёт переменная v» допустима, только если
      (a) для каждого отношения, где v участвует, значение v оттуда
          выводимо и правдоподобно (обычная проверка одного отношения:
          «если v — единственная ошибка, каким значением она должна
          была быть»), И
      (b) каждое отношение, где v НЕ участвует, остаётся согласованным
          само по себе — если оно тоже нарушено, а v в нём не участвует,
          значит, испорчена ещё какая-то переменная, и v одна вину не
          объясняет.

    Для df условие (a) даёт бесплатное оправдание почти всегда:
    двусторонний p ограничен снизу нормальным пределом 2(1−Φ(|t|)),
    поэтому при сообщённом p ниже этого предела никакое df тройку
    не согласует.

    Три разных положения отношения относительно гипотезы «врёт v», а не
    два, и путать их значит дать голос тому, у кого его быть не должно:

      1. v не входит в vars отношения вовсе — оно ПОДТВЕРЖДАЕТ или
         ОПРОВЕРГАЕТ гипотезу тем, держится ли само по себе на
         заявленных значениях (_relation_holds). Это обоснованно: если
         v ни при чём, а отношение всё равно нарушено — виновата не
         одна v.
      2. v входит в vars И у отношения есть solvers[v] — оно ГОЛОСУЕТ:
         предлагает значение v (через solve) и голосует «против», если
         предложенное значение неприемлемо или расходится с другим
         таким же голосом.
      3. v входит в vars, но solvers[v] НЕТ — отношение не умеет ни
         предложить v, ни (в отличие от случая 1) быть проверено «само
         по себе» на заявленных значениях: раз v — его вход, а v сейчас
         предполагается неверной, проверка через _relation_holds на
         РЕПОРТИРОВАННОМ (неверном) v почти неизбежно провалится —
         и это было бы засчитано как голос ПРОТИВ гипотезы, которую
         сама эта неверная v и объясняет. Отношение в этом положении
         обязано ВОЗДЕРЖАТЬСЯ — не голосовать ни за, ни против.

    Смешение случаев 1 и 3 было настоящей ошибкой первой версии этого
    исправления: относя R4:width (которое не умеет решать df — см.
    relations.R4_WIDTH) к «уже держится сама по себе», локализация
    проверяла R4:width на испорченном df и вето получала уже ЛЮБая df,
    в том числе честно локализуемая раньше (df=10 и подобные) — это
    было хуже, чем ошибка, которую чинили: правильных ответов стало
    меньше, а не больше. Собственно голосующий вклад R4:width (случай 2)
    был неисправен из-за асимптотической неустойчивости df_from — тот
    отдельный разбор остаётся в relations.py.
    """
    all_vars = sorted({v for r in usable for v in r.blame_vars if v in rep})
    out = []
    for v in all_vars:
        can_vote = [r for r in usable if v in r.vars and v in r.solvers]
        must_hold = [r for r in usable if v not in r.vars]
        # Случай 3 выше: v — вход отношения, но оно не умеет его решать.
        # Не in must_hold (проверка на испорченном v дала бы ложный голос
        # против) и не в can_vote (нечем голосовать) — воздержание.

        admissible = True
        reasons: List[str] = []
        derived_for_v: Optional[Interval] = None
        for r in can_vote:
            others = {k: iv for k, iv in rep.items() if k != v and k in r.vars}
            d = r.solve(v, others)
            ok, why = _plausible(v, d)
            if not ok:
                admissible = False
                reasons.append(f"{r.key}: {why}")
                continue
            if derived_for_v is None:
                derived_for_v = d
            else:
                inter = derived_for_v.meet(d)
                if inter is None:
                    admissible = False
                    reasons.append(f"{r.key}: не согласуется с другим отношением "
                                    f"на ту же переменную")
                else:
                    derived_for_v = inter

        if admissible:
            for r in must_hold:
                if not _relation_holds(r, rep):
                    admissible = False
                    reasons.append(f"{r.key}: без {v} остаётся противоречивым")

        out.append(Hypothesis(v, derived_for_v, admissible, "; ".join(reasons)))
    return out


def _ci_assumption(claim: Claim, usable: List[Relation]) -> List[str]:
    uses_ci = any(r.key == "R4:width" for r in usable)
    if uses_ci and not claim.ci_alpha_explicit:
        return [f"уровень CI не указан в тексте явно, "
                f"предположено alpha={claim.ci_alpha:g}"]
    return []


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
        missing = sorted({v for r in rels for v in r.vars} - set(rep) - {"alpha"})
        return Verdict("UNVERIFIABLE",
                       reason=f"недостаёт слотов: {', '.join(missing)}")

    assumptions = _ci_assumption(claim, usable)

    violated = []
    for rel in usable:
        # derived is None — тоже нарушение, не абстенция: значения,
        # согласующего оставшиеся переменные этого отношения, не
        # существует в области определения (например, из явно неверного
        # ci_lo решается отрицательная SE). Раньше (при одном отношении
        # R2a) этот путь был почти недостижим, потому что p_two_tailed
        # почти всегда даёт валидную вероятность; с R4:width, где решаемая
        # величина обязана быть положительной, он стал реальным — и
        # прежнее решение «в этом случае абстенция» тихо прятало бы
        # находку за UNVERIFIABLE, не давая локализации её увидеть.
        target = rel.blame_vars[-1] if rel.blame_vars else rel.vars[-1]
        others = {k: v for k, v in rep.items() if k != target and k in rel.vars}
        derived = rel.solve(target, others)
        if derived is None or not derived.intersects(rep[target]):
            violated.append(rel)

    if not violated:
        return Verdict("CONSISTENT", assumptions=assumptions)

    hyps = localize(usable, rep)
    live = [h for h in hyps if h.admissible]
    rel_label = ", ".join(sorted({r.key for r in violated}))
    if len(live) == 1:
        h = live[0]
        return Verdict("INCONSISTENT", rel_label, h.var, h.derived,
                       rep[h.var], hypotheses=hyps, assumptions=assumptions)
    if not live:
        return Verdict("INCONSISTENT_UNLOCALIZED", rel_label, hypotheses=hyps,
                       assumptions=assumptions)
    return Verdict("INCONSISTENT_AMBIGUOUS", rel_label, hypotheses=hyps,
                   assumptions=assumptions)
