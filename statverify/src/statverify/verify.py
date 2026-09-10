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
from .tstats import p_two_tailed

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
    # Расхождение, вероятно, объясняется печатью нижнего порога ('p = .001'
    # вместо настоящего значения, на порядки меньшего) — см. _floor_convention.
    # НЕ переводит вердикт в CONSISTENT: противоречие в заявленных числах
    # остаётся, называется лишь его вероятное происхождение. Разница
    # существенна, потому что настоящий p при такой печати утрачен и
    # проверить дальше нельзя (docs/CORPUS_REAL_NOTES.md).
    floor_convention: bool = False
    floor_threshold: Optional[float] = None

    def _tail(self) -> str:
        notes = list(self.assumptions)
        if self.floor_convention:
            notes.append(
                f"расхождение может объясняться печатью нижнего порога "
                f"p={self.floor_threshold:g} — настоящий p утрачен при печати, "
                f"проверить точнее нельзя")
        return f" [{'; '.join(notes)}]" if notes else ""

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

    Случай 3 — не гипотетический, а зафиксирован тем, как ломалась первая
    версия этого исправления: относя отношение, которое временно не имело
    решателя для v, к случаю 1 («уже держится сама по себе»), локализация
    проверяла его на ИСПОРЧЕННОМ v и получала вето для любой v, в том
    числе честно локализуемой раньше — это было хуже ошибки, которую
    чинили: правильных ответов стало меньше, а не больше. Разграничение
    трёх случаев остаётся нужным независимо от того, у скольких отношений
    сейчас есть решатель для конкретной переменной: solvers у relations.py
    может не быть по любой причине (ещё не написан, сознательно не введён
    на момент правки), и в любом случае отношение обязано воздержаться,
    а не голосовать на заведомо неверном входе.
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


# SPSS и часть журналов печатают 'p = .001' (или .05/.01/.0001) для всего,
# что меньше порога, — не ошибка авторов, а способ печати. Найдено на
# реальном корпусе (docs/CORPUS_REAL_NOTES.md): 5 из 8 расхождений на 49
# утверждениях из настоящих статей PLOS ONE именно такие.
_FLOOR_THRESHOLDS = (0.05, 0.01, 0.001, 0.0001)


def _floor_convention(claim: Claim) -> Optional[float]:
    """Порог, объясняющий расхождение печатью, если условие выполнено.

    Условие жёсткое и проверяемое, не эвристика и не смягчение вердикта:
    отношение записано как '=', сообщённый p совпадает РОВНО с одним из
    типовых порогов, и точно вычисленный двусторонний p по заявленным
    (точечным, не интервальным) t и df строго меньше сообщённого. Сравнение
    здесь намеренно точечное, не интервальное: вопрос не «пересекаются ли
    интервалы записи», а «настоящий p меньше того, что напечатано, ровно
    настолько, насколько ожидается при усечении к порогу» — то же
    вычисление, что в scripts/build_corpus_real.py, при котором находка
    и была впервые замечена.

    Ничего не решает за пользователя: возвращает порог как кандидатное
    объяснение, а не сигнал ослабить вердикт. Статья, напечатавшая
    'p = .001' вместо истинных 3e-11, всё ещё содержит утверждение, чей
    истинный p нельзя восстановить точнее этого порога, — то есть
    настоящую, не устранимую неопределённость, а не просто найденную и
    объяснённую придирку.
    """
    t_slot = claim.slots.get("t")
    df_slot = claim.slots.get("df")
    p_slot = claim.slots.get("p")
    if not (t_slot and df_slot and p_slot):
        return None
    if p_slot.relation != "=":
        return None
    threshold = next((th for th in _FLOOR_THRESHOLDS
                       if abs(p_slot.value - th) < 1e-12), None)
    if threshold is None:
        return None
    exact = p_two_tailed(t_slot.value, df_slot.value)
    return threshold if exact < p_slot.value else None


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
    floor_threshold = _floor_convention(claim)
    floor_kwargs = dict(floor_convention=floor_threshold is not None,
                         floor_threshold=floor_threshold)
    if len(live) == 1:
        h = live[0]
        return Verdict("INCONSISTENT", rel_label, h.var, h.derived,
                       rep[h.var], hypotheses=hyps, assumptions=assumptions,
                       **floor_kwargs)
    if not live:
        return Verdict("INCONSISTENT_UNLOCALIZED", rel_label, hypotheses=hyps,
                       assumptions=assumptions, **floor_kwargs)
    return Verdict("INCONSISTENT_AMBIGUOUS", rel_label, hypotheses=hyps,
                   assumptions=assumptions, **floor_kwargs)
