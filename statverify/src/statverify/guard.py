"""L2 — охранник извлечения.

Ворота между моделью и решателем. Чистое сравнение строк, никакой модели:
проверка завершаема по построению.

Охранник несёт не «повышение надёжности», а единственное в системе
различение двух причин отказа. Если значение дословно в исходнике,
лежит внутри границ утверждения и роль однозначна — извлечение верное,
значит неверна статья. Без этого различения петля переизвлечения
либо крутится, пока не подберёт согласованный набор (теряя настоящие
ошибки), либо валит всё в ложные срабатывания.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .model import Claim, Slot, parse_number

WINDOW = 48
# Расширение WINDOW не усиливает охранника — не принимай его за защиту.
# ROLE_MARKER всегда привязан к $ (концу префикса, вплотную перед
# числом), поэтому раздвижение окна лишь даёт маркеру больше места
# НАЙТИ совпадение там, где он и так его находит; лишний текст левее
# совпадения ни на что не влияет. Проверено мутационным прогоном:
# WINDOW=48 -> 4800 не роняет ни одного теста (run_tests.py, 255/255,
# те же 0 ложных отказов и 100% чувствительность к порче, что и на 48).
# Работу здесь делает $-привязка самого ROLE_MARKER, а не число WINDOW.
#
# Сужение — другое дело, оно режет то пространство, где маркер вообще
# может встретиться: WINDOW=12 роняет 1 тест, WINDOW=6 роняет 33 — вот
# это и есть настоящая, измеримая роль этого числа. Значит 48 — нижняя
# граница с запасом, а не произвольная константа для симметрии с чем-то.

# Маркер роли, привязанный к концу префикса ($): между маркером и числом
# не допускается ничего, кроме пробелов и знака отношения. Именно эта
# привязка ловит загрязнение из соседнего теста и перепутанные роли —
# любой вклинившийся текст разрывает совпадение.
ROLE_MARKER: Dict[str, re.Pattern] = {
    "df": re.compile(r"\bt\s*\(\s*(?:df\s*[=:]\s*)?$"),
    "t": re.compile(r"\bt\s*\([^()]*\)\s*[=:]\s*$"),
    "p": re.compile(r"\b[pP]\s*(?P<rel>[<>=≤≥]{1,2})\s*$"),
    "n": re.compile(r"\b[Nn]\s*[=:]\s*$"),
    # b/β — регрессионный коэффициент, M — среднее; оба играют роль
    # точечной оценки в R5/R4 и не различаются на этом уровне маркера.
    "est": re.compile(r"\b[bβM]\s*[=:]\s*$"),
    # SE и SD здесь не разграничены сознательно: маркер роли подтверждает
    # только «это стандартная ошибка/отклонение перед числом», а не то,
    # какая именно статистика имелась в виду в источнике — это решение
    # принадлежит извлечению (L1), не охраннику.
    "se": re.compile(r"\b(?:SE|SD)\s*[=:]\s*$"),
    "ci_lo": re.compile(r"\bCI\s*(?:[=:]\s*)?\[?\s*$"),
    # Два формата CI дают разный непосредственный префикс перед верхней
    # границей: '[lo, hi]' — запятая, 'lo to hi' — слово 'to'.
    "ci_hi": re.compile(r"(?:,|\bto)\s*$"),
}

_REL_CANON = {"=": "=", "==": "=", "<": "<", ">": ">", "≤": "<=", "≥": ">=",
              "<=": "<=", ">=": ">="}


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str
    slot: str
    detail: str = ""


def check_slot(slot: Slot, source: str, claim: Claim) -> GuardResult:
    lo, hi = slot.span

    # C0. Границы утверждения. Слот, вышедший за них, взят из соседнего
    # теста — структурно валидного, потому ни один из остальных
    # критериев его бы не поймал.
    if not (claim.span[0] <= lo and hi <= claim.span[1]):
        return GuardResult(False, "not_contained", slot.name,
                           f"span {slot.span} вне утверждения {claim.span}")

    # C1. Литерал действительно лежит по указанному смещению.
    if not (0 <= lo < hi <= len(source)):
        return GuardResult(False, "span_mismatch", slot.name, "span вне исходника")
    actual = source[lo:hi]
    if actual != slot.literal:
        return GuardResult(False, "span_mismatch", slot.name,
                           f"в исходнике {actual!r}, заявлено {slot.literal!r}")

    # C2. Значение порождается литералом, а не приписано к нему.
    parsed = parse_number(slot.literal)
    if parsed is None or abs(parsed - slot.value) > 1e-12:
        return GuardResult(False, "value_mismatch", slot.name,
                           f"{slot.literal!r} -> {parsed}, заявлено {slot.value}")

    # C3. Роль подтверждена маркером вплотную перед числом.
    marker = ROLE_MARKER.get(slot.name)
    if marker is None:
        return GuardResult(False, "role_missing", slot.name, "нет правила роли")
    prefix = source[max(0, lo - WINDOW):lo]
    m = marker.search(prefix)
    if m is None:
        return GuardResult(False, "role_missing", slot.name,
                           f"нет маркера {slot.name!r} перед {prefix[-24:]!r}")

    # C4. Записанное отношение совпадает с тем, что стоит в тексте.
    # Существенно для L4: 'p < .05' и 'p = .05' — разные интервалы.
    if "rel" in (m.groupdict() or {}):
        found = _REL_CANON.get(m.group("rel"), m.group("rel"))
        if found != slot.relation:
            return GuardResult(False, "relation_mismatch", slot.name,
                               f"в тексте {found!r}, заявлено {slot.relation!r}")

    return GuardResult(True, "ok", slot.name)


def check_claim(claim: Claim, source: str) -> List[GuardResult]:
    results = [check_slot(s, source, claim) for s in claim.slots.values()]
    for cands in claim.candidates.values():
        results.extend(check_slot(s, source, claim) for s in cands)
    return results


def admit(claim: Claim, source: str) -> Tuple[bool, List[GuardResult]]:
    """Пропустить ли утверждение в решатель целиком."""
    results = check_claim(claim, source)
    return all(r.ok for r in results), results
