"""Извлечение на размеченном корпусе.

Важная оговорка о смысле этих цифр: корпус и извлекатель написаны
под одну задачу, поэтому 100% здесь означают, что регекс справляется
с предусмотренными для него форматами, и ничего не говорят о реальных
статьях. Настоящая проверка появится с задачей 3 из
docs/TASK_FOR_CLAUDE_CODE.md.
"""

import pytest

from statverify.extract import extract

from corpus import CORPUS


def _match(found, expected):
    pairs, rest = [], list(found)
    for exp in expected:
        hit = next((c for c in rest
                    if "t" in c.slots and "df" in c.slots
                    and abs(c.slots["t"].value - exp["t"]) < 1e-9
                    and abs(c.slots["df"].value - exp["df"]) < 1e-9), None)
        if hit is not None:
            rest.remove(hit)
        pairs.append((exp, hit))
    return pairs, rest


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c["id"])
def test_claims_found(case):
    doc = extract(case["text"])
    pairs, spurious = _match(doc.claims, case["claims"])
    assert not spurious, f"лишние утверждения: {len(spurious)}"
    for exp, got in pairs:
        assert got is not None, f"не найдено t={exp['t']}, df={exp['df']}"


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c["id"])
def test_p_slot(case):
    doc = extract(case["text"])
    for exp, got in _match(doc.claims, case["claims"])[0]:
        if got is None:
            continue
        if exp["p"] is None:
            # Неоднозначный p не разрешается принудительно: отсекать
            # кандидатов — работа решателя, а не извлекателя. Извлекатель,
            # выбирающий один вариант из нескольких, — та самая точка,
            # где statcheck теряет точность.
            assert "p" not in got.slots
            continue
        s = got.slots.get("p")
        assert s is not None, "слот p отсутствует"
        assert abs(s.value - exp["p"]) < 1e-9
        assert s.relation == exp["prel"]


def test_precision_is_recorded():
    """Точность записи питает интервалы L4 и обязана сохраняться."""
    doc = extract("Scores differed, t(99) = 2.45, p = .016.")
    slots = doc.claims[0].slots
    assert slots["p"].decimals == 3
    assert slots["t"].decimals == 2
    assert slots["df"].decimals == 0


def test_relation_is_recorded():
    """'p < .001' и 'p = .001' дают непересекающиеся интервалы."""
    doc = extract("A large effect emerged, t(31) = 4.21, p < .001.")
    assert doc.claims[0].slots["p"].relation == "<"
