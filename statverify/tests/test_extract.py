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


# --- Задача 1: est/SE и доверительный интервал ------------------------

@pytest.mark.parametrize("text,est,se", [
    ("t(98) = 3.75, p = .001, b = 0.45, SE = 0.12.", 0.45, 0.12),
    ("t(98) = 3.75, p = .001, β = .45 (SE = .12).", 0.45, 0.12),
    ("t(98) = 3.75, p = .001, M = 3.2, SD = 1.1.", 3.2, 1.1),
])
def test_est_se_extracted(text, est, se):
    doc = extract(text)
    slots = doc.claims[0].slots
    assert abs(slots["est"].value - est) < 1e-9
    assert abs(slots["se"].value - se) < 1e-9


@pytest.mark.parametrize("text", [
    "t(98) = 3.75, p = .001, 95% CI [0.21, 0.69].",
    "t(98) = 3.75, p = .001, 95% CI = 0.21 to 0.69.",
])
def test_ci_extracted(text):
    doc = extract(text)
    slots = doc.claims[0].slots
    assert abs(slots["ci_lo"].value - 0.21) < 1e-9
    assert abs(slots["ci_hi"].value - 0.69) < 1e-9
    assert doc.claims[0].ci_alpha_explicit
    assert abs(doc.claims[0].ci_alpha - 0.05) < 1e-9


def test_ci_without_percentage_is_an_assumption():
    """Без явного уровня CI решатель обязан пометить alpha=0.05 как
    допущение (правило 1.2/8.5: ничего не проваливается молча)."""
    doc = extract("t(98) = 3.75, p = .001, CI [0.21, 0.69].")
    claim = doc.claims[0]
    assert not claim.ci_alpha_explicit
    assert abs(claim.ci_alpha - 0.05) < 1e-9


def test_est_se_and_ci_together():
    doc = extract("t(98) = 3.75, p = .001, b = 0.45, SE = 0.12, "
                  "95% CI [0.21, 0.69].")
    slots = doc.claims[0].slots
    assert {"t", "df", "p", "est", "se", "ci_lo", "ci_hi"} <= set(slots)


def test_loose_boundary_excludes_column_spliced_p_with_own_t_test():
    """ДЫРА 2 (найдена мутационным фаззингом PDF-порчи, реальный случай
    corpus_real.py id='A01' + 'A02', PLOS ONE 10.1371/journal.pone.0267297):
    склейка колонок вклинивает фрагмент соседней колонки между
    't(15) = 3.82' и его собственным 'p = .002)'. Вклинившийся фрагмент
    несёт СВОЙ t(27) раньше своего собственного 'p = .030'.

    До починки _claim_from_loose растягивал span утверждения до
    НАЙДЕННОГО кандидата p, и C0 в guard.py ('кандидат внутри границ
    утверждения') проверяла вложенность в границу, построенную по тому
    же кандидату, — тождественно истинно по построению. Извлекатель
    подменял p = .002 на чужой p = .030, и охранник это пропускал.

    Теперь окно поиска p обрывается на начале чужого t(...) НЕЗАВИСИМО
    от того, что за ним нашлось бы: p для исходного t(15)=3.82 не
    находится вовсе — безопасный отказ вместо тихо подмененного
    значения.
    """
    from corpus_real import CORPUS_REAL

    rec = next(r for r in CORPUS_REAL if r["id"] == "A01")
    foreign = next(r for r in CORPUS_REAL if r["id"] == "A02")["text"]
    assert "t(" in foreign, "у вклинившегося фрагмента обязан быть свой t-тест"

    spliced = rec["text"].replace(
        "t(15) = 3.82, p = .002)",
        f"t(15) = 3.82 {foreign[:70]} , p = .002)",
    )
    assert spliced != rec["text"]

    doc = extract(spliced)
    original = next(c for c in doc.claims
                     if c.slots.get("t") and c.slots["t"].value == 3.82
                     and c.slots.get("df") and c.slots["df"].value == 15.0)
    assert "p" not in original.slots, (
        "p из чужого предложения приклеился к исходному t/df — "
        "граница утверждения снова зависит от найденного кандидата")
    assert not original.candidates.get("p")
