"""Тестовый корпус с разметкой.

Не бенчмарк — калибровочный набор для шага 1. Сюда специально
включены ловушки, на которых регексповые извлекатели ломаются
в реальной литературе: числа-отвлекатели рядом со слотом, два теста
в одном предложении, разрыв между t и p, юникодные минусы,
перенос строки внутри отчёта о тесте.
"""

from typing import Any, Dict, List

# p=None означает: p в этом утверждении не сообщён либо не должен
# быть разрешён принудительно (уходит в кандидаты).
CORPUS: List[Dict[str, Any]] = [
    dict(id="std_apa", text="Performance differed between conditions, t(99) = 2.45, p = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="std_leading_zero", text="The effect was reliable, t(24) = 3.10, p = 0.005.",
         claims=[dict(t=3.10, df=24.0, p=0.005, prel="=")]),
    dict(id="space_before_paren", text="Scores dropped, t (48) = -2.02, p = .049.",
         claims=[dict(t=-2.02, df=48.0, p=0.049, prel="=")]),
    dict(id="no_spaces", text="Result: t(99)=2.45, p=.016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="df_labelled", text="We found t(df = 99) = 2.45, p = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="semicolon", text="Comparison yielded t(99) = 2.45; p = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="p_in_parens", text="The contrast was significant, t(99) = 2.45 (p = .016).",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="uppercase_p", text="Group means differed, t(99) = 2.45, P = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="p_less_than", text="A large effect emerged, t(31) = 4.21, p < .001.",
         claims=[dict(t=4.21, df=31.0, p=0.001, prel="<")]),
    dict(id="p_nonsig", text="No difference was detected, t(60) = 1.75, p = .085.",
         claims=[dict(t=1.75, df=60.0, p=0.085, prel="=")]),
    dict(id="unicode_minus", text="The reversal held, t(40) = −2.15, p = .038.",
         claims=[dict(t=-2.15, df=40.0, p=0.038, prel="=")]),
    dict(id="welch_df", text="Using Welch correction, t(45.3) = 2.11, p = .040.",
         claims=[dict(t=2.11, df=45.3, p=0.040, prel="=")]),
    dict(id="gap_clause", text="Accuracy improved, t(99) = 2.45, which remained reliable, p = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="nbsp", text="Latency fell, t(99) = 2.45, p = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),

    # Ловушки.
    dict(id="two_tests", text="Accuracy rose, t(99) = 2.45, p = .016 and speed fell, t(49) = 1.20, p = .236.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="="),
                 dict(t=1.20, df=49.0, p=0.236, prel="=")]),
    dict(id="distractor_before_df", text="Among 99 participants, t(45) = 2.45, p = .016.",
         claims=[dict(t=2.45, df=45.0, p=0.016, prel="=")]),
    dict(id="distractor_before_t", text="The 2.45 criterion was applied; t(30) = 1.98, p = .057.",
         claims=[dict(t=1.98, df=30.0, p=0.057, prel="=")]),
    dict(id="distractor_p_like", text="With alpha = .05, t(72) = 2.90, p = .005.",
         claims=[dict(t=2.90, df=72.0, p=0.005, prel="=")]),
    dict(id="sentence_break", text="The main effect held, t(99) = 2.45. This was reliable, p = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="line_break", text="Reaction times differed, t(99) = 2.45,\np = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="ambiguous_p", text="The test gave t(99) = 2.45. Both p = .016 and p = .023 appear.",
         claims=[dict(t=2.45, df=99.0, p=None, prel=None)]),
    dict(id="no_p", text="The comparison produced t(99) = 2.45 for the primary outcome.",
         claims=[dict(t=2.45, df=99.0, p=None, prel=None)]),
    dict(id="ns_form", text="The groups did not differ, t(99) = 0.45, ns.",
         claims=[dict(t=0.45, df=99.0, p=None, prel=None)]),

    # Ловушки на повтор литерала: то же число встречается второй раз,
    # так что сравнение литерала и значения слепо и работать обязаны
    # границы утверждения (C0) либо маркер роли (C3).
    dict(id="df_repeats_outside", text="Among 99 participants, t(99) = 2.45, p = .016.",
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="=")]),
    dict(id="df_repeats_inside", text="The contrast held, t(45) = 2.45, p = .045.",
         claims=[dict(t=2.45, df=45.0, p=0.045, prel="=")]),
    dict(id="df_repeats_both", text="All 30 trials counted, t(30) = 2.30, p = .030.",
         claims=[dict(t=2.30, df=30.0, p=0.030, prel="=")]),

    # Больше документов с несколькими утверждениями — порча cross_claim
    # проверялась всего на двух примерах.
    dict(id="three_tests",
         text=("Accuracy rose, t(99) = 2.45, p = .016; latency fell, t(99) = 3.01, p = .003; "
               "confidence was flat, t(98) = 0.44, p = .661."),
         claims=[dict(t=2.45, df=99.0, p=0.016, prel="="),
                 dict(t=3.01, df=99.0, p=0.003, prel="="),
                 dict(t=0.44, df=98.0, p=0.661, prel="=")]),
    dict(id="two_tests_adjacent_df",
         text="Group A improved, t(24) = 2.10, p = .046, whereas group B did not, t(25) = 0.90, p = .377.",
         claims=[dict(t=2.10, df=24.0, p=0.046, prel="="),
                 dict(t=0.90, df=25.0, p=0.377, prel="=")]),
    dict(id="two_tests_no_p_second",
         text="The primary test gave t(50) = 2.60, p = .012, and the secondary gave t(50) = 1.10.",
         claims=[dict(t=2.60, df=50.0, p=0.012, prel="="),
                 dict(t=1.10, df=50.0, p=None, prel=None)]),

    # Ничего не должно извлекаться.
    dict(id="prose_only", text="The t-test showed a significant difference between groups.", claims=[]),
    dict(id="f_test", text="The omnibus test was reliable, F(1, 99) = 6.00, p = .016.", claims=[]),
    dict(id="correlation", text="The association was moderate, r(98) = .24, p = .016.", claims=[]),
    dict(id="chi_square", text="Frequencies differed, chi2(3) = 8.10, p = .044.", claims=[]),
]


def multi_claim_docs() -> List[Dict[str, Any]]:
    """Документы с двумя и более утверждениями — для порчи вида cross_claim."""
    return [c for c in CORPUS if len(c["claims"]) >= 2]
