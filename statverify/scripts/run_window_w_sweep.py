"""Подбор WIDE_CANDIDATE_WINDOW измерением, а не назначением.

extract._claim_from_loose: если обычное окно (CANDIDATE_WINDOW=60) даёт
РОВНО одного кандидата p, отделённого от t-образца переносом строки,
поиск повторяется в более широком окне WIDE_CANDIDATE_WINDOW. Слишком
узкое W не достаёт настоящего p, спрятанного за вклинившейся колонкой
(остаётся один кандидат — опасное продвижение). Слишком широкое цепляет
p из следующего, не связанного предложения (кандидатов становится два
там, где должен быть один, — лишняя, но безопасная неоднозначность).

ДВЕ КРИВЫЕ:
  A. доля опасных исходов (неверное значение прошло охранника) на трёх
     реалистичных видах склейки колонок (tests/pdf_corrupt.py) — должна
     падать с ростом W.
  B. доля утраченных извлечений на синтетических ЛЕГИТИМНЫХ переносах
     строки внутри предложения — должна расти с ростом W, когда окно
     дотягивается до следующего, не связанного предложения со своим p.

ЧЕСТНОЕ ОГРАНИЧЕНИЕ, НЕ СКРЫТОЕ В КОММЕНТАРИИ: ни кривая A, ни кривая B
не измерены на живом PDF-тексте. corpus_real.py получен из HTML и не
содержит переносов строк вовсе — по нему нельзя откалибровать ни это
правило, ни признак "перенос = чужая колонка" из прошлого раунда. Обе
кривые строятся на синтетической модели вытекания колонки и синтетической
модели легитимного переноса (сконструированы в tests/pdf_corrupt.py), то
есть на ПРЕДПОЛОЖЕНИИ о том, как выглядит настоящий PDF-артефакт, а не
на измерении настоящего. Выбор W ниже обоснован формой ЭТИХ кривых, а
не проверен против реального PDF-извлечения, которого в проекте нет.
"""

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from corpus_real import CORPUS_REAL
from pdf_corrupt import (COLUMN_SPLICE_KINDS, column_splicing, ground_truth,
                         legit_linebreak, legit_linebreak_with_trailing_p,
                         resolve_after_corruption)

ex = importlib.import_module("statverify.extract")

RECORDS = [r for r in CORPUS_REAL if r.get("has_t_test")]


def danger_rate(w: int) -> float:
    ex.WIDE_CANDIDATE_WINDOW = w
    total = wrong = 0
    for kind in COLUMN_SPLICE_KINDS:
        for rec in RECORDS:
            gt = ground_truth(rec["text"])
            corrupted = column_splicing(rec["text"], gt, kind)
            if corrupted is None:
                continue
            total += 1
            _claim, admitted, values = resolve_after_corruption(
                corrupted, rec["t"], rec["df"])
            if values is None or not admitted:
                continue
            truth = {"t": rec["t"], "df": rec["df"], "p": rec["p"]}
            if any(abs(values[n] - truth[n]) > 1e-9 for n in ("t", "df", "p")):
                wrong += 1
    return wrong / total


def loss_rate_simple(w: int) -> float:
    ex.WIDE_CANDIDATE_WINDOW = w
    total = lost = 0
    for rec in RECORDS:
        gt = ground_truth(rec["text"])
        corrupted = legit_linebreak(rec["text"], gt)
        if corrupted is None:
            continue
        total += 1
        _claim, admitted, values = resolve_after_corruption(
            corrupted, rec["t"], rec["df"])
        if values is None or not admitted or abs(values["p"] - rec["p"]) > 1e-9:
            lost += 1
    return lost / total


def loss_rate_trailing(w: int) -> float:
    # pad_chars разнится по записи (0, 15, 30, ...), чтобы расстояние
    # до следующего p не схлопывалось в одну и ту же точку разрыва для
    # всех 43 записей сразу — иначе кривая была бы ступенькой, а не
    # формой, показывающей поведение на разных расстояниях.
    ex.WIDE_CANDIDATE_WINDOW = w
    total = lost = 0
    for i, rec in enumerate(RECORDS):
        gt = ground_truth(rec["text"])
        corrupted = legit_linebreak_with_trailing_p(
            rec["text"], gt, pad_chars=15 * i)
        if corrupted is None:
            continue
        total += 1
        _claim, admitted, values = resolve_after_corruption(
            corrupted, rec["t"], rec["df"])
        if values is None or not admitted or abs(values["p"] - rec["p"]) > 1e-9:
            lost += 1
    return lost / total


def main() -> int:
    # Мельче шаг вокруг перехода кривой A (около 100-105 — уточнено
    # отдельным прогоном), крупнее там, где обе кривые уже устоялись.
    ws = list(range(60, 100, 10)) + list(range(100, 140, 5)) + list(range(140, 601, 20))

    print("=" * 78)
    print("ПОДБОР WIDE_CANDIDATE_WINDOW: две кривые, обе на синтетической порче")
    print("=" * 78)
    print(f"{'W':>6}{'опасно (склейка, 3 вида)':>30}"
          f"{'потеряно (простой перенос)':>32}{'потеряно (перенос+чужой p)':>32}")

    rows = []
    for w in ws:
        d = danger_rate(w)
        l_simple = loss_rate_simple(w)
        l_trailing = loss_rate_trailing(w)
        rows.append((w, d, l_simple, l_trailing))
        print(f"{w:>6}{100.0*d:>29.1f}%{100.0*l_simple:>31.1f}%{100.0*l_trailing:>31.1f}%")

    # Выбор в два шага, не один "наименьшее W с нулём".
    # 1. Порог опасности: наименьшее W, при котором кривая A уже 0.0% —
    #    настоящий p, спрятанный за вклинившейся колонкой, гарантированно
    #    виден. Меньше этого расширять некуда — там ещё есть опасные
    #    исходы.
    # 2. Внутри плато: сразу после порога кривая C (потери от чужого p
    #    дальше по тексту) не растёт МГНОВЕННО — держится на одном
    #    минимальном значении до какого-то W, и только потом начинает
    #    расти. Взять именно ПОРОГ (наименьшее W) было бы хрупко: реальные
    #    фрагменты склейки не имеют фиксированной длины, а порог подобран
    #    по конкретным длинам трёх синтетических видов. Конец плато даёт
    #    запас против этой изменчивости БЕСПЛАТНО — цена (кривая C) там
    #    та же, что и на самом пороге.
    zero_danger = [w for w, d, _, _ in rows if d == 0.0]
    threshold = min(zero_danger) if zero_danger else max(ws)
    min_loss = min(l for w, _, _, l in rows if w >= threshold)
    plateau = [w for w, _, _, l in rows if w >= threshold and l == min_loss]
    chosen = max(plateau)
    chosen_row = next(r for r in rows if r[0] == chosen)

    print("-" * 78)
    print(f"Порог опасности (кривая A впервые 0.0%): W = {threshold}.")
    print(f"Плато минимальных потерь ({100.0*min_loss:.1f}%) держится до "
          f"W = {chosen} включительно, дальше кривая C начинает расти.")
    print(f"Выбрано W = {chosen}: конец этого плато — тот же 0.0% по кривой A "
          f"и та же минимальная {100.0*chosen_row[3]:.1f}% по кривой C, что и "
          f"на самом пороге {threshold}, но с запасом на случай, что реальные "
          f"фрагменты склейки колонок длиннее трёх синтетических видов, "
          f"которыми измерена кривая A.")
    print()
    print("ОГРАНИЧЕНИЕ (см. докстринг модуля): обе кривые построены на "
          "синтетической порче — реального PDF-текста с переносами строк "
          "в этом проекте нет, поэтому W не откалибровано на живых данных, "
          "только на модели того, как вытекание колонки и легитимный "
          "перенос строки должны выглядеть.")
    return chosen


if __name__ == "__main__":
    main()
    sys.exit(0)
