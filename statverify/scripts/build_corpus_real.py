"""Сборка corpus_real.py из дословно извлечённых предложений.

Разметка вычисляется арифметикой, а не проставляется рукой:
двусторонний p считается из (t, df) и сравнивается с сообщённым по той же
интервальной конвенции, что в interval.py. Поэтому «согласовано» здесь —
не чьё-то суждение, а следствие расчёта.

Держит происхождение воспроизводимым: чтобы получить tests/corpus_real.py
заново из тех же сырых цитат (RAW ниже), запусти этот скрипт — он пересчитает
разметку и разбиение dev/holdout тем же кодом и тем же seed.
"""

import json
import random
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from statverify.tstats import p_two_tailed

# (id, doi, raw-строка как отдана источником)
RAW = [
 # --- PLOS ONE 10.1371/journal.pone.0267297 ---
 ("A01","10.1371/journal.pone.0267297","relatedness measures for typical pairings (*M* = .23, *SD* = .14) were significantly larger than those for unusual pairings (*M* = -.7, *SD* = .06, *t*(15) = 3.82, *p* = .002)"),
 ("A02","10.1371/journal.pone.0267297","children, *M* = .16, *SD* = .10, *t*(27) = -2.28, *p* = .030, α = .05, *d* = -0.43"),
 ("A03","10.1371/journal.pone.0267297","children: *M* = .40, *SD* = .17, *t*(27) = 6.17, *p* = .001, α = .05, *d* = 1.16"),
 ("A04","10.1371/journal.pone.0267297","adults: *M* = .37, *SD* = .20, *t*(29) = 4.67, *p* = .002, α = .05, *d =* 0.85"),
 ("A05","10.1371/journal.pone.0267297","children: *t*(30) = 7.07, *p* = .001, α = .016, *d* = 1.27"),
 ("A06","10.1371/journal.pone.0267297","adults: *t*(32) = 7.83, *p* = .001, α = .016, *d* = 1.36"),
 ("A07","10.1371/journal.pone.0267297","children looked to the SCR at a rate above chance (*t*(30) = 5.37, *p* = .001, α = .05, *d* = 0.964)"),
 ("A08","10.1371/journal.pone.0267297","children: *t*(30) = 2.13, *p* = .045, α = 0.025, *d* = 0.38"),
 ("A09","10.1371/journal.pone.0267297","adults: *t*(32) = 3.27, *p* = .003, α = 0.05, *d* = 0.57"),
 ("A10","10.1371/journal.pone.0267297","looked at a rate above chance to the previously mentioned objects (*M* = .25, *SD* = .11, *t*(32) = 2.499, *p* = .020, α = .05, *d* = 0.43)"),

 # --- PLOS ONE 10.1371/journal.pone.0058546 ---
 ("B01","10.1371/journal.pone.0058546","[*t*(15) = −5.80, *p*<.001, *d* = −2.19]"),
 ("B02","10.1371/journal.pone.0058546","[*t*(15) = −5.42, *p*<.001, *d* = −2.26]"),
 ("B03","10.1371/journal.pone.0058546","[*t*(15) = −3.55, *p* = .003, *d* = −1.53]"),
 ("B04","10.1371/journal.pone.0058546","[*t*(15) = −4.11, *p* = .001, *d* = −1.71]"),
 ("B05","10.1371/journal.pone.0058546","[*t*(15) = −2.78, *p* = .014, *d* = −1.02]"),
 ("B06","10.1371/journal.pone.0058546","[*t*(14) = 3.95, *p =* .001, *d =* 1.54]"),
 ("B07","10.1371/journal.pone.0058546","[*t*(14) = 2.50, *p* = .025, *d* = 0.91]"),
 ("B08","10.1371/journal.pone.0058546","[*t*(14) = 2.33, *p* = .035, *d* = 0.85]"),
 ("B09","10.1371/journal.pone.0058546","[*t*(14) = 6.38, *p*<.001, *d* = 2.49]"),
 ("B10","10.1371/journal.pone.0058546","[*t*(14) = 4.92, *p* = .001, *d* = 1.81]"),
 ("B11","10.1371/journal.pone.0058546","[*t*(14) = 4.40, *p =* .001, *d* = 1.91]"),
 ("B12","10.1371/journal.pone.0058546","[*t*(13) = 2.38, *p* = .034, *d* = 1.23]"),
 ("B13","10.1371/journal.pone.0058546","[*t*(13) = 3.19, *p* = .007, *d* = 1.21]"),
 ("B14","10.1371/journal.pone.0058546","[*t*(13) = 2.78, *p* = .016, *d* = 1.16]"),
 ("B15","10.1371/journal.pone.0058546","[*t*(13) = 4.00, *p* = .002, *d* = 1.67]"),
 ("B16","10.1371/journal.pone.0058546","[*t*(13) = 2.64, *p* = .02, *d* = 1.09]"),
 ("B17","10.1371/journal.pone.0058546","[*t*(13) = 3.16, *p* = .008, *d* = 1.21]"),
 ("B18","10.1371/journal.pone.0058546","[*t*(13) = 2.71, *p* = .018, *d* = 1.02]"),
 ("B19","10.1371/journal.pone.0058546","[*t*(13) = 3.04, *p* = .009, *d* = 1.34]"),
 ("B20","10.1371/journal.pone.0058546","[*t*(13) = −2.43, *p* = .03, *d* = −1.32]"),
 ("B21","10.1371/journal.pone.0058546","[*t*(13) = −2.81, *p* = .015, *d* = −2.4]"),
 ("B22","10.1371/journal.pone.0058546","[*t*(13) = −2.80, *p* = .015, *d* = −1.35]"),
 ("B23","10.1371/journal.pone.0058546","[*t*(14) = −3.13, *p =* .007, *d* = −1.19]"),
 ("B24","10.1371/journal.pone.0058546","[*t*(14) = −3.29, *p* = .005, *d* = −1.21]"),
 ("B25","10.1371/journal.pone.0058546","[*t*(13) = −2.28, *p* = .04, *d* = -.90]"),
 ("B26","10.1371/journal.pone.0058546","[*t*(13) = −1.91, *p* = .078, *d* = .79]"),
 ("B27","10.1371/journal.pone.0058546","[*t*(9) = -3.36, *p* = .008, *d* = −1.51]"),
 ("B28","10.1371/journal.pone.0058546","[*t*(11) = −3.24, *p* = .008, *d* = −1.78]"),
 ("B29","10.1371/journal.pone.0058546","[*t*(10) = -2.63, *p* = .025, *d* = −1.13]"),

 # --- PLOS ONE 10.1371/journal.pone.0309006 ---
 ("C01","10.1371/journal.pone.0309006","No practice effects were found for Reaction Time (online: M = 344.62 ± 69.02; in-person: M = 359.64 ± 95.18; t(29) = -0.503, p = 0.619, d = 0.18)"),
 ("C02","10.1371/journal.pone.0309006","Go/No-Go (online: M = 1.22 ± 1.64, in-person: M = 1.42 ± 1.36; t(30) = -0.596, p = 0.556, d = 0.13)"),
 ("C03","10.1371/journal.pone.0309006","Allocentric Orientation (online: M = 3.11 ± 1.57, in-person: M = 3.11 ± 1.62; t(26) = 0.107, p = 0.915, d = 0.01)"),
 ("C04","10.1371/journal.pone.0309006","Egocentric Orientation (online: M = 51.31 ± 34.16, in-person: M = 56.52 ± 36.37; t(28) = -0.684, p = 0.500, d = 0.15)"),

 # --- PLOS ONE 10.1371/journal.pone.0084698 (без тестовой статистики) ---
 ("D01","10.1371/journal.pone.0084698","Endothelial function (% forearm blood flow above saline) was not changed with progesterone (487±189%, n = 18) compared with placebo (408±278%, n = 16) (95% CI diff [−74 to 232], P = 0.30)."),
 ("D02","10.1371/journal.pone.0084698","High-density lipoprotein was lower (−0.14 mmol/L, P = 0.001) on progesterone compared with placebo."),
 ("D03","10.1371/journal.pone.0084698","The difference between progesterone and placebo arms was 0.1 (95% CI: −0.3 to 0.6; p = 0.53)"),
 ("D04","10.1371/journal.pone.0084698","D-dimer was available for 43 women (progesterone = 24, placebo = 19). Results were stable within and not significantly different between progesterone and placebo (95% CI: −7.3 to 113.2 FEU)"),
 ("D05","10.1371/journal.pone.0084698","Fasting glucose, hs-C-reactive protein, albumin and D-dimer changes were all comparable to placebo."),
 ("D06","10.1371/journal.pone.0084698","Framingham General Cardiovascular Risk Profile scores were initially low and remained low with progesterone therapy and not statistically different from placebo."),
]

SOURCES = {
 "10.1371/journal.pone.0267297": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267297",
 "10.1371/journal.pone.0058546": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0058546",
 "10.1371/journal.pone.0309006": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0309006",
 "10.1371/journal.pone.0084698": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0084698",
}

PAT = re.compile(r"t\s*\(\s*(\d+(?:\.\d+)?)\s*\)\s*=\s*([-−–+]?\d*\.?\d+)"
                 r".{0,12}?[pP]\s*([<>=])\s*(\d*\.\d+|\d+)")


def plain(s: str) -> str:
    """Снять markdown-курсив: так текст выглядит для извлекателя из PDF/HTML."""
    return s.replace("*", "")


def rhu(x, d):
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def annotate(text):
    m = PAT.search(text)
    if not m:
        return None
    df = float(m.group(1))
    t = float(m.group(2).replace("−", "-").replace("–", "-").lstrip("+"))
    rel = m.group(3)
    p_lit = m.group(4)
    p = float("0" + p_lit if p_lit.startswith(".") else p_lit)
    dec = len(p_lit.split(".")[1]) if "." in p_lit else 0
    calc = p_two_tailed(t, df)

    # Интервальная разметка: t тоже округлён, и не учитывать это значит
    # над-флагить. Реализация здесь своя, verify_claim не вызывается —
    # иначе получилась бы проверка системы самой собой.
    t_dec = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
    t_half = 0.5 * 10 ** -t_dec
    t_abs = abs(t)
    p_from_t = sorted((p_two_tailed(max(t_abs - t_half, 0.0), df),
                       p_two_tailed(t_abs + t_half - 1e-12, df)))
    derived_lo, derived_hi = p_from_t

    if rel == "=":
        half = 0.5 * 10 ** -dec
        rep_lo, rep_hi = p - half, p + half
    elif rel == "<":
        rep_lo, rep_hi = 0.0, p
    else:
        rep_lo, rep_hi = p, 1.0
    consistent = derived_lo <= rep_hi and rep_lo <= derived_hi

    # Конвенция нижнего порога: SPSS печатает 'p = .001' для всего, что
    # меньше. Формально это расхождение, по сути — не ошибка авторов.
    #
    # Порог выведен, а не перечислен списком: раньше здесь стоял список
    # (.05, .01, .001, .0001), и это была ошибка — .05 не порог печати,
    # это уровень значимости, и 'p = .05' при вычисленном .02 — либо
    # опечатка, либо подгонка под уровень значимости, самый интересный
    # класс находок, который список тихо гасил бы под видом технической
    # печати. Порог печати — это наименьшее ПРЕДСТАВИМОЕ ненулевое число
    # при записанной точности, 10^-dec: воспроизводит .001/.0001/.01 без
    # перечисления и само исключает .05 и .002 (см. verify.py, то же
    # правило продублировано намеренно — здесь независимая реализация
    # разметки, verify_claim не вызывается).
    floor = (rel == "=" and not consistent and calc < p
             and abs(p - 10.0 ** -dec) < 1e-12)
    return dict(t=t, df=df, t_decimals=t_dec, p=p, p_relation=rel,
                p_decimals=dec, p_computed=round(calc, 8),
                p_derived_lo=round(derived_lo, 8), p_derived_hi=round(derived_hi, 8),
                consistent=consistent, floor_convention=floor)


rows = []
for cid, doi, raw in RAW:
    text = plain(raw)
    ann = annotate(text)
    rows.append(dict(id=cid, doi=doi, url=SOURCES[doi], text=text, raw=raw,
                     has_t_test=ann is not None,
                     design="unknown",          # из предложения не определяется
                     tail="unknown",
                     **(ann or {})))

rnd = random.Random(20260910)
ids = [r["id"] for r in rows]
rnd.shuffle(ids)
half = len(ids) // 2
dev, hold = set(ids[:half]), set(ids[half:])
for r in rows:
    r["split"] = "dev" if r["id"] in dev else "holdout"

if __name__ == "__main__":
    out_path = _ROOT / "corpus_real.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    n_t = sum(r["has_t_test"] for r in rows)
    n_cons = sum(1 for r in rows if r["has_t_test"] and r["consistent"])
    n_floor = sum(1 for r in rows if r["has_t_test"] and r.get("floor_convention"))
    n_incons = n_t - n_cons
    print(f"всего утверждений         {len(rows)}")
    print(f"  с тройкой (t, df, p)    {n_t}")
    print(f"  без тестовой статистики {len(rows) - n_t}")
    print(f"  согласовано             {n_cons}")
    print(f"  расходится              {n_incons}")
    print(f"    из них порог 'p=.001' {n_floor}")
    print(f"    настоящих кандидатов  {n_incons - n_floor}")
    print(f"  dev / holdout           {sum(r['split']=='dev' for r in rows)} / "
          f"{sum(r['split']=='holdout' for r in rows)}")
    print()
    for r in rows:
        if r["has_t_test"] and not r["consistent"]:
            tag = "порог" if r["floor_convention"] else "КАНДИДАТ"
            print(f"  {r['id']}  t({r['df']:g})={r['t']}, p{r['p_relation']}{r['p']}"
                  f"  вычислено {r['p_computed']:.6f}  [{tag}]")
