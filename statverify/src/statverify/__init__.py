"""statverify — поиск внутренних противоречий в статистических утверждениях.

Цепочка вывода оканчивается на арифметике или сравнении строк, никогда
на суждении модели. Устройство и обоснования — docs/ARCHITECTURE.md.
"""

from .guard import admit, check_claim, check_slot
from .extract import extract
from .interval import Interval, from_reported
from .model import Claim, Document, Slot
from .relations import CATALOG, Relation, for_design
from .tstats import df_from, p_normal_limit, p_two_tailed, t_from_p
from .verify import Verdict, verify_claim

__version__ = "0.2.0"

__all__ = [
    "extract",
    "admit", "check_claim", "check_slot",
    "verify_claim", "Verdict",
    "Claim", "Document", "Slot",
    "Interval", "from_reported",
    "Relation", "CATALOG", "for_design",
    "p_two_tailed", "t_from_p", "df_from", "p_normal_limit",
]
