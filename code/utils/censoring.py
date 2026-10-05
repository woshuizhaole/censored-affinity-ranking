"""Censoring sign convention + unit conversion — THE most error-prone piece.

We work in p-scale:  pAffinity = -log10(value[M]) = 9 - log10(value_nM)
so LARGER pAffinity == STRONGER binding.

The relation qualifier is recorded on the CONCENTRATION axis (Ki/IC50/Kd/EC50 in nM).
Under the -log10 transform the inequality FLIPS. Authoritative mapping
(matches proposal.md 4.2, verified against Svensson/uq4dd & SparseChem):

  original conc relation | meaning              | p-scale censor code c | true pAff constraint
  '='                    | exact                | 0                     | == y
  '>'  or '>='           | weak  (conc > X)     | -1 (LEFT-censored)    | true pAff <  y   (interval (-inf, y])
  '<'  or '<='           | strong(conc < X)     | +1 (RIGHT-censored)   | true pAff >  y   (interval [y, +inf))
  '~'                    | approximate          | drop (return None)

Tobit likelihood uses c with z=(y-mu)/sigma:
  c=0 : gaussian NLL
  c=-1: -log Phi(z)      (true value < y)
  c=+1: -log(1-Phi(z))   (true value > y)
"""
from __future__ import annotations
import math
from typing import Optional

# canonical censor codes
EXACT = 0
LEFT = -1   # true pAffinity < threshold  (weak binding, original conc '>')
RIGHT = +1  # true pAffinity > threshold  (strong binding, original conc '<')


def relation_to_censor(relation: str) -> Optional[int]:
    """Map a concentration-axis relation qualifier to a p-scale censor code.

    Returns None for approximate ('~') / unknown -> caller should drop.
    """
    if relation is None:
        return EXACT  # BindingDB: absent qualifier == exact
    r = str(relation).strip()
    if r in ("=", "==", ""):
        return EXACT
    if r in (">", ">="):
        return LEFT     # weak on conc -> small pAff -> left-censored in p-scale
    if r in ("<", "<="):
        return RIGHT    # strong on conc -> large pAff -> right-censored in p-scale
    return None         # '~' or anything else


def parse_bindingdb_value(raw: str):
    """Parse a BindingDB affinity cell (e.g. '>10000', '<0.5', '50', ' 12.3 ').

    Returns (value_nM: float, censor_code: int) or None if unparseable/approx.
    The '>' / '<' prefix is embedded in the value string (no separate column).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    rel = "="
    if s[:2] in (">=", "<="):
        rel = s[:2]
        s = s[2:].strip()
    elif s[0] in "><":
        rel = s[0]
        s = s[1:].strip()
    # some cells carry stray chars; keep only a leading number
    try:
        val = float(s)
    except ValueError:
        return None
    if val <= 0:
        return None
    c = relation_to_censor(rel)
    if c is None:
        return None
    return val, c


def nM_to_pAffinity(value_nM: float) -> float:
    """nM -> pAffinity (= -log10(M)). e.g. 1 nM -> 9, 10000 nM -> 5."""
    return 9.0 - math.log10(value_nM)


def pAffinity_to_nM(pAff: float) -> float:
    return 10.0 ** (9.0 - pAff)


def value_interval(pAff_threshold: float, censor: int):
    """True-value feasible interval [lo, hi] in p-scale for a (threshold, censor)."""
    if censor == EXACT:
        return (pAff_threshold, pAff_threshold)
    if censor == LEFT:
        return (float("-inf"), pAff_threshold)
    if censor == RIGHT:
        return (pAff_threshold, float("inf"))
    raise ValueError(f"bad censor code {censor}")
