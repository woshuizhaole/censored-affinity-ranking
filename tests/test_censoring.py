"""Sign-convention unit test for censoring (proposal's key CI requirement).

Verifies the concentration->p-scale inequality FLIP is not written backward,
across both nM cells and hand-labelled pAffinity thresholds.
"""
import math
from utils.censoring import (
    relation_to_censor, parse_bindingdb_value, nM_to_pAffinity,
    value_interval, EXACT, LEFT, RIGHT,
)


def test_relation_mapping():
    assert relation_to_censor("=") == EXACT
    assert relation_to_censor(">") == LEFT     # weak conc -> left in p-scale
    assert relation_to_censor(">=") == LEFT
    assert relation_to_censor("<") == RIGHT    # strong conc -> right in p-scale
    assert relation_to_censor("<=") == RIGHT
    assert relation_to_censor("~") is None     # dropped
    assert relation_to_censor(None) == EXACT


def test_bindingdb_value_parse():
    assert parse_bindingdb_value("50") == (50.0, EXACT)
    assert parse_bindingdb_value(">10000") == (10000.0, LEFT)   # weak -> left
    assert parse_bindingdb_value("<0.5") == (0.5, RIGHT)        # strong -> right
    assert parse_bindingdb_value(" 12.3 ") == (12.3, EXACT)
    assert parse_bindingdb_value(">=100") == (100.0, LEFT)
    assert parse_bindingdb_value("") is None
    assert parse_bindingdb_value("n.d.") is None
    assert parse_bindingdb_value("-5") is None


def test_pscale_values():
    assert abs(nM_to_pAffinity(1.0) - 9.0) < 1e-9
    assert abs(nM_to_pAffinity(10000.0) - 5.0) < 1e-9
    assert abs(nM_to_pAffinity(50.0) - (9.0 - math.log10(50.0))) < 1e-9


def test_intervals_direction():
    # IC50 > 10000 nM (weak): p-scale threshold 5, true pAff < 5 -> (-inf, 5]
    v, c = parse_bindingdb_value(">10000")
    y = nM_to_pAffinity(v)
    assert c == LEFT and abs(y - 5.0) < 1e-9
    lo, hi = value_interval(y, c)
    assert lo == float("-inf") and abs(hi - 5.0) < 1e-9

    # Ki < 1 nM (strong): p-scale threshold 9, true pAff > 9 -> [9, +inf)
    v, c = parse_bindingdb_value("<1")
    y = nM_to_pAffinity(v)
    assert c == RIGHT and abs(y - 9.0) < 1e-9
    lo, hi = value_interval(y, c)
    assert abs(lo - 9.0) < 1e-9 and hi == float("inf")


def test_decidable_pair_rule():
    # i: strong <1nM (right, [9,inf)) ; j: weak >10uM (left, (-inf,5]) -> i > j decidable
    _, ci = parse_bindingdb_value("<1"); yi = nM_to_pAffinity(1.0)
    _, cj = parse_bindingdb_value(">10000"); yj = nM_to_pAffinity(10000.0)
    lo_i, hi_i = value_interval(yi, ci)
    lo_j, hi_j = value_interval(yj, cj)
    assert lo_i > hi_j            # decidable: i strictly stronger than j
