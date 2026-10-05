"""Stage 4 loss/model correctness: Tobit direction & numerics, bounded MSE,
decidable ranking, arm dispatch. Direction tests are the proposal's key CI guard."""
import math
import torch

from losses.tobit import tobit_nll, sigma_from_raw
from losses.bounded import bounded_mse
from losses.ranking import decidable_ranking_loss, decidable_pairs_in_group
from losses.combined import compute_loss, ARMS
from models.mlp import AffinityMLP


def _grad_wrt_mu(y, censor, mu0=0.0, logsig=0.0):
    mu = torch.tensor([mu0], requires_grad=True)
    ls = torch.tensor([logsig])
    y_t = torch.tensor([y]); c_t = torch.tensor([censor])
    nll = tobit_nll(mu, ls, y_t, c_t, reduction="sum")
    nll.backward()
    return float(mu.grad)


def test_tobit_direction_right_censored_pushes_up():
    # right-censored (+1): true value > y. To lower NLL, mu should increase.
    # gradient dNLL/dmu > 0 means GD step (mu -= lr*grad) decreases mu -> WRONG.
    # We want mu to move UP, so gradient must be NEGATIVE.
    g = _grad_wrt_mu(y=5.0, censor=+1, mu0=5.0)
    assert g < 0, f"right-censored should pull mu up (grad<0), got {g}"


def test_tobit_direction_left_censored_pushes_down():
    # left-censored (-1): true value < y. mu should decrease -> gradient POSITIVE.
    g = _grad_wrt_mu(y=5.0, censor=-1, mu0=5.0)
    assert g > 0, f"left-censored should pull mu down (grad>0), got {g}"


def test_tobit_exact_matches_gaussian():
    mu = torch.tensor([2.0]); ls = torch.tensor([0.0]); y = torch.tensor([3.0]); c = torch.tensor([0])
    sigma = float(sigma_from_raw(torch.tensor(0.0)))
    got = float(tobit_nll(mu, ls, y, c, reduction="sum"))
    z = (3.0 - 2.0) / sigma
    exp = 0.5 * z * z + math.log(sigma) + 0.5 * math.log(2 * math.pi)
    assert abs(got - exp) < 1e-4


def test_tobit_numerics_extreme_finite():
    mu = torch.tensor([0.0]); ls = torch.tensor([0.0])
    for y in [-40.0, 40.0]:
        for c in [-1, 0, 1]:
            v = float(tobit_nll(mu, ls, torch.tensor([y]), torch.tensor([c]), reduction="sum"))
            assert math.isfinite(v)


def test_tobit_recovers_censored_upper_better_than_cast():
    """Fit a single mu to N right-censored obs at threshold t (true values >> t).
    Tobit should push mu above t; naive 'cast to =t' (MSE) would sit at t."""
    torch.manual_seed(0)
    t = 5.0
    y = torch.full((200,), t); c = torch.full((200,), 1)
    mu = torch.zeros(1, requires_grad=True); ls = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([mu, ls], lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        loss = tobit_nll(mu.expand(200), ls.expand(200), y, c)
        loss.backward(); opt.step()
    assert float(mu) > t + 0.3, f"Tobit mu should exceed censoring threshold, got {float(mu)}"


def test_bounded_mse_onesided():
    y = torch.tensor([5.0, 5.0, 5.0]); c = torch.tensor([-1, -1, +1])
    # left(-1): penalize only mu>y ; right(+1): penalize only mu<y
    mu_ok = torch.tensor([3.0, 5.0, 7.0])   # left below y ok; right above y ok
    mu_bad = torch.tensor([7.0, 7.0, 3.0])  # left above y bad; right below y bad
    assert float(bounded_mse(mu_ok, y, c)) < 1e-6
    assert float(bounded_mse(mu_bad, y, c)) > 1.0


def test_decidable_pairs_rule():
    # samples: 0 strong(right,+1,y=9); 1 weak(left,-1,y=5); 2 exact 7; 3 exact 6
    y = torch.tensor([9.0, 5.0, 7.0, 6.0]); c = torch.tensor([1, -1, 0, 0])
    ii, jj = decidable_pairs_in_group(y, c)
    pairs = set(zip(ii.tolist(), jj.tolist()))
    assert (0, 1) in pairs   # strong>weak decidable
    assert (2, 3) in pairs   # exact 7 > exact 6
    assert (2, 1) in pairs   # exact7 lo=7 > weak hi=5
    assert (0, 2) in pairs   # strong lo=9 > exact hi=7
    # weak(1) can't be the stronger side of anything with hi<=5 except nothing lower
    assert not any(i == 1 for i, _ in pairs)


def test_ranking_only_exact_filter():
    y = torch.tensor([9.0, 5.0, 7.0, 6.0]); c = torch.tensor([1, -1, 0, 0])
    tid = torch.zeros(4, dtype=torch.long); mu = torch.tensor([0.1, 0.2, 0.3, 0.4])
    _, s_all = decidable_ranking_loss(mu, y, c, tid, only_exact=False)
    _, s_obs = decidable_ranking_loss(mu, y, c, tid, only_exact=True)
    assert s_all["n_pairs"] > s_obs["n_pairs"]        # censored pairs add pairs
    assert s_obs["n_pairs"] == 1                       # only (2,3)
    assert 0.0 < s_all["frac_censored_involved"] <= 1.0


def test_ranking_recovers_order():
    # two exact samples 8>6; hinge should push mu[0]-mu[1] up past margin
    y = torch.tensor([8.0, 6.0]); c = torch.tensor([0, 0]); tid = torch.zeros(2, dtype=torch.long)
    mu = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        loss, _ = decidable_ranking_loss(mu, y, c, tid, margin=1.0)
        loss.backward(); opt.step()
    assert float(mu[0] - mu[1]) > 0.9


def test_all_arms_run():
    torch.manual_seed(1)
    N = 64
    mu = torch.randn(N, requires_grad=True); ls = torch.zeros(N, requires_grad=True)
    y = torch.randn(N) + 6; c = torch.randint(-1, 2, (N,)); tid = torch.randint(0, 4, (N,))
    for arm in ARMS:
        cc = torch.zeros_like(c) if arm in ("A_drop", "B_cast") else c
        loss, stats = compute_loss(arm, mu, ls, y, cc, tid, epoch=999)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)


def test_model_forward():
    net = AffinityMLP(in_dim=1504)
    x = torch.randn(8, 1504)
    mu, ls = net(x)
    assert mu.shape == (8,) and ls.shape == (8,)
