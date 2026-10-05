#!/usr/bin/env python
"""improvement.md #1 analysis (DECISIVE): does tuned-lambda ranking on a STRONG base (bounded=C's loss,
cast=B's loss) beat C? Val-select lambda per G variant (val BEDROC20, 51-target val set — NO test peeking),
then paired Wilcoxon vs C_bounded + effect size + bootstrap CI + Holm. Also G_bounded_rankdec vs
G_bounded_rankobs (does adding CENSORED decidable pairs beat exact-only ranking, on the strong base)."""
import glob, os, re, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
RNG = np.random.default_rng(20260708)
LAMS = ["1.0", "3.0", "10.0", "30.0", "100.0", "300.0"]
# val plateaus at lambda>=30; doc HEADLINES the conservative plateau-knee lambda=30 (base still contributes
# ~3%), and also reports the val-argmax as robustness. Persist BOTH to the CSV so the headline is traceable.
REPORT_EXTRA = {"G_bounded_rankdec": ["30.0"]}
GVAR = ["G_bounded_rankdec", "G_cast_rankdec"]


def val_bedroc(lam, scheme, arm, seed):
    lf = "%s/results/gtuned/l%s/%s/log_%s_seed%d.log" % (PROJ, lam, scheme, arm, seed)
    if not os.path.exists(lf):
        return np.nan
    vals = [float(m) for m in re.findall(r"val BEDROC20=([0-9.]+)", open(lf).read())]
    return max(vals) if vals else np.nan


def pertarget(root, scheme, arm, seeds=range(1, 6)):
    fs = []
    for s in seeds:
        p = "%s/%s/%s/pertarget_%s_seed%d.csv" % (PROJ, root, scheme, arm, s)
        if os.path.exists(p):
            fs.append(pd.read_csv(p))
    if not fs:
        return None
    d = pd.concat(fs, ignore_index=True)
    return d.groupby("target")[["EF1%", "BEDROC20", "AUC"]].mean()


def gt_pertarget(lam, scheme, arm):
    fs = glob.glob("%s/results/gtuned/l%s/%s/pertarget_%s_seed*.csv" % (PROJ, lam, scheme, arm))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d.groupby("target")[["EF1%", "BEDROC20", "AUC"]].mean()


def bootci(d, n=5000):
    idx = RNG.integers(0, len(d), size=(n, len(d))); mu = d[idx].mean(1)
    return np.percentile(mu, 2.5), np.percentile(mu, 97.5)


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min((m - rank) * p[i], 1.0)); adj[i] = run
    return adj


# ---- 1) val-select lambda per variant (on random val) ----
print("=== VAL selection (mean val BEDROC20 over 5 seeds, random; 51-target val set) ===")
best = {}
for arm in GVAR:
    print(" %s:" % arm)
    scores = {}
    for lam in LAMS:
        v = np.nanmean([val_bedroc(lam, "random", arm, s) for s in range(1, 6)])
        scores[lam] = v
        print("   lambda=%s  val BEDROC20=%.4f" % (lam, v))
    best[arm] = max(scores, key=lambda k: scores[k])
    print("   -> val-best lambda = %s" % best[arm])

# ---- 2) paired test vs C_bounded at val-best lambda (both splits) + effect/CI/Holm ----
C = {sc: pertarget("results/mainline", sc, "C_bounded") for sc in ["random", "scaffold"]}
B = {sc: pertarget("results/mainline", sc, "B_cast") for sc in ["random", "scaffold"]}
Ftu = {sc: pertarget("results/p1/f_l3m05", sc, "F_tobit_rankdec") for sc in ["random", "scaffold"]}
rows = []
print("\n=== TEST: tuned-G vs C_bounded / B_cast (per-target paired, val-best lambda) ===")
print("%-20s %-8s %-9s %7s %7s %8s %18s %9s" % ("arm(lam)", "split", "metric", "hi", "lo", "delta", "95%CI", "p1"))
for arm in GVAR:
    for lam in dict.fromkeys([best[arm]] + REPORT_EXTRA.get(arm, [])):  # val-argmax + headline lambda(s), deduped
      for sc in ["random", "scaffold"]:
        G = gt_pertarget(lam, sc, arm)
        if G is None:
            continue
        for ref_name, ref in [("C_bounded", C[sc]), ("B_cast", B[sc])]:
            com = sorted(set(G.index) & set(ref.index))
            for m in ["BEDROC20", "AUC", "EF1%"]:
                x = G.loc[com, m].values; y = ref.loc[com, m].values; d = x - y
                lo, hi = bootci(d)
                try:
                    _, p = wilcoxon(x, y, alternative="greater")
                except Exception:
                    p = np.nan
                rows.append({"arm": arm, "lam": lam, "split": sc, "ref": ref_name, "metric": m,
                             "hi": x.mean(), "lo": y.mean(), "delta": d.mean(), "ci_lo": lo, "ci_hi": hi, "p1": p})
                print("%-20s %-8s %-9s %7.3f %7.3f %8.4f  [%7.4f,%7.4f] %9.4g %s"
                      % ("%s(l%s)" % (arm.replace("G_", "G_").replace("_rank", "_r"), lam), sc, m,
                         x.mean(), y.mean(), d.mean(), lo, hi, p, "*" if p < 0.05 else ""))
df = pd.DataFrame(rows)
# Holm within (each arm's vs-C BEDROC20+AUC family across splits)
for arm in GVAR:
    fam = df[(df.arm == arm) & (df.ref == "C_bounded") & (df.metric.isin(["BEDROC20", "AUC"]))]
    if len(fam):
        df.loc[fam.index, "p_holm_vsC"] = holm(fam["p1"].values)
df.to_csv(PROJ + "/results/tables/gtuned_paired_effect.csv", index=False)

# ---- 3) censored-pair value: G_bounded_rankdec vs G_bounded_rankobs (control, lambda=3) ----
print("\n=== control: G_bounded_rankdec vs G_bounded_rankobs (does CENSORED pair help on strong base?) lambda=3 ===")
for sc in ["random", "scaffold"]:
    dec = gt_pertarget("3.0", sc, "G_bounded_rankdec")
    obs = None
    fs = glob.glob("%s/results/gtuned/rankobs_l3/%s/pertarget_G_bounded_rankobs_seed*.csv" % (PROJ, sc))
    if fs:
        obs = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).groupby("target")[["EF1%", "BEDROC20", "AUC"]].mean()
    if dec is None or obs is None:
        print("  %s: missing" % sc); continue
    com = sorted(set(dec.index) & set(obs.index))
    for m in ["BEDROC20", "AUC"]:
        x = dec.loc[com, m].values; y = obs.loc[com, m].values
        try:
            _, p = wilcoxon(x, y, alternative="greater")
        except Exception:
            p = np.nan
        print("  %s %-9s dec=%.3f obs=%.3f delta=%+.4f p=%.4g %s" % (sc, m, x.mean(), y.mean(), (x - y).mean(), p, "*" if p < 0.05 else ""))
print("\nwrote -> results/tables/gtuned_paired_effect.csv")
