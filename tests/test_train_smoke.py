"""End-to-end A-F smoke (proposal Stage 0 requirement) on synthetic data:
each arm trains + evaluates VS enrichment without error and early-stops sanely."""
import numpy as np


def _cfg():
    from analysis.train_arm import load_cfg
    cfg = load_cfg()
    cfg["train"]["max_epochs"] = 3
    cfg["train"]["patience"] = 3
    cfg["loss"]["rank_warmup_epochs"] = 1
    cfg["eval"]["min_actives_per_target"] = 2
    return cfg


def test_all_arms_end_to_end():
    from analysis.train_arm import make_synthetic, ArmData, train_and_eval, ARMS
    cfg = _cfg()
    df, fp_tbl, emb_tbl, ci, ti = make_synthetic(cfg, n_targets=24, n_comp=250, seed=0)
    for arm in ARMS:
        tr = ArmData(df[df.split == "train"], fp_tbl, emb_tbl, ci, ti, arm)
        va = ArmData(df[df.split == "valid"], fp_tbl, emb_tbl, ci, ti, "eval")
        te = ArmData(df[df.split == "test"], fp_tbl, emb_tbl, ci, ti, "eval")
        assert len(tr) > 0 and len(te) > 0
        model, test_pt, info = train_and_eval(arm, tr, va, te, "cpu", cfg, 0)
        assert info["best_epoch"] >= 0
        # eval produced per-target metrics on >=1 test target
        assert len(test_pt) >= 1
        for col in ["EF1%", "BEDROC20", "AUC"]:
            assert col in test_pt.columns


def test_binarize_eval_directions():
    from analysis.train_arm import binarize_eval
    # exact strong active, exact weak inactive, right-cens strong active,
    # left-cens weak inactive, grey excluded
    y = np.array([8.0, 4.0, 7.0, 5.0, 5.5])
    c = np.array([0,   0,   1,   -1,  0])
    lab = binarize_eval(y, c, active_thr=6.0, inactive_thr=5.0)
    assert list(lab) == [1, 0, 1, 0, -1]
