# Censored-Consistent Affinity Ranking

Reference implementation for the study *"Censoring rate and decoy choice govern the
screening value of censored bioactivity."*

Public bioactivity archives report a large fraction of their measurements as
inequalities: `IC50 > 10000 nM`, `Ki < 0.5 nM`. These are **censored
observations**, not missing data and not point values. Common practice either
discards them or casts them to the reported bound, and both choices distort the
learned affinity scale. This repository implements an alternative that keeps
censored records in their correct measurement-theoretic form, and evaluates what
that buys for virtual screening under early-enrichment metrics.

## Method

Two components, combined into a single training objective.

**1. Tobit censored-Gaussian likelihood.** A two-head network predicts a mean
`mu` and a raw scale from which `sigma = softplus(raw) + floor`. With
`z = (y - mu) / sigma` on the p-scale (`pAff = 9 - log10(value_nM)`), the
per-sample negative log-likelihood is selected by a censor code `c`:

| `c` | meaning | NLL term |
|----:|---------|----------|
| `0`  | exact (`= y`)        | `0.5 z² + log sigma + 0.5 log 2π` |
| `-1` | left (true `< y`)    | `-log Φ(z)` |
| `+1` | right (true `> y`)   | `-log(1 - Φ(z))` |

The tail terms use `torch.special.log_ndtr`, which stays finite out to `|z| ~ 40`
where a naive `log(1 - cdf + eps)` collapses. A left-censored record therefore
contributes the probability mass below its bound rather than a fabricated point
target, and a right-censored record the mass above it.

**2. Censoring-decidable pairwise ranking.** Virtual screening consumes a
ranking, so the objective adds a margin hinge over compound pairs *within the
same target*. The key point is which pairs are admissible. A pair is **decidable**
when the two censored intervals do not overlap, so their true order is known
regardless of where each true value falls inside its interval. That admits many
censored-vs-censored and censored-vs-exact pairs that an exact-only scheme throws
away, while never asserting an order the data cannot support. Pair enumeration and
the decidability rule live in `code/utils/censoring.py`; the hinge is in
`code/losses/ranking.py`.

### Training arms

`code/losses/combined.py` exposes the nine arms used throughout, as literal
strings accepted by `train_arm.py --arm`:

| Arm | Base loss | Ranking term |
|-----|-----------|--------------|
| `A_drop`            | MSE, censored rows dropped        | — |
| `B_cast`            | MSE, censored cast to the bound   | — |
| `C_bounded`         | one-sided bounded squared hinge   | — |
| `D_tobit`           | Tobit                             | — |
| `E_tobit_rankobs`   | Tobit                             | exact-exact pairs only |
| `F_tobit_rankdec`   | Tobit                             | decidable pairs, censored included |
| `G_bounded_rankdec` | bounded                           | decidable pairs |
| `G_cast_rankdec`    | cast MSE                          | decidable pairs |
| `G_bounded_rankobs` | bounded                           | exact-only pairs |

`A_drop` and `B_cast` are the two prevailing baselines. `F_tobit_rankdec` is the
full method. `E` isolates the contribution of censored pairs against `F`, and the
`G` arms separate "censored-aware base loss" from "censored-aware ranking" so the
two effects are not confounded.

Evaluation labels are constructed identically for every arm (actives at
`pAff >= 6.0`, inactives at `pAff <= 5.0`, the grey zone discarded), and the decoy
pool is drawn from a fixed RNG so the screening library is byte-identical across
arms and seeds. Model selection is validation `BEDROC20` with patience 20, never
test. Metrics are EF at 0.5/1/5 %, BEDROC at α = 20/80.5/160.9, RIE and AUC
(`code/eval/enrichment.py`).

## Installation

```bash
git clone https://github.com/woshuizhaole/censored-affinity-ranking.git
cd censored-affinity-ranking

conda create -n censored python=3.10.20
conda activate censored
conda install -c conda-forge rdkit=2026.3.3 pyarrow=24.0.0
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Versions in `requirements.txt` are the ones used to produce the reported numbers:
Python 3.10.20, PyTorch 2.4.1+cu118, RDKit 2026.03.3, NumPy 1.26.4, SciPy 1.15.2,
pandas 2.3.3, scikit-learn 1.7.2, statsmodels 0.14.6, fair-esm 2.0.0,
chembl_structure_pipeline 1.2.4, OmegaConf 2.3.0.

**MMseqs2** is a separate native binary, used for target-disjoint clustering and
for sequence-identity leakage filters. Install from
[github.com/soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2) or
`conda install -c conda-forge -c bioconda mmseqs2`, and make sure `mmseqs` is on
`PATH`. The results were produced with a source build at commit `771ee23`.

**ESM-2 checkpoint.** Target sequences are embedded with the frozen
`esm2_t12_35M_UR50D` model (480-d mean-pooled representation), loaded from a
**local** `.pt` file via `esm.pretrained.load_model_and_alphabet_local`. Download it
once from the [fair-esm release assets](https://github.com/facebookresearch/esm)
to `data/raw/esm2_t12_35M_UR50D.pt`.

No packaging step is needed. Scripts put `code/` on `sys.path` themselves via
`__file__`, so they are runnable from any working directory. Run `pytest` from the
repository root so the top-level `conftest.py` is picked up — it exists because the
directory `code/` would otherwise shadow Python's stdlib `code` module.

```bash
pytest tests/ -v
```

All five test modules are self-contained and need no downloaded data. They cover
the censoring sign conventions, Tobit gradient direction and numerical stability,
decidable-pair enumeration, the enrichment metrics, and an end-to-end synthetic
train/eval pass over all nine arms.

## Data

No data is redistributed here. Everything is rebuilt from public sources by the
pipeline scripts.

| Dataset | Source | Expected location |
|---|---|---|
| BindingDB (full TSV) | <https://www.bindingdb.org/bind/chemsearch/marvin/Download.jsp> | `data/raw/bindingdb/BindingDB_All.tsv` |
| ChEMBL 37 (SQLite) | <https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/> | `data/raw/chembl/chembl_37/chembl_37_sqlite/chembl_37.db` |
| DUD-E (all targets) | <https://dude.docking.org/db/subsets/all/all.tar.gz> | `data/external/dude/all.tar.gz` |
| LIT-PCBA | <https://drugdesign.unistra.fr/LIT-PCBA/> | `data/external/litpcba/` |
| ESM-2 `esm2_t12_35M_UR50D` | <https://github.com/facebookresearch/esm> | `data/raw/esm2_t12_35M_UR50D.pt` |

Censoring is read from the relation column (`>`, `<`, `=`) rather than inferred.
BindingDB parsing keys on header *names*, not column indices, since indices shift
between releases. Duplicate `(compound, target, endpoint)` triples are collapsed
with a censoring-aware rule: any exact measurement wins and takes the median;
all-left collapses to the minimum bound; all-right to the maximum; a triple with
conflicting left and right bounds and no exact value is dropped.

### Configure paths before running

`configs/base.yaml` has a single project root that every other path interpolates
from. Set it first:

```yaml
proj: /path/to/censored-affinity-ranking
```

Several later-stage scripts (`22`, `23`, `24`, `25`, `26`, `27`, `28`, and the
`diag_*` set) carry a module-level `PROJ` constant with the same root and take no
CLI override, so edit those to match. The two orchestrators `06_run_grid.py` and
`12_censoring_stress.py` launch workers through `PYBIN`, which points at
`<repo>/env/conda/bin/python`; repoint it at your interpreter. `22_dude_batch_5seed.py`
and `26_litpcba_score.py` likewise expect `mmseqs` at `<PROJ>/env/conda/bin/mmseqs`.

## Reproducing

### Build the dataset

```bash
# 1. Parse the archives into censored measurement tables
python code/pipeline/01a_parse_bindingdb.py \
    --tsv data/raw/bindingdb/BindingDB_All.tsv --out data/interim
python code/pipeline/01c_extract_chembl.py \
    --db data/raw/chembl/chembl_37/chembl_37_sqlite/chembl_37.db --out data/interim

# 2. Standardize structures, then assemble per-endpoint datasets
python code/pipeline/02a_standardize_smiles.py \
    --inputs data/interim/bindingdb_measurements.parquet \
    --out data/interim/stdmap_bindingdb.parquet --njobs 48
python code/pipeline/02b_build_datasets.py \
    --measurements data/interim/bindingdb_measurements.parquet \
    --targets data/interim/bindingdb_targets.parquet \
    --stdmap data/interim/stdmap_bindingdb.parquet \
    --source bindingdb --out-dir data/processed

# 3. Features: frozen ESM-2 target embeddings and ECFP4 ligand fingerprints
python code/pipeline/03a_esm_embed.py \
    --targets data/interim/bindingdb_targets.parquet \
    --ckpt data/raw/esm2_t12_35M_UR50D.pt \
    --out data/features/target_emb_esm2_35M.npz --device cuda
python code/pipeline/03b_fingerprints.py \
    --inputs data/processed/bindingdb_IC50_censored.parquet \
    --out data/features/ligand_fp.npz --bits 1024 --radius 2

# 4. Splits: target-disjoint (MMseqs2, 30% identity), scaffold, random
python code/pipeline/05_make_splits.py \
    --dataset data/processed/bindingdb_IC50_censored.parquet \
    --name bindingdb_IC50 --out data/processed/splits --seed 42
```

Ligand representation is ECFP4 throughout (Morgan, radius 2, 1024 bits);
concatenated with the 480-d ESM-2 embedding this gives a 1504-d input to the MLP
(hidden `[512, 256]`, dropout 0.1, two-head). Target-disjoint splits cluster
sequences with `mmseqs easy-cluster --min-seq-id 0.3 -c 0.8` and assign whole
clusters to one fold, so no sequence cluster straddles train and test.

### Train

A single arm:

```bash
python code/analysis/train_arm.py \
    --dataset data/processed/bindingdb_IC50_censored.parquet \
    --split   data/processed/splits/splits_bindingdb_IC50_target_disjoint.parquet \
    --fp      data/features/ligand_fp.npz \
    --emb     data/features/target_emb_esm2_35M.npz \
    --arm F_tobit_rankdec --seed 1 --device cuda \
    --out results/main/target_disjoint
```

The full grid, all arms × splits × 5 seeds. It is resumable — jobs whose
`result_<arm>_seed<seed>.json` already exists are skipped:

```bash
python code/analysis/06_run_grid.py \
    --dataset data/processed/bindingdb_IC50_censored.parquet \
    --fp data/features/ligand_fp.npz --emb data/features/target_emb_esm2_35M.npz \
    --splits-dir data/processed/splits --split-tag bindingdb_IC50 \
    --arms A_drop B_cast C_bounded D_tobit E_tobit_rankobs F_tobit_rankdec \
    --schemes target_disjoint scaffold random --seeds 1 2 3 4 5 \
    --out results/main --name bindingdb_IC50 --device cuda
```

### Experiments and statistics

```bash
# Main table, bootstrap CIs, pre-registered paired comparisons with Holm
python code/analysis/30_aggregate.py \
    --results-root results/main --name bindingdb_IC50 --out results/tables

# Where the gains come from: per-target gain vs censoring rate and train identity
python code/analysis/07_generalization.py \
    --pertarget-dir results/main/target_disjoint \
    --dataset data/processed/bindingdb_IC50_censored.parquet \
    --split data/processed/splits/splits_bindingdb_IC50_target_disjoint.parquet \
    --out results/tables --name bindingdb_IC50

# Censoring stress: synthetically re-censor the training split at thresholds theta
python code/analysis/12_censoring_stress.py \
    --dataset data/processed/bindingdb_IC50_censored.parquet \
    --split data/processed/splits/splits_bindingdb_IC50_random.parquet \
    --fp data/features/ligand_fp.npz --emb data/features/target_emb_esm2_35M.npz \
    --out results/stress --name bindingdb_IC50 --thetas 4.0 4.5 5.0 5.5 6.0

# External screening benchmarks
python code/pipeline/20_process_dude.py --tar data/external/dude/all.tar.gz --out data/external/dude
python code/analysis/22_dude_batch_5seed.py     # DUD-E, 5 seeds, 95% identity leakage filter
python code/analysis/26_litpcba_score.py        # LIT-PCBA, experimentally measured inactives

# Prospective-style temporal evaluation on ChEMBL document years
python code/analysis/25_chembl_timesplit.py
python code/analysis/28_timesplit_analyze.py

# Effect sizes, paired bootstrap CIs, Holm correction
python code/analysis/24_effect_sizes.py
python code/analysis/27_gtuned_analyze.py       # validation-selected lambda for the G arms

# Diagnostics
python code/analysis/diag_censoring_phasediagram.py   # gain stratified by censoring rate
python code/analysis/diag_dude_paired.py
python code/analysis/diag_ftuned_paired.py
python code/analysis/diag_regression_uq.py            # MSE / MAE / censored NLL / mean sigma
```

`diag_censoring_phasediagram.py` produces the censoring-rate stratification that
the phase diagram in the paper is built from: per-target gains binned by natural
censoring rate across datasets and splits, with partial Spearman controlling for
target size, plus a left- versus right-censoring specificity check.

Comparisons are per-target paired, one-sided Wilcoxon, with paired bootstrap 95 %
confidence intervals and Holm correction within each pre-registered family.
Hyperparameters, thresholds and seeds are all in `configs/base.yaml`
(`seed: 42`; grid seeds `1..5`; lr 1e-3; batch 1024; max 200 epochs; patience 20).

## Repository layout

```
configs/base.yaml              project root, hyperparameters, thresholds, seed
conftest.py                    puts code/ on sys.path for pytest

code/
  losses/
    tobit.py                   censored-Gaussian NLL via log_ndtr
    bounded.py                 one-sided bounded squared hinge
    ranking.py                 censoring-decidable pairwise margin hinge
    combined.py                ARMS registry and the compute_loss dispatcher
  models/mlp.py                two-head MLP over [ECFP4 || ESM-2]
  utils/
    censoring.py               relation parsing, censor codes, decidability rule
    seeds.py                   seeding and run metadata
  eval/enrichment.py           EF, BEDROC, RIE, AUC, per-target summaries, bootstrap CI
  splits/split_utils.py        MMseqs2 clustering, Bemis-Murcko scaffolds, near-dup masks

  pipeline/
    01a_parse_bindingdb.py     BindingDB TSV -> censored measurements
    01c_extract_chembl.py      ChEMBL SQLite -> censored measurements
    02a_standardize_smiles.py  ChEMBL structure pipeline standardization
    02b_build_datasets.py      pAff conversion, censoring-aware deduplication
    03a_esm_embed.py           frozen ESM-2 t12 35M target embeddings
    03b_fingerprints.py        ECFP4 (Morgan r=2, 1024 bits)
    05_make_splits.py          target-disjoint / scaffold / random splits
    20_process_dude.py         DUD-E tarball -> compounds and target sequences

  analysis/
    train_arm.py               single-arm training and evaluation entry point
    06_run_grid.py             arms x splits x seeds orchestrator (resumable)
    07_generalization.py       gain vs censoring rate and train sequence identity
    12_censoring_stress.py     synthetic re-censoring sweep
    21_score_external.py       score a saved model on an external set
    22_dude_batch_5seed.py     DUD-E, 5 seeds
    25_chembl_timesplit.py     ChEMBL temporal split construction
    26_litpcba_score.py        LIT-PCBA scoring
    23_p1_aggregate.py         supplementary battery roll-up
    24_effect_sizes.py         effect sizes, bootstrap CIs, Holm
    27_gtuned_analyze.py       validation-selected lambda for G arms
    28_timesplit_analyze.py    temporal-split statistics
    30_aggregate.py            main results table and paired tests
    diag_*.py                  phase diagram, paired benchmarks, regression/UQ

tests/                         5 self-contained test modules
```

## Citation

```bibtex
@article{censored_affinity_2026,
  title   = {Censoring rate and decoy choice govern the screening value of
             censored bioactivity},
  author  = {TBD},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

MIT, see [LICENSE](LICENSE).
