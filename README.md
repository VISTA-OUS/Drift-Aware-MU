# Drift-Aware Machine Unlearning for Tabular Data: An Efficient Alternative to Full Retraining

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VISTA-OUS/Drift-Aware-MU/blob/main/source/feature_unlearning_gbt_source.ipynb)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![Status](https://img.shields.io/badge/paper-under%20review-orange)

Official code for the paper **"Drift-Aware Machine Unlearning for Tabular Data: An Efficient Alternative to Full Retraining"**
by Gia-Gia Ngo-Trieu, Viet-Tuan Le and Hung Phuoc Truong.

> **Paper status:** under review. The citation entry below will be updated once the paper is published.

---

## Overview

Gradient-boosted decision trees (GBDTs) are the dominant model family for tabular prediction, but they degrade under
**concept drift** when feature-target relationships become obsolete. Full retraining removes the influence of such
*stale* features, yet it is expensive and has to be repeated whenever drift recurs. Existing feature-level unlearning
methods are restricted to convex / differentiable models, which leaves tree ensembles largely unexplored.

This repository studies **feature-level unlearning for GBDTs under synthetic concept drift**. It contains a complete,
end-to-end notebook that:

1. builds a controlled drift benchmark (six sequential domains **D0-D5**) from two public tabular datasets,
2. **detects stale features** automatically with a drift score,
3. **unlearns** those features from an already-trained `GradientBoostingClassifier`,
4. **audits** the result against a fully retrained *Gold* model with attribute-inference and reconstruction attacks.

### Methods compared

| Name in paper | Name in notebook | Type | Description |
|---|---|---|---|
| **Leaf and Split Masking** (Method 1, proposed) | `LeafSplitMask` | In-place unlearning, **no retraining** | Zeroes the leaf values below every split on a stale feature, and the leaves never reached by stale-masked retain data. Tree structure (thresholds, feature indices, child pointers) is left untouched. |
| **Feature Masking** (Method 2, retraining baseline) | `MaskedRetrain` | Partial retraining | Sets the stale columns to 0 and retrains on retain data with an adaptive tree budget `n_new = max(10, M - n_stale)`. |
| **Gold Retrain** (reference) | `Gold` | Full retraining | New ensemble trained from scratch on retain data with stale features masked, full original tree budget. |
| Warnecke et al. (NDSS'23) | `Warnecke-1st(LR)` / `Warnecke-2nd(LR)` | Cross-architecture baseline | First- and second-order closed-form feature unlearning, adapted to logistic regression on the same drift data. |

### Pipeline

```
 Census / Forest  ->  class balancing  ->  drift injection (D0-D5)  ->  train/val/test split
                       (SMOTE / ROS)       forget = D0-D2, retain = D3-D5
                                                      |
                                  Step 1: drift score  (L2-regularised logistic models
                                          theta_forget vs theta_retain)  ->  stale set f
                                                      |
                     Step 2: unlearn f from the trained GBDT  (Method 1 / Method 2 / Gold)
                                                      |
                     Evaluation: Ret Acc, delta-conf, AIA, reconstruction attack (ceiling-normalised)
```

---

## Key results

All numbers are mean over five seeds (42, 7, 123, 2024, 777); runtimes were measured on a shared CPU, so absolute
values are machine-dependent. See the paper (Tables 3-8) for full details, including standard deviations.

**Census Income** (stale features: top-10 of 108 one-hot features, driven by `capital-gain`)

| Method | Ret Acc | Δconf | Runtime | AIA R² | Top-1 reconstruction R² |
|---|---|---|---|---|---|
| Original | 0.8017 | 0.0208 | 5.60 s | 0.055 | 0.245 |
| Gold Retrain | 0.8177 | 0.0000 | 2.05 s | 0.025 | 0.055 |
| **Leaf and Split Masking** | 0.7896 | 0.0000 | **0.07 s** | 0.006 | 0.055 |
| Feature Masking | 0.8175 | 0.0000 | 1.83 s | 0.015 | 0.051 |

**Forest Cover Type** (stale features: top-3 of 10, the three `Hillshade_*` attributes)

| Method | Ret Acc | Δconf | Runtime | AIA R² | Top-1 reconstruction R² |
|---|---|---|---|---|---|
| Original | 0.5312 | 0.0616 | 73.40 s | 0.653 | 0.226 |
| Gold Retrain | 0.5471 | 0.0000 | 28.29 s | 0.654 | 0.141 |
| **Leaf and Split Masking** | 0.5178 | 0.0000 | **0.49 s** | 0.654 | 0.156 |
| Feature Masking | 0.5432 | 0.0000 | 2.71 s | 0.653 | 0.175 |

Take-aways (as reported in the paper):

- Both strategies reach the Gold model's residual confidence (`Δconf = 0`, *Toward Gold = 100 %*). Note that this is
  a deterministic consequence of the masking construction and acts as a pipeline correctness check; the attack-based
  metrics are the empirical evidence of forgetting.
- Leaf and Split Masking is ~29x (Census) to ~58x (Forest) faster than Gold Retrain (≈44x on average); Feature
  Masking is ~1.1x to ~10x faster.
- Single-feature reconstruction drops by up to 79 % compared with the original model, at a utility cost of a few
  percentage points (Leaf and Split Masking) or almost none (Feature Masking).
- On Forest, AIA stays flat (≈0.65) for every model because the remaining features are strongly collinear with the
  hillshade attributes; the ceiling-normalised and Top-1 reconstruction metrics are used to separate this
  baseline leakage from model leakage.

---

## Repository structure

```
Drift-Aware-MU/
├── README.md
├── requirements.txt
├── dataset/                                   # the two CSV files + descriptive analysis (see "Data")
│   ├── README.md                              # dataset schema, descriptive statistics, label distributions
│   ├── census_stable.csv
│   ├── covtype_stable.csv
│   └── figures/                               # EDA figures used in dataset/README.md
└── source/
    └── feature_unlearning_gbt_source.ipynb   # full pipeline reported in the paper
```

Figures and CSV result files produced by the notebook are written to the notebook's working directory (`source/`).

---

## Data

Two cleaned CSV files are used. They are derived from public UCI datasets:

| File | Source dataset | Raw shape | Description |
|---|---|---|---|
| `census_stable.csv` | [Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) | 48,841 x 15 | 6 numerical + 8 categorical features, binary label (`<=50K` / `>50K`) |
| `covtype_stable.csv` | [Covertype](https://archive.ics.uci.edu/dataset/31/covertype) | 581,012 x 11 | 10 numerical features, 7-class label (`label` ∈ 1-7) |

Both files are small (≈5.7 MB and ≈24 MB). Download links (these are the same files the notebook downloads with `gdown`):

- `census_stable.csv`: <https://drive.google.com/uc?id=1BrR9UMFox4-DKYt4g3gYc8hDQTAwwnO2>
- `covtype_stable.csv`: <https://drive.google.com/uc?id=1aE4163bJluyY9Y95X551Ggc6foC8rdXW>

A descriptive analysis of both datasets (schema, summary statistics, label distributions) is in
[`dataset/README.md`](dataset/README.md).

Class balancing (SMOTE for Census, random oversampling for Forest) and the drift construction (D0-D5) are performed
**inside the notebook**, so no further preprocessing is needed.

---

## Getting started

### Option 1 - Google Colab (zero setup)

1. Click the **Open in Colab** badge at the top of this page (or upload the notebook to Colab).
2. Run all cells. The data is downloaded automatically with `gdown`, and the required libraries are pre-installed on
   Colab (`imbalanced-learn` is installed by the notebook itself when needed).

### Option 2 - Run locally

**1. Clone and create an environment** (Python 3.10-3.12)

```bash
git clone https://github.com/VISTA-OUS/Drift-Aware-MU.git
cd Drift-Aware-MU

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install jupyterlab             # or `notebook`; any Jupyter front-end / VS Code works
```

**2. Get the data** into `dataset/` using **one** of the following:

- *Manual*: download the two files from the links in [Data](#data) and save them as
  `dataset/census_stable.csv` and `dataset/covtype_stable.csv`.
- *Command line*:
  ```bash
  mkdir -p dataset
  gdown 1BrR9UMFox4-DKYt4g3gYc8hDQTAwwnO2 -O dataset/census_stable.csv
  gdown 1aE4163bJluyY9Y95X551Ggc6foC8rdXW -O dataset/covtype_stable.csv
  ```
- *From the notebook*: the commented cell right after the `!gdown` cell contains the same download logic using the
  `gdown` Python API.

**3. Launch the notebook from `source/`**

```bash
cd source
jupyter lab feature_unlearning_gbt_source.ipynb
```

**4. Adjust the notebook for local execution** (only needed once; Colab users can skip this)

1. **Skip** the `!gdown` cell in *Data Upload* (it downloads into the Colab working directory).
2. In the **"Load Census"** and **"Load Forest"** cells, comment out the Colab line and uncomment the local one:

   ```python
   # df_census = pd.read_csv('../dataset/census_stable.csv')   # <- local (uncomment)
   df_census = pd.read_csv('census_stable.csv')                # <- Colab (comment out)
   ```

   ```python
   # df_forest = pd.read_csv('../dataset/covtype_stable.csv')  # <- local (uncomment)
   df_forest = pd.read_csv('covtype_stable.csv')               # <- Colab (comment out)
   ```
3. Run all cells top to bottom.

> **Alternative:** if `gdown` works on your machine you can simply run the `!gdown` cell unchanged; the CSVs will be
> saved in `source/` and the notebook's default `pd.read_csv('census_stable.csv')` lines work as they are.

**Troubleshooting**

- `FileNotFoundError: ../dataset/census_stable.csv` - the kernel's working directory is not `source/`. Start Jupyter
  from `source/`, or (VS Code) set the notebook's working directory to the notebook folder.
- `!gdown` fails (blocked network / quota exceeded) - use the manual download.
- `numpy` build/wheel errors on Python 3.13 - use Python 3.10-3.12.

---

## Notebook guide

The notebook contains exactly the pipeline reported in the paper. Part numbers are kept as in the original
development notebook for traceability (hence Parts 3 and 9 are absent).

| Part | Content | Paper reference |
|---|---|---|
| 0 | Environment setup, shared imports, global `SEED = 42` | - |
| 1 / 1b | Minimal EDA; class balancing (SMOTE on Census, RandomOverSampler on Forest) | Sec. 4.1, Table 2 |
| 2 | Synthetic drift generation: six domains D0-D5 via weighted resampling (covariate drift + label shift + Gaussian noise, no row overlap between domains) | Sec. 4.2 |
| 4 | Encoding, stratified 60/20/20 split; linear helper models (`BinaryLogisticModel`, `MultinomialLogisticModel`) | Sec. 4 |
| 5 | Training the "Original" GBDT on the drifted data | Sec. 3.1 |
| 6 | **Step 1** - drift score and stale-feature selection (`TOP_K_CENSUS = 10`, `TOP_K_FOREST = 3`, `LAMBDA = 0.01`) | Sec. 3.2, Fig. 1 |
| 7 | **Step 2** - Leaf & Split Masking, Feature Masking, Gold Retrain; utility / forgetting / runtime; **7.10 five-seed robustness check** | Sec. 3.3-3.5, 5.1, 6.1, Tables 3-4 |
| 8 | Attribute Inference Attack (AIA) | Sec. 5.2, 6.2, Tables 5-6 |
| 10 | Cross-architecture baseline: Warnecke et al. (NDSS'23) first / second-order unlearning on logistic regression | Sec. 3.5, 6.3 |
| 11 | Reconstruction attack: ensemble attacker, positive-control ceiling, ceiling-normalised and Top-1 metrics | Sec. 5.3, 6.3, Tables 7-8 |

**Generated files** (saved next to the notebook): `census_label_dist.png`, `forest_label_dist.png`,
`census_drift_analysis.png`, `forest_drift_analysis.png` (Fig. 1), `census_gbt_unlearn.png`, `forest_gbt_unlearn.png`,
`multi_seed_results_raw.csv`, `multi_seed_results_summary.csv` (Tables 3-4), `security_aia_comparison.png`,
`gbt_vs_warnecke_comparison.png`.

### Reproducibility notes

- **Model**: scikit-learn `GradientBoostingClassifier` (`n_estimators=100`, `learning_rate=0.1`, `subsample=0.8`,
  `max_depth=4` for Census and `5` for Forest). XGBoost/LightGBM are deliberately **not** used because the unlearning
  methods edit the internal tree arrays (`estimators_`, `tree_.value`, `tree_.feature`, ...) that these libraries do
  not expose.
- **Seeds**: single runs use `SEED = 42`; Section 7.10 repeats the pipeline for seeds 42, 7, 123, 2024 and 777.
- **Runtime**: the five-seed loop in 7.10 takes roughly 3 minutes per seed on a Colab CPU, and the reconstruction
  attack in Part 11 can take 10-20 minutes (it fits many attacker models). The notebook is CPU-only; no GPU is needed.
- **Versions**: `requirements.txt` pins one mutually compatible set of library versions. Small numerical differences
  from the paper's tables may appear with other scikit-learn / NumPy versions or on different hardware.
- **Timings** in the paper's tables are wall-clock measurements and will differ on your machine; the relative speedups
  are what matter.

---

## Limitations

(Discussed in Section 7 of the paper.) The drift score is linear and may miss non-linear feature-target shifts; the
10 % importance threshold of Feature Masking has not been swept systematically; and all experiments use a single
gradient-boosting implementation (scikit-learn) with fixed drift strengths on synthetic drift.

---

## Citation

If you use this code, please cite the paper (entry to be updated after publication):

```bibtex
@inproceedings{ngotrieu2026driftawaremu,
  title     = {Drift-Aware Machine Unlearning for Tabular Data: An Efficient Alternative to Full Retraining},
  author    = {Ngo-Trieu, Gia-Gia and Le, Viet-Tuan and Truong, Hung Phuoc},
  year      = {2026},
  note      = {Under review},
  url       = {https://github.com/VISTA-OUS/Drift-Aware-MU}
}
```

<!-- ## Contact

- Gia-Gia Ngo-Trieu - 25C1103733@student.hcmus.edu.vn
- Viet-Tuan Le - tuan.lv@ou.edu.vn
- Hung Phuoc Truong (corresponding author) - hung.tphuoc@ou.edu.vn -->

## License

<!-- TODO: choose and add a LICENSE file (e.g. MIT / Apache-2.0), then name it here. -->
License to be specified.

## Acknowledgements and references

- The linear feature-unlearning baseline follows Warnecke et al., *Machine Unlearning of Features and Labels*, NDSS 2023.
- Drift-domain construction follows the weighted-resampling approach of Helli et al., *Drift-Resilient TabPFN*, NeurIPS 2024.
- Datasets: UCI Machine Learning Repository - Adult (Becker & Kohavi) and Covertype (Blackard & Dean).
