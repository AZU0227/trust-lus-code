# TRUST_LUS — analysis code for reliability-based B-line routing

Code and aggregate tables for the manuscript *Does reliability-based model routing improve
cross-domain detection of B-lines in lung ultrasound?* (submitted to BMC Medical Imaging).

## What is here

```
scripts/    analysis scripts written for the reported analyses (pure NumPy; no GPU)
pipeline/   the frozen model-evaluation and gate-fitting pipeline used for the archived run
tables/     aggregate result tables (no per-clip identifiers)
figures/    Figure 1 and the frame-free alternative to Figure 2
derived/    per-clip derived tables (NOT MIT — see derived/LICENSES.md)
```

## Environment

Python 3.12 with NumPy. `scikit-learn` and `scipy` are **not** required: the logistic gates are
solved by Newton's method to the exact optimum of the same convex objective that
`sklearn.linear_model.LogisticRegression(penalty='l2', C=..., solver='lbfgs')` minimises.

```
pip install -r requirements.txt
python scripts/shared_cluster_sensitivity.py
```

## Scripts

| script | purpose | produces |
|---|---|---|
| `shared_cluster_sensitivity.py` | excludes Vave clips in clusters shared with Butterfly; four-arm comparison with a paired cluster bootstrap | `Table_S6_shared_cluster_sensitivity.csv` |
| `gate_weight_sensitivity.py` | refits the primary gate under three source-weighting schemes | `Table_S5_*.csv` |
| `rebuild_p4_soft_hard.py` | rebuilds per-clip soft/hard routing for the extension (needs the 5 source OOF files of M1) | `Vave374_soft_hard_rebuilt.csv` |
| `build_oof_and_vave.py` | joins source out-of-fold tables to the frozen fold table; rebuilds the Vave five-expert table | `OOF_source_*.csv`, `Vave_target_pM1_pM5_reconstructed.csv` |
| `mp4_probe.py` | ISO-BMFF container probe (frames, fps, duration, resolution) without ffmpeg | stdout / JSON |
| `make_figures.py` | draws Figure 1 and the frame-free Figure 2 alternative (PIL) | `figures/*.png` |

## Data

No image or video data are redistributed here, and no patient-level information is included.

- **Per-clip derived tables** (pseudonymous clip identifiers, reference labels, model probabilities)
  are in `derived/`. They are **not** MIT-licensed: each table follows the licence of the collection
  it derives from (CC BY-NC-SA 4.0 or CC BY-NC-ND 4.0). See `derived/LICENSES.md`.
- OpenPOCUS: CC BY-NC-SA 4.0. COVID-BLUeS: CC BY-NC-ND 4.0. Source-study ethical determinations are
  reported in the manuscript (Stanford IRB protocol 74680; METC azM/UM application no. 2020-2229).

## Licence

Code and aggregate tables: MIT (see `LICENSE`), because the aggregate tables contain no source-dataset
material beyond derived summary statistics. `derived/` is excluded from the MIT grant: each table there
follows the licence of the collection it derives from. See `derived/LICENSES.md`.

## Citation

See `CITATION.cff`. If you use the code, please cite the manuscript and the two source datasets.
