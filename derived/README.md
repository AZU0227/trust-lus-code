# Per-clip derived tables

These files support the sensitivity analyses in the manuscript. They contain **no image data**, no
videos and no mapping from any identifier to an individual. Identifiers are the pseudonymous clip and
cluster identifiers used by the two source releases.

| file | rows | content | licence |
|---|---|---|---|
| `C2_OpenPOCUS_to_COVIDBLUES.csv` | 360 | target per-clip probabilities (M3, M4), gate weight, routed probability | CC BY-NC-ND 4.0 |
| `C2_COVIDBLUES_to_OpenPOCUS.csv` | 1,385 | as above, reverse direction | CC BY-NC-SA 4.0 |
| `OOF_source_OpenPOCUS_to_COVIDBLUES.csv` | 1,385 | source out-of-fold gate inputs, fold and cluster | CC BY-NC-SA 4.0 |
| `OOF_source_COVIDBLUES_to_OpenPOCUS.csv` | 360 | as above, reverse direction | CC BY-NC-ND 4.0 |
| `Vave_target_pM1_pM5_reconstructed.csv` | 374 | five-expert per-clip probabilities (mean of five fold models) | CC BY-NC-SA 4.0 |
| `Vave374_soft_hard_rebuilt.csv` | 374 | gate weights, soft and hard routed probabilities | CC BY-NC-SA 4.0 |
| `exclusion_map_shared_clusters.csv` | 24 | Vave clips inside clusters shared with the Butterfly cohort | CC BY-NC-SA 4.0 |

## Column notes

- `unit_id` — clip identifier from the source release (COVID-BLUeS: `patient_<id>_<BLUE point>`;
  OpenPOCUS: `<case>_z<zone>`).
- `cluster_id` — acquisition cluster used as the resampling unit.
- `y` — reference label supplied by the source release (1 = B-line present).
- `p_M3`, `p_M4` — expert probabilities (M3 spatiotemporal CNN; M4 video transformer).
- `alpha` / `q_M4` — weight given to M4 by the fitted gate.
- `p_router`, `p_soft`, `p_hard` — routed probabilities.
- `fold` — source fold from the frozen fold assignment.

## Attribution

Both source collections require attribution.

- OpenPOCUS: Kumar A, Nandakishore P, Gordon AJ, et al. Creation of an open-access lung POCUS image
  database for deep learning and neural network applications. POCUS J. 2026;11:62-67.
  doi:10.24908/pocusj.v11i01.19439. Repository: https://github.com/kumarandre/OpenPOCUS.
  Licence: CC BY-NC-SA 4.0.
- COVID-BLUeS: Wiedemann N, de Korte-de Boer D, Richter M, et al. COVID-BLUeS — a prospective study
  on the value of AI in lung ultrasound analysis. IEEE J Biomed Health Inform. 2025;29:6301-6310.
  doi:10.1109/JBHI.2025.3543686. Repository: https://github.com/NinaWie/COVID-BLUES.
  Licence: CC BY-NC-ND 4.0.

## What these tables are not

They are not a redistribution of either dataset: no frames, no videos, no clinical variables and no
patient-level records are included. They are derived model outputs joined to the released labels.

## Regenerating them

`scripts/build_oof_and_vave.py` rebuilds the two out-of-fold tables and
`Vave_target_pM1_pM5_reconstructed.csv`; `scripts/rebuild_p4_soft_hard.py` rebuilds
`Vave374_soft_hard_rebuilt.csv`; `scripts/shared_cluster_sensitivity.py` rebuilds
`exclusion_map_shared_clusters.csv` and Table S6.
