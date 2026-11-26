# Per-clip derived tables — deposit notes

These files are intended for a separate archived record (Zenodo / OSF / institutional repository)
with a DOI, not for the code repository.

| file | rows | content | licence |
|---|---|---|---|
| `C2_OpenPOCUS_to_COVIDBLUES.csv` | 360 | target per-clip probabilities, gate weight, routed probability | CC BY-NC-ND 4.0 |
| `C2_COVIDBLUES_to_OpenPOCUS.csv` | 1,385 | as above, reverse direction | CC BY-NC-SA 4.0 |
| `OOF_source_OpenPOCUS_to_COVIDBLUES.csv` | 1,385 | source out-of-fold gate inputs + fold | CC BY-NC-SA 4.0 |
| `OOF_source_COVIDBLUES_to_OpenPOCUS.csv` | 360 | as above, reverse direction | CC BY-NC-ND 4.0 |
| `Vave_target_pM1_pM5_reconstructed.csv` | 374 | five-expert per-clip probabilities (five-fold mean) | CC BY-NC-SA 4.0 |
| `Vave374_soft_hard_rebuilt.csv` | 374 | gate weights, soft and hard routed probabilities | CC BY-NC-SA 4.0 |
| `exclusion_map_shared_clusters.csv` | 24 | Vave clips in clusters shared with Butterfly | CC BY-NC-SA 4.0 |

No image data, no patient-level information and no mapping from clip identifiers to individuals are
included. Clip identifiers are those used by the source releases.

Attribution requirement (both sources): cite the dataset papers and repositories named in the
manuscript. For OpenPOCUS: Kumar et al., POCUS J 2026;11:62-67, doi:10.24908/pocusj.v11i01.19439.
For COVID-BLUeS: Wiedemann et al., IEEE J Biomed Health Inform 2025;29:6301-6310,
doi:10.1109/JBHI.2025.3543686.
