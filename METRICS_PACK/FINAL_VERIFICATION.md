# Final verification record

The final verification for E023 checked the following conditions:

| Check | Result |
|---|---|
| OOF rows valid | PASS |
| Patient coverage valid | PASS |
| OOF gates | PASS |
| Development transfer gates | PASS |
| Sealed test untouched by E023 | PASS / false |
| Manifest hash | PASS |
| Frozen split hashes | PASS |
| All five fold artifacts | PASS |
| Correctness tests | 47 passed |
| CLI smoke | PASS |

The source verification output is preserved in the research state file `.research/e027_final_verification_raw.log`. The machine-readable E023 artifacts are in `e023_evidence/`.

## Reproducibility anchors

- Research commit anchor: `68b0c8d`
- E023 seed: `43`
- Data manifest SHA-256: `b4ec07b42bd38e8dcc4b3689b50cd18080ed9b2dd7e5ade38728bfc8d24ef206`
- Outer split SHA-256: `966d1d8f4d0b17a09db1182e3a9a5194ab7283fe85d63738cb4023e0e4b9c931`
- CV folds SHA-256: `32bf8373df2b09e4d9b53b6ba93616aff618a668c2592bc3f2a532b4379b4290`

These hashes identify the data manifest and frozen split definitions used by the candidate.
