# Published research package

This branch publishes the verified kidney-stone detection research package.

- Current verified champion: E023 patient-grouped five-fold ensemble
- Patient OOF ROC-AUC: `0.8534136546`
- Development transfer guard: passed
- Final verification: 47 tests passed; CLI smoke test passed
- Model artifacts: [`artifacts/models/e023`](artifacts/models/e023), stored with Git LFS
- Dataset source: [Kidney Stone Dataset, Mendeley Data V2](https://data.mendeley.com/datasets/fwhytt5mzd/2)

The raw CT image collection is not committed to this public repository. It is
approximately 2.22 GB of medical imagery and remains available from the official
dataset source above. The tracked manifests, frozen splits, research ledger,
reports, and model checksums document the exact development state and allow the
dataset to be prepared locally.

Historical failed or superseded checkpoints remain in the local research archive
and are intentionally not published as part of the verified release artifact.
