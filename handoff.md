# Kidney Stone Detection CNN Handoff

**Updated:** 2026-08-27
**Repository:** `D:\Documents\Desktop\kidney stone detection-isolated`
**Current status:** The confirmed partial-shuffle defect is fixed and a development-only remediation CV improved OOF ranking. Its eight-epoch final-retrain simulation was not collapsed but failed score-transfer gates, so the sealed test has not been touched by remediation. The sealed test was opened once by the prior run, and no deployment bundle exists.

## Read This First

- The historical near-chance full CV is `runs/20260826_164227`; do not use it for final retraining.
- The prior test-opening run is `runs/20260826_231405_460200`; its OOF passed but final/test gates failed.
- The latest development-only remediation CV is `runs/20260827_173706_573400`; its OOF gates passed after the full-shuffle fix.
- Do not run `finalize` yet. Best epochs remain unstable (`[30, 31, 1, 1, 8]`), and the latest run has development-only final artifacts but no test artifacts or bundle.
- The sealed test must not be retuned or re-evaluated after the historical one-time measurement.

## User Requirements And Decisions

- Preserve the existing `kidney_cnn_v1` architecture. Do not replace it with transfer learning, attention, a different model family, or a larger architecture.
- Patient-level evaluation is required. The primary aggregation is the mean Stone probability across a patient's slices.
- Use the available RTX 4060 Laptop GPU, RAM, and CPU, but avoid Windows/WDDM VRAM spill and native CUDA crashes.
- Use staged verification: data/split integrity, smoke test, hardware probe, single-fold probe, full CV, then only after a passing CV result final retraining and sealed-test evaluation.
- Supplied derivatives were initially enabled because the user asked for the full data. The derivative experiment produced immediate memorization and validation collapse, so the current configuration deliberately uses originals only. This decision should remain explicit in future reports.
- `Fp16BatchNorm` was introduced as an implementation-level optimization with the same layer parameters and topology. A stock-BatchNormalization math-equivalence regression test now passes.
- No commit was created during this work. The worktree was already dirty and contains changes from this effort plus files whose origin should be reviewed before committing.

## Environment

- OS: Windows native
- Python: `3.12.5`
- TensorFlow: `2.20.0`
- Keras: `3.11.3`
- PyTorch: `2.5.1+cu121`
- `torch.cuda.is_available() == True`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- CPU: AMD Ryzen 7 7840HS, 16 logical processors
- RAM: approximately 15.29 GB
- `train.py` sets `KERAS_BACKEND=torch` before importing Keras.
- Keras 3 torch backend auto-places variables on `cuda:0`; the original metadata value `"gpu_name": "CPU"` was caused by a TensorFlow-only probe. Current metadata reports the NVIDIA GPU.
- A laptop power investigation initially saw contradictory battery status and a low 30 W reading. Later checks showed AC online, no active NVIDIA throttle, and a current GPU power limit of approximately 140 W. The principal performance problem was VRAM spill, not confirmed thermal throttling.
- Windows emitted `expandable_segments not supported on this platform` for `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; this setting has no useful effect on this platform.
- One derivative training process crashed in `nvcuda64.dll` with exception `0xc0000409`. Windows Event ID 1000 recorded the crash. The retry completed far enough to diagnose the model behavior.

## Dataset And Splits

- Dataset directory: `data/dataset/Kindey Stone Dataset` (the directory name is intentionally misspelled as `Kindey`).
- Manifest: `data/manifest.csv`.
- Total manifest rows: `38,821`.
- Original images: `3,364` total:
  - Non-Stone: `1,787`
  - Stone: `1,577`
- Supplied derivatives: `35,457`.
- Patients: `201` total:
  - Non-Stone: `97`
  - Stone: `104`
- Frozen outer split:
  - Development: `161` patients and `2,796` original slices
  - Sealed test: `40` patients and `568` original slices
  - Test slice labels: `349` Stone and `219` Non-Stone
- Five development folds contain `33, 33, 31, 31, 33` patients.
- Most original images are approximately `1275x737`; direct square resizing distorted anatomy.
- Many files with a `.jpg` name contain PNG bytes and/or RGBA channels. PIL RGB-to-L conversion is deterministic and the PIL/TensorFlow decode comparison found no meaningful pixel discrepancy.
- Patient labels are copied to every axial slice. The visual audit estimated that approximately `25-45%` of Stone-labeled slices may not visibly contain a stone. This is a weak-label limitation, not a verified annotation percentage.
- There are `154` cross-patient byte-identical original slices, including many duplicate groups of size two. The split implementation groups duplicate hashes and source IDs so they do not cross held-out boundaries.
- Leakage audit passed:
  - zero patient overlap between development and test
  - zero patient overlap across CV train/validation boundaries
  - zero source-image-ID boundary violations
  - zero SHA-256 duplicate boundary violations
  - zero unknown patient IDs in the audited manifest
- Site and format are confounded:
  - RA: `99` development / `8` test patients; predominantly PNG-like files
  - FA: `42` development / `23` test patients
  - ME: `20` development / `9` test patients
  - FA and ME are predominantly JPEG-like
- Development to test prevalence shifts:
  - image positive rate approximately `43.9%` to `61.4%`
  - patient positive rate approximately `48.4%` to `65.0%`
- A site-only baseline achieved approximately `0.562` AUC, close to the original model's `0.566` AUC. Within dominant RA, original OOF AUC was approximately `0.522`.

## Original Failed Run

Run directory: `runs/20260823_174206`

- Command: `cv`
- Commit recorded: `68b0c8dc09591f21b663c71d2a2a9d1a9922a845`
- `git_dirty: true`
- Metadata timestamp: `2026-08-23 23:51:13 UTC`
- Elapsed time: `41,947.43974232674` seconds, approximately `11.65 h`
- Configuration included:
  - image size `384`, one grayscale channel
  - batch size `16`
  - learning rate `3e-4`
  - weight decay `1e-4`
  - dropout `0.35`
  - epochs `60`
  - early-stopping patience `8`
  - ReduceLR patience `3`
  - derivatives disabled
- OOF slice ROC-AUC: `0.565912`
- OOF PR-AUC: `0.489543`
- OOF Youden threshold: `0.43277672`
- OOF Youden J: `0.1486489`
- Per-fold slice AUC: `[0.4014, 0.7129, 0.5706, 0.5881, 0.6199]`
- Best epochs: `[8, 13, 9, 6, 50]`; median `9`
- Training loss generally rose from approximately `0.585` toward `0.68`, while accuracy declined. The effective learning rate was too high for the weakly labeled sample size.
- At threshold `0.5`, confusion matrix was `TN=708, FP=860, FN=431, TP=797`.
- Locked OOF metrics were approximately sensitivity `0.927`, specificity `0.222`, balanced accuracy `0.574`.
- Final sealed test:
  - slice ROC-AUC `0.538819`
  - slice PR-AUC `0.702299`; test prevalence was `0.614`, so chance AP was approximately `0.614`
  - sensitivity `1.000`
  - specificity `0.000`
  - confusion matrix `TN=0, FP=219, FN=0, TP=349`
  - all probabilities were in `[0.5663, 0.9024]`, causing the locked threshold to classify every test slice as Stone
- The blind final retrain trained for nine epochs without validation, checkpointing, or LR decay. Its loss rose approximately `0.493` to `0.647`, so it saved the worst end-of-run model.
- Final-model/test versus OOF probability distribution KS distance was approximately `0.987`, demonstrating threshold/calibration transfer failure.
- Existing post-mortem: `runs/20260823_174206/FINDINGS.md`.

## Root Cause Findings

1. **Weak slice labels.** Patient-level labels are applied to every slice, so a substantial fraction of positive slices may contain no visible stone. Patient mean aggregation was empirically better than max or top-k aggregation.
2. **Small effective sample size.** The original model trained from scratch on roughly `2,237` original training images per fold, despite the large number of correlated slices.
3. **Learning-rate and callback instability.** The original `3e-4` learning rate was too aggressive. Noisy validation loss, no `min_delta`, and patience `8` caused unstable or prematurely stopped folds.
4. **Torch RNG was not seeded.** The old `set_seed()` seeded Python, NumPy, and TensorFlow but not torch/Keras. The recorded seed `42` did not make weight initialization and dropout reproducible under the torch backend.
5. **Blind final retrain.** The final model was independently retrained without validation or a checkpoint, so it could not be guaranteed to match the selected CV epoch or calibration regime.
6. **Preprocessing mismatch.** The RAM-cached path was using direct `resize((384,384))`, while the documented/inference path claimed aspect-preserving `resize_with_pad`. This squashed the common `1275x737` images by roughly `1.7x` in one dimension.
7. **Site/format shortcut risk.** Site distribution and file format correlate with labels and split membership. The site-only baseline nearly matched the original CNN.
8. **VRAM spill.** Stock Keras BatchNormalization upcasts activations to fp32 under `mixed_float16`. On this 8 GB Windows laptop, batch 16 entered the WDDM spill regime instead of failing fast.
9. **Current remaining issue: fold instability.** The shuffle remediation improved OOF ranking, but best epochs remain bimodal and two folds select epoch 1. A final retrain still needs a development-only transfer check before the sealed test can be considered.

## Hardware Evidence

Direct profiling of the current model and Keras torch stack contradicted the first agent projection of `204 ms/step` and `3.93 GiB` at batch 16. The reproducible local probe measured the following with fp16-native NHWC BatchNorm:

| Batch | ms/step | Peak allocated | Reserved | Throughput | Estimated 27k-image epoch |
|---:|---:|---:|---:|---:|---:|
| 8 | 220.8 | 4.13 GiB | 4.64 GiB | 36.2 img/s | 12.4 min |
| 10 | 272.7 | 5.15 GiB | 5.86 GiB | 36.7 img/s | 12.3 min |
| 12 | 324.3 | 6.18 GiB | 7.11 GiB | 37.0 img/s | 12.2 min |
| 14 | 491.5 | 7.27 GiB | 8.28 GiB | 28.5 img/s; spill | 15.8 min |
| 16 | 1,668 | 8.15 GiB | 9.24 GiB; spill | 9.6 img/s | 46 min |

- Batch `8` is the current safe choice. Batch `12` is slightly faster but has less VRAM margin.
- Direct batch-16 probes measured approximately `1667-1706 ms/step`, not the initially expected `204 ms/step`.
- tf.data feed cost was approximately `2.7 ms/batch`, so the input pipeline is not the bottleneck.
- PIL preload of approximately `29k` images was approximately `120 seconds/fold`.

## Implemented Changes

### `src/data.py`

- Added Mendeley derivative suffixes `cr`, `rot`, and `lc` to filename parsing.
- Changed the RAM-cached loader to RGB then grayscale conversion.
- Changed cached preprocessing to centered pad-to-square, then resize to `384x384`.
- Changed cached image storage from float32 to uint8; casting and division by `255` occur inside the dataset map.
- Kept validation and test restricted to original images.
- Added configurable mild brightness augmentation.
- Changed the default training shuffle buffer from `1024` to the full filtered dataset size; explicit smaller buffers remain opt-in.

### `src/model.py`

- Added registered `Fp16BatchNorm`.
- It has gamma, beta, moving mean, and moving variance with the same channel-wise parameter structure, epsilon `1e-3`, and momentum `0.99`.
- It computes NHWC statistics over `(N,H,W)` and casts stored statistics/parameters to the activation dtype.
- `build_model()` uses this layer in the existing four-block CNN.
- Model parameter count remains `4,788,866`.
- The docstring still claims `204 ms/step` versus `1,859 ms/step`; that claim is stale relative to the later direct profiling and should be corrected.

### `src/utils.py`

- `set_seed()` now calls `keras.utils.set_random_seed()` and torch CPU/CUDA seed functions in addition to Python, NumPy, and TensorFlow.
- `get_gpu_info()` probes torch CUDA when `KERAS_BACKEND=torch`, then falls back to TensorFlow.
- `configure_mixed_precision()` checks the active backend before enabling mixed precision.
- Remaining risk: seed failures are swallowed, and setting `PYTHONHASHSEED` during process execution cannot retroactively change Python hash randomization.

### `src/trainer.py`

- Added registered `SmoothedSparseCCE` because Keras `3.11` does not expose `label_smoothing` on sparse categorical cross-entropy.
- Added exclusion of BatchNorm gamma/beta and bias variables from AdamW weight decay.
- Seeds each fold with `seed + fold_idx`.
- Fold training uses `ModelCheckpoint`, configurable early stopping with `restore_best_weights=True` and `min_delta`, ReduceLR, and CSV logging.
- Fold predictions reload `best.keras`.
- Final training filters to development patients and honors `use_supplied_derivatives`.
- Final training seeds with `seed + 9999` and uses ReduceLR. It trains for the exact locked epoch budget without a validation set or checkpoint; the new pre-test development-transfer guard is the safety check before opening the sealed test.

### `src/evaluate.py`

- Patient-level mean aggregation is now primary.
- OOF threshold selection uses patient-level mean predictions.
- Slice-level metrics remain secondary.
- Added Youden infinite-threshold clamping and warnings for degenerate ROC results.
- Added per-site metrics and collapse warnings.
- Added final probability standard-deviation, KS-vs-OOF, and threshold-transfer gates.
- Added patient-level test prediction artifacts.
- Added a development-only final-model transfer guard that checks probability spread, KS distance, and threshold-rate transfer before sealed-test evaluation.
- Inference model loads use `compile=False` with explicit `Fp16BatchNorm` registration, avoiding optimizer/loss deserialization for prediction.
- `aggregate_patient_level()` infers site from `hospital` or, when absent, from path tokens.
- The historical test-time gates remain deployment safety checks after a test measurement. The new development-transfer guard runs before any future sealed-test evaluation and is intentionally conservative because it compares one in-sample final model with heterogeneous OOF models.

### `train.py`

- Sets `KERAS_BACKEND=torch` unless already set.
- Sets `PYTORCH_CUDA_ALLOC_CONF` by default.
- Refuses to run CV without an existing `data/manifest.csv`; CV no longer silently regenerates splits.
- Records manifest and split hashes.
- Warns when the git tree is dirty.
- Adds finalize-time manifest/split provenance checks.
- Requires final-model development transfer gates to pass before opening the sealed test.
- Skips bundle export when collapse/transfer gates fail.

## Experiment Timeline After The Fixes

### Derivatives-enabled experiment

- Derivatives were enabled for the requested full-data test.
- The first process was slow and was stopped for profiling. A later attempt crashed in `nvcuda64.dll` with `0xc0000409`; the retry ran successfully.
- The retry reached approximately:
  - train accuracy `0.967`
  - train loss `0.084`
  - validation accuracy `0.3802` (`211/555`)
  - validation loss `7.3723`, then approximately `9.12` and `9.43`
- This was immediate memorization/OOD collapse, not useful augmentation. Current config sets `use_supplied_derivatives: false`.

### Standalone fold-0 probe: `runs/20260826_132955`

- Originals only with stronger anti-memorization settings.
- Slice AUC `0.6017`.
- Patient mean AUC `0.5588`.
- Probability standard deviation `0.1109`.
- Best epoch `1`; training ended at epoch `16`.

### Over-regularized full CV: `runs/20260826_140347`

- Fold slice AUCs approximately `[0.6026, 0.4935, 0.4747, 0.4562, 0.5401]`.
- Pooled slice AUC `0.4877`.
- Patient AUC `0.4669`.
- Best epochs `[1,1,1,16,3]`; median `1`.
- This configuration was too strongly regularized.

### Standalone fold-3 middle-ground probe: `runs/20260826_155939`

- Originals only, learning rate `1e-4`, dropout `0.35`, weight decay `1e-4`, smoothing `0.05`, mild augmentation, batch `8`.
- Slice AUC `0.6823`.
- Patient mean AUC `0.6250`.
- Probability standard deviation `0.073`.
- Best epoch `13`.
- This single fold looked better than original fold 3 (`0.588`) and the over-regularized fold 3 (`0.456`), but it did not predict full-CV behavior.

## Prior Complete CV Before Shuffle Remediation: `runs/20260826_164227`

Current config is in `configs/default.yaml`:

- `use_supplied_derivatives: false`
- image size `384`, channels `1`
- rotation `10` degrees
- translation `0.03`
- zoom `0.05`
- contrast `0.1`
- brightness `0.05`
- dropout `0.35`
- batch size `8`
- epochs `60`
- learning rate `1e-4`
- weight decay `1e-4`
- label smoothing `0.05`
- early-stopping patience `15`
- early-stopping `min_delta=0.002`
- ReduceLR patience `5`
- ReduceLR factor `0.5`
- minimum LR `1e-6`
- mixed precision `auto`

Protocol deviations used by this remediation experiment, relative to `spec_text.txt`:

- Batch size `8` instead of `16` is the documented hardware-specific override for the 8 GB RTX 4060 laptop.
- Learning rate is `1e-4` instead of `3e-4`; early-stopping patience is `15` instead of `8`; ReduceLR patience is `5` instead of `3`.
- Label smoothing `0.05`, early-stopping `min_delta=0.002`, and brightness jitter `0.05` are additional remediation settings.
- Code restores the best validation weights and then reloads `best.keras`; the specification says `restore_best_weights=false` and explicit reload.
- `Fp16BatchNorm` replaces stock BatchNormalization for the same canonical topology to avoid mixed-precision VRAM spill. These settings must be reported as an experimental protocol, not silently presented as the locked V1 defaults.

Run results:

- Runtime metadata: approximately `9,210.5957736969` seconds, approximately `2 h 33 min`.
- Metadata reports the RTX 4060 Laptop GPU.
- Patient-level OOF ROC-AUC: `0.544640`.
- Slice-level OOF ROC-AUC: `0.550361`.
- Patient-level Youden threshold: `0.37695435666666666`.
- Youden J: `0.099166`.
- Patient-level locked metrics:
  - sensitivity `0.846154`
  - specificity `0.253012`
  - balanced accuracy `0.549583`
  - confusion matrix `TN=21, FP=62, FN=12, TP=66`
- At threshold `0.5`: sensitivity `0.397436`, specificity `0.698795`.
- Per-site patient AUC:
  - FA: `0.723457`, `42` patients
  - ME: `0.541667`, `20` patients
  - RA: `0.557851`, `99` patients
- Best epochs `[31,19,5,4,1]`; median `5`.

Per-fold artifact metrics:

| Fold | Best epoch | Best val loss | Total epochs | Val samples | Slice AUC | Probability std |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31 | `0.715915` | 46 | 555 | `0.4867` | `0.1011` |
| 1 | 19 | `0.703859` | 34 | 620 | `0.6992` | `0.1592` |
| 2 | 5 | `1.132693` | 20 | 671 | `0.6014` | `0.0913` |
| 3 | 4 | `0.704294` | 19 | 452 | `0.3566` | `0.0205` |
| 4 | 1 | `0.668310` | 16 | 498 | `0.5083` | `0.0508` |

## Validation Already Completed

- Verified after writing this handoff:
  - `python -m pytest tests -q` -> `46 passed`, `48 warnings` in `15.49s`.
  - `python train.py smoke --config configs/default.yaml` -> `SMOKE TEST PASSED`.
- The torch smoke test passed:
  - model construction
  - parameter count `4,788,866`
  - forward shape `(2,2)`
  - softmax row sums
  - one-epoch checkpoint save/reload
  - metric formulas
- `SmoothedSparseCCE` fit and `.keras` reload passed under the torch backend.
- `test_fp16_batch_norm_matches_stock_batch_norm_math` passes for training outputs, moving statistics, and inference outputs.
- Prediction CSVs now use `float_format="%.17g"` so serialized probabilities preserve threshold comparisons.

## Latest Complete CV And Finalization: `runs/20260826_231405_460200`

- All five folds completed with complete checkpoints and validation predictions after fold 4 was resumed in a separate invocation.
- OOF integrity passed: `2,796` original development slices, `161` patients, all five folds present once, with no duplicate paths or patient/source-image rows.
- Patient-level OOF ROC-AUC: `0.785758`; slice-level OOF ROC-AUC: `0.749966`.
- Fold patient-level AUCs: `[0.8860, 0.6544, 0.6397, 0.7708, 0.9417]`; no collapsed folds. Site AUCs were FA `0.7654`, ME `0.9167`, and RA `0.7145`.
- OOF acceptance passed. The locked Youden threshold was `0.489227`; the rounded median best epoch was `4` from best epochs `[26, 3, 2, 4, 26]`.
- Final retraining followed the locked protocol for exactly four epochs on all development patients. Training loss worsened from `0.5079` to `0.5311`.
- The sealed 40-patient test was evaluated once. Patient ROC-AUC was `0.538462`, with sensitivity `1.0`, specificity `0.0`, and every patient above the locked threshold.
- Final collapse/transfer gates failed: probability standard deviation `0.02986`, KS-vs-OOF `0.5652`, and threshold-positive-rate delta `0.4161`. Bundle export was correctly refused.
- This is a deployment-blocking final result. The test has now been opened and must not be used for another tuning loop or presented as an unbiased result after model changes. Any follow-up must be a new development-only experiment with a separately documented protocol.

## Shuffle-Remediated Development CV: `runs/20260827_173706_573400`

- This run used the same frozen development folds and canonical CNN configuration as the prior run. It never evaluated the sealed test set.
- The first CV invocation exceeded the two-hour command limit after completing folds 0-2 and starting fold 3. Folds 3-4 were resumed in the same run directory, then OOF aggregation completed.
- The loader now uses a full dataset shuffle buffer by default. A label-only reproduction produced both classes in all first 20 batches, unlike the previous class-blocked schedule.
- OOF integrity passed for `2,796` original development slices and `161` patients.
- Patient-level OOF ROC-AUC: `0.804140`; slice-level OOF ROC-AUC: `0.796691`.
- Compared with `runs/20260826_231405_460200`, patient AUC improved by `0.018381`, slice AUC by `0.046725`, mean fold AUC by `0.034216`, and worst fold AUC by `0.022059`.
- Fold patient AUCs: `[0.8676, 0.9301, 0.6618, 0.7417, 0.8625]`; no fold met the collapse criterion. Probability standard deviations were `[0.2947, 0.4057, 0.0791, 0.0661, 0.2051]`.
- Best epochs were `[30, 31, 1, 1, 8]`; median `8`. The two epoch-1 folds and low spread in folds 2-3 still make the blind final retrain unsafe.
- OOF acceptance passed with locked threshold `0.430775`.
- Recovery metadata intentionally leaves elapsed time unset and records the timeout recovery.

### Development-only final-retrain simulation

- `train_final_model()` was run on the `2,796` original development slices for exactly the locked median budget of `8` epochs. This was not an official finalization.
- Training loss decreased from `0.714325` to `0.329237` and training accuracy increased from `0.571888` to `0.892704`.
- In-sample development patient AUC was `0.972814`; this is optimistic and is not an acceptance metric.
- Development probabilities had mean `0.201644` and standard deviation `0.222743`, so the spread gate passed.
- The OOF-vs-development KS distance was `0.397516` and locked-threshold positive rates were `41.0%` OOF versus `17.4%` development, delta `0.236025`; both transfer gates failed.
- Development threshold confusion was `TN=83, FP=0, FN=50, TP=28`. The CLI safety gate therefore skipped sealed-test evaluation as intended.
- Artifacts are `final/model.keras`, `final/retrain_history.csv`, `final/development_predictions_patient_level.csv`, and `final/development_transfer.json`; no `test_predictions.csv`, `test_metrics.json`, or bundle exists.

### Final-model transfer audit

- An inference-only audit ran each existing fold checkpoint over all original development images and compared its training-patient scores with held-out-patient scores.
- Train-versus-held-out KS distances were `[0.3622, 0.2088, 0.1321, 0.1640, 0.3124]` for folds 0-4; threshold-rate deltas were `[0.1432, 0.0308, 0.0199, 0.0303, 0.2310]`.
- This reproduces the final model's score-regime shift and supports expected in-sample score drift as the primary explanation, while leaving the transfer failure deployment-blocking.
- Regenerated held-out checkpoint probabilities matched persisted validation predictions within `7e-5` at worst.

### Root-Cause Analysis

1. **Partial shuffle created class-blocked training epochs.** Before remediation, `src/data.py` used `shuffle_buffer=1024`, while the sorted original manifest was grouped as all Non-Stone rows followed by all Stone rows. On the final development set (`2,796` rows), a TensorFlow label-only reproduction produced no positive examples in the first `72` batches; the final `50` batches were `71.5%` positive. The loader now defaults to a full dataset buffer, and the first 20 batches of the remediated fold contained both classes. The OOF improvement supports this as a material contributor, but not the only remaining issue.
2. **The frozen outer split is not representative.** The development/test patient composition is FA `42/23`, ME `20/9`, and RA `99/8`; patient positive prevalence shifts from `48.4%` in development to `65.0%` in test. Within every site it also increases: FA `35.7% -> 60.9%`, ME `40.0% -> 66.7%`, and RA `55.6% -> 75.0%`. The model sees test probabilities around `0.61` for both negative and positive patients, so it is reacting to the shifted test regime rather than ranking labels.
3. **Split generation had reproducibility and data-conflict defects.** Before remediation, `generate_patient_splits()` accepted `seed` but never used it; it now uses seeded tie-breaking. Fresh generation still correctly fails on conflicting-label duplicate components (`P066/P088`, `P070/P087`, and the `P074/P090/P091/P092/P094` component). The existing frozen CSV keeps these components together in test, so leakage checks pass, but that split cannot be regenerated from the current manifest without a data-policy decision.
4. **The epoch lock is statistically fragile.** Fold best epochs are bimodal `[26, 3, 2, 4, 26]`, so the mandated median is `4`. Validation loss is slice-level and noisy under patient-level weak labels, while the final model is a different single-seed model trained without validation. The final loss worsened across all four epochs (`0.5079 -> 0.5311`), and the resulting probabilities collapsed.
5. **The OOF gate is necessary but insufficient.** OOF evaluates five separate fold checkpoints and does not test the single final model, epoch dispersion, or calibration transfer. The new pre-test development-transfer guard now checks the final model's probability spread and OOF distribution before the sealed test can be opened.
6. **The pre-test transfer guard is conservative but informative.** It compares one final model evaluated in-sample with five heterogeneous OOF models, so KS and threshold-rate differences cannot be called independent calibration error. The inference-only fold audit reproduced comparable train-versus-held-out shifts, confirming that the final failure is primarily score-regime drift rather than a unique parser or label bug. It still correctly blocks a final-test measurement whose threshold transfer has not been demonstrated.
7. **Prediction artifact precision matters.** The completed run was written with pandas' default float formatting. Patient `P188` is exactly tied with the in-memory threshold (`0.43077478799999996`) but was serialized one ULP below it (`0.4307747879999999`), making the CSV-based transfer guard count `66` positives while `oof_metrics.json` records `67`. Future prediction artifacts now use `float_format="%.17g"`; the completed run is retained unchanged.
8. **There are additional reproducibility risks.** The recovered CV metadata leaves `elapsed_time_seconds` unset and records the recovery explicitly. Inference loading now uses `compile=False` and explicit `Fp16BatchNorm` registration, and a clean process successfully loaded a real checkpoint. Neither issue caused the measured test collapse, but both required correction before any future bundle is considered deployable.

## Known Worktree State

The following paths are currently modified or untracked. Do not revert unrelated changes without inspecting them:

```text
 M configs/default.yaml
 M requirements.txt
 M src/data.py
 M src/evaluate.py
 M src/model.py
 M src/splits.py
 M src/trainer.py
 M src/utils.py
 M tests/test_data.py
 M tests/test_evaluate.py
 M tests/test_smoke.py
 M tests/test_splits.py
 M train.py
?? handoff.md
?? spec_text.txt
```

Temporary investigation logs and scripts are under `runs/staging/`. Important known files include:

- `runs/staging/fold0_train.log`
- `runs/staging/cv_full.log`
- `runs/staging/cv_full2.log`
- `runs/staging/cv_full2_pid.txt`
- hardware probe scripts created during the profiling investigation

## Next Plan

1. Keep `runs/20260827_173706_573400` as a development-only result. Do not run `finalize`; epoch instability remains despite passing OOF gates.
2. Decide how contradictory duplicate components should be handled in a future dataset release before regenerating the outer split. Do not overwrite the frozen split CSVs.
3. Audit the active batch-size/learning-rate drift against `spec_text.txt` before another controlled experiment.
4. Use repeated-seed, patient-wise, site-aware development experiments to determine whether the remaining fold instability is acceptable. Keep derivatives disabled and preserve the CNN architecture.
5. Only a future independently documented protocol, with a passing pre-test development-transfer guard, may open the already-invalidated sealed test for a new external evaluation.

Validation commands already completed:

   ```powershell
   python -m pytest tests -q
   $env:KERAS_BACKEND='torch'; python train.py smoke --config configs/default.yaml
   ```

## Relevant Files

- `configs/default.yaml`: current training configuration.
- `train.py`: `prepare`, `smoke`, `cv`, and `finalize` CLI stages.
- `src/data.py`: manifest parsing, preprocessing, caching, and augmentation.
- `src/model.py`: canonical CNN and `Fp16BatchNorm`.
- `src/trainer.py`: loss, optimizer, callbacks, fold training, and final training.
- `src/evaluate.py`: metrics, patient aggregation, threshold selection, gates, and bundle export.
- `src/splits.py`: frozen patient-wise split validation and duplicate/source grouping.
- `src/utils.py`: configuration, seeding, backend-aware GPU reporting, hashes, and metadata.
- `data/manifest.csv`: indexed dataset manifest.
- `splits/outer_split.csv`: frozen 161-development/40-test patient split.
- `splits/cv_folds.csv`: frozen five-fold development assignments.
- `runs/20260823_174206/FINDINGS.md`: original post-mortem; partially stale.
- `runs/20260823_174206/`: original failed CV/final/test artifacts.
- `runs/20260826_132955/`: standalone anti-memorization fold-0 probe.
- `runs/20260826_140347/`: over-regularized full CV.
- `runs/20260826_155939/`: standalone middle-ground fold-3 probe.
- `runs/20260826_164227/`: prior near-chance full CV; not suitable for finalization.
- `runs/20260827_173706_573400/`: shuffle-remediated development-only CV; OOF passed, final development-transfer guard failed, and no sealed-test artifacts or bundle exists.
- `runs/20260826_231405_460200/`: prior OOF-passing run whose one-time final/test evaluation failed; not deployable.
- `tests/`: current unit/smoke coverage.
- `C:\Users\parth\.agents\skills\error-debugging-error-analysis\resources\implementation-playbook.md`: investigation framework used during the diagnosis.
