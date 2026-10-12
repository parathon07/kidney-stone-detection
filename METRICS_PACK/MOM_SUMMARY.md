# Simple explanation for family

## What was built?

This is a research model that looks at CT image slices and estimates whether the patient is associated with a kidney stone. It combines the evidence from all slices belonging to one patient before producing the main score.

## What is the headline result?

The best development result is a patient-level ROC-AUC of **0.8534**.

The original research baseline was **0.8041**. The measured improvement is **0.0493**.

In plain language, the newer model ranks positive patients above negative patients more reliably than the original baseline on the development data.

## What do the threshold numbers say?

At one selected decision threshold on the development predictions:

- It identified about **71.8%** of positive patients.
- It correctly rejected about **89.2%** of negative patients.
- When it called a patient positive, it was correct about **86.2%** of the time in this development evaluation.

These are research measurements on this dataset. They are not a promise of performance in a hospital.

## How did we check the result?

- The patients were split into five groups.
- Each patient was evaluated by a fold that did not train on that patient.
- The model passed the out-of-fold quality gates.
- The five fold results were reasonably consistent; the weakest fold scored 0.864 patient ROC-AUC.
- The score distribution remained stable in the development transfer check.
- 47 correctness tests passed, and the command-line smoke check passed.

## Why are we careful about the claim?

There are only 201 patients in the source dataset. Many image slices come from the same patient, so the images are not 3,364 independent medical cases. The dataset also contains different sites/formats and uneven numbers of slices per patient.

The reserved 40-patient sealed test set was not used to tune this result. An earlier historical test attempt collapsed to predicting every patient as Stone, so that sealed test is permanently excluded from future tuning. The current result is therefore a careful development result, not external clinical validation.

## How to use the folder

Start with `START_HERE.md`, then open the graphs in `graphs/`. `CURRENT_BEST_METRICS.md` has the detailed scorecard. `ALL_EXPERIMENT_METRICS.md` shows the complete research history, including experiments that failed or were rejected.

For a visual explanation of the entire project, open `PROJECT_EXPLAINER.html`. It works as a standalone offline file and has no quiz or Q&A section.
