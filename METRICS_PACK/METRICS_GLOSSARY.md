# Metrics glossary

## Patient-level ROC-AUC

The primary score. Each patient's slices are averaged into one score, then the model is evaluated on patients. A value of 0.5 is chance-level ranking; 1.0 is perfect ranking. It is not accuracy.

## Slice-level ROC-AUC

The same ranking idea applied to individual slices. It is secondary because slices from one patient are related and the research question is patient-level.

## OOF: out-of-fold

Each development patient is scored by a fold that did not train on that patient. This is the main protection against reporting a training-set score.

## Sensitivity / recall

Among positive patients, the fraction the model labels positive at a chosen threshold.

## Specificity

Among negative patients, the fraction the model labels negative at a chosen threshold.

## Precision

Among patients the model labels positive, the fraction that are positive in the recorded labels.

## F1 score

A combined score balancing precision and recall. It depends on the selected threshold.

## PR-AUC

Area under the precision-recall curve. It is useful when the positive and negative groups are not perfectly balanced.

## Balanced accuracy

The average of sensitivity and specificity. It gives the two classes equal weight.

## Probability standard deviation

How spread out the model's scores are. Very small spread can indicate that the model is predicting almost the same thing for everyone.

## KS distance

The Kolmogorov-Smirnov distance between two score distributions. Here it compares the full-development ensemble scores with the OOF score distribution. Lower means the score regime transferred more closely; E023 passed below 0.15.

## Threshold positive-rate delta

The absolute difference between the proportion of patients called positive in OOF and in the full-development transfer check. Smaller means the locked threshold behaves more consistently.

## Fold

One of five patient-wise partitions. Four folds train the model and one fold evaluates it. The process rotates so every development patient is evaluated once.

## Sealed test

The reserved 40-patient set. It was deliberately not used to tune E023 and must not be reopened for research optimization.
