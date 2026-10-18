# UNSW-NB15 security model

This workspace contains a binary classifier for `normal` (`0`) versus `attack` (`1`).
It does not include GNN forecasting, genetic routing, or SDN-controller integration.

## Project-only experiment environment

AutoGluon and H2O experiments use an isolated `.venv` so their dependencies do not affect
other projects or the pinned baseline environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-experiments.txt
```

## Train

```powershell
python -m pip install -r requirements-security.txt
python train_security_model.py
```

To run the deduplicated-training experiment without changing the source data or
overwriting the baseline artifacts:

```powershell
python train_security_model.py --deduplicate --output artifacts/security-deduplicated
```

The training script:

- removes `id`, `attack_cat`, and `label` from the predictors;
- optionally removes exact repeated labeled training records while ignoring the unique `id`;
- keeps identical feature rows in the same development split;
- compares Logistic Regression, two focused Random Forest configurations, and Extra Trees on validation data;
- selects by balanced accuracy;
- calibrates the selected model on a separate calibration split;
- selects a decision threshold on another separate split while requiring at least 95% recall;
- records validation-only false-positive and false-negative breakdowns for error analysis;
- evaluates the frozen model on the official test CSV;
- saves the model, metrics, and test predictions under `artifacts/security/`.

## Apply the trained model

```powershell
python predict_security_risk.py input.csv predictions.csv
```

The input must contain the same UNSW-NB15 predictor columns used during training.

## Current result

Random Forest was selected. The threshold-tuning split selected a threshold of `0.6318`
while requiring at least 95% recall. On the official test set it achieved:

- accuracy: 0.8977
- balanced accuracy: 0.8890
- precision: 0.8584
- recall: 0.9752
- F1: 0.9130
- ROC-AUC: 0.9836
- PR-AUC: 0.9877
- false-positive rate: 0.1972

The model detects most attacks but still flags many normal flows. Do not describe its score
as a real-world attack probability or route by it until it is validated on traffic from the
target SDN environment. Approximately 10.37% of official test rows have feature vectors also
present in the supplied training file, so benchmark performance may overstate generalization.

## Deduplication experiment

Ignoring the unique `id`, 67,601 repeated labeled rows were removed from the training data.
The official test set was kept unchanged.

| Training data | Rows | Threshold | Balanced accuracy | Recall | F1 | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 175,341 | 0.6318 | 0.8890 | 0.9752 | 0.9130 | 0.1972 |
| Deduplicated | 107,740 | 0.4421 | 0.8606 | 0.9862 | 0.8955 | 0.2651 |

Deduplication made classification performance worse. Keep the original-data Random Forest as
the baseline model. Compared with its previous fixed `0.5` threshold, validation-only threshold
tuning removed 1,773 false alarms from the official test predictions while adding 430 missed attacks.

## Validation error analysis

The selected Random Forest produced 408 false positives and 429 false negatives on its internal
model-selection split at threshold `0.5`. This split, rather than the official test set, was used
to guide the next model experiment.

- Fuzzers accounted for 380 of the 429 missed attacks; Analysis accounted for another 35.
- Unnamed services (`-`) accounted for 343 of 408 false alarms.
- FTP had the highest false-positive rate among common named services at approximately 19%.
- `INT` traffic had a 17% false-positive rate, while most missed attacks were in `FIN` state.

A 300-tree Random Forest with one-sample leaves increased validation recall from `0.9645` to
`0.9739`, but increased the false-positive rate from `0.0721` to `0.0912` and reduced balanced
accuracy from `0.9462` to `0.9413`. The original two-sample-leaf Random Forest remains selected.
