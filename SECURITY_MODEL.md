# UNSW-NB15 security model

This workspace contains a binary classifier for `normal` (`0`) versus `attack` (`1`).
It does not include GNN forecasting, genetic routing, or SDN-controller integration.

## Train

```powershell
python -m pip install -r requirements-security.txt
python train_security_model.py
```

The training script:

- removes `id`, `attack_cat`, and `label` from the predictors;
- keeps identical feature rows in the same development split;
- compares Logistic Regression and Random Forest on validation data;
- selects by balanced accuracy;
- calibrates the selected model on a separate calibration split;
- evaluates the frozen model on the official test CSV;
- saves the model, metrics, and test predictions under `artifacts/security/`.

## Apply the trained model

```powershell
python predict_security_risk.py input.csv predictions.csv
```

The input must contain the same UNSW-NB15 predictor columns used during training.

## Current result

Random Forest was selected. On the official test set it achieved:

- accuracy: 0.8814
- balanced accuracy: 0.8698
- precision: 0.8311
- recall: 0.9846
- F1: 0.9014
- ROC-AUC: 0.9839
- PR-AUC: 0.9879
- false-positive rate: 0.2451

The model detects most attacks but still flags many normal flows. Do not describe its score
as a real-world attack probability or route by it until it is validated on traffic from the
target SDN environment. Approximately 10.37% of official test rows have feature vectors also
present in the supplied training file, so benchmark performance may overstate generalization.
