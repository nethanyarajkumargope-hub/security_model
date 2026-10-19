# UNSW-NB15 security model

This workspace contains a binary classifier for `normal` (`0`) versus `attack` (`1`).
It does not include GNN forecasting, genetic routing, or SDN-controller integration.

The original 42-feature UNSW-NB15 model remains the frozen benchmark. NF-UNSW-NB15-v2 uses a
different 43-feature NetFlow schema and is trained as a separate deployable-flow model with
`train_netflow_security_model.py`; the two datasets are not merged or presented as independent
because both derive from the same underlying UNSW-NB15 packet captures.

## Release status

Model development is complete for the supplied dataset scope. The primary frozen benchmark is
`artifacts/security/security_model.joblib`; its sibling manifest identifies the exact tested file.
`artifacts/netflow-security/security_model.joblib` is the separate flow-oriented variant. Both
artifacts have reproducible training scripts, saved evaluation reports, strict input validation,
and checksum verification before deserialization. Live SDN deployment remains a separate
integration task because the reference simulator does not emit either model's required features.

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

To include validation-only permutation feature analysis:

```powershell
python train_security_model.py --analyze-features
```

To check development stability without repeatedly evaluating the official test set:

```powershell
python run_security_multiseed.py
```

Verify that the saved model is the exact tested release artifact before loading it:

```powershell
python verify_security_release.py
```

After an intentional retraining and evaluation, regenerate the manifest with
`python verify_security_release.py --write` and review the changed metrics and checksum.

The training script:

- removes `id`, `attack_cat`, and `label` from the predictors;
- optionally removes exact repeated labeled training records while ignoring the unique `id`;
- keeps identical feature rows in the same development split;
- compares Logistic Regression, two focused Random Forest configurations, and Extra Trees on validation data;
- selects by balanced accuracy;
- calibrates the selected model on a separate calibration split;
- selects a decision threshold on another separate split while requiring at least 95% recall;
- records validation-only false-positive and false-negative breakdowns for error analysis;
- optionally measures validation-only permutation importance for the original input features;
- evaluates the frozen model on the official test CSV;
- saves the model, metrics, and test predictions under `artifacts/security/`.

## Apply the trained model

```powershell
python predict_security_risk.py input.csv predictions.csv
```

The input must contain the same UNSW-NB15 predictor columns used during training.
Extra metadata columns are ignored. Missing, duplicate, or nonnumeric required columns fail before
inference. The prediction, evaluation, and benchmark commands verify the sibling
`model_manifest.json` before loading the model. Use `--manifest` when supplying a model whose
manifest is stored elsewhere. Run `python benchmark_security_inference.py` to reproduce local
latency measurements.

Evaluate any labelled compatible dataset without fitting or changing the model:

```powershell
python evaluate_security_model.py labelled_flows.csv evaluation.json
```

The CSV must include both binary `label` values (`0` and `1`) in addition to the 42 predictors.
If `attack_cat` is present, the report also includes support, mean risk, and predicted-attack rate
by category.

## Reference simulator compatibility

The reference project defines security risk as `P(label=1)` clipped to `[0, 1]`, followed by a
thresholded binary decision. This workspace exposes the same contract through `security_risk.py`:

```python
from security_risk import attack_risk, threshold_risk
from verify_security_release import load_verified_model

artifact = load_verified_model("artifacts/security/security_model.joblib")
risk = attack_risk(flow_frame, artifact)
decision = threshold_risk(risk, artifact["threshold"])
```

Unlike the reference baseline, `attack_risk` uses the calibrated classifier, validates and orders
the saved 42-feature schema, locates label `1` from the estimator's class order, and uses the
threshold stored with the trained model. The simulator must still supply UNSW-compatible flow
records; aggregate node or edge telemetry cannot be passed to this function directly.

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

## NF-UNSW-NB15-v2 model

The supplied `NF-UNSW-NB15-v2.csv` contains 2,390,275 rows, 43 NetFlow predictors, 2,295,222
benign records, and 95,053 attacks. The streaming audit found no missing values, infinite values,
exact duplicate rows, or disagreements between `Label` and `Attack`. Its report is saved at
`artifacts/security/nf_unsw_nb15_v2_audit.json`.

IP identity is unsafe as a predictor in this dataset: approximately 95.25% of rows belong to source
or destination IP values that occur in only one label class. The NetFlow model therefore removes
both IP columns and uses 41 remaining predictors. Ten stratified folds are grouped by ordered
source-destination IP pair; six folds train the model and separate folds handle validation,
calibration, threshold tuning, and final held-out evaluation.

Train the separate model without changing the original benchmark artifact:

```powershell
python train_netflow_security_model.py "C:\path\to\archive.zip"
python verify_security_release.py `
  --model artifacts/netflow-security/security_model.joblib `
  --manifest artifacts/netflow-security/model_manifest.json
```

The HistGradientBoosting model selected a calibrated threshold of `0.158761`. On the untouched
IP-pair-held-out fold it achieved:

- accuracy: 0.9976
- balanced accuracy: 0.9980
- precision: 0.9460
- recall: 0.9985
- F1: 0.9715
- ROC-AUC: 0.9998
- PR-AUC: 0.9945
- false-positive rate: 0.0024

The held-out fold has 239,349 rows but only 15 IP pairs. These results show excellent separation
inside NF-UNSW-NB15-v2; they do not establish performance on another network. NF-UNSW-NB15-v2 and
the original UNSW-NB15 benchmark derive from the same packet captures, so this is a separate
deployment-oriented representation rather than independent external validation.

## Original raw UNSW-NB15 archive audit

The additional archive containing `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv` is the raw source of
the existing curated benchmark, not a new training dataset. The four files contain 2,540,047 CSV
records; the last record of each of the first three partitions is repeated as the first record of
the next partition, leaving 2,540,044 distinct boundary-adjusted records.

After removing IP addresses, ports, timestamps, and the derived `rate` column, every one of the
154,098 unique predictor/label combinations in the two curated CSVs occurs in the raw partitions.
The archive's two curated filenames are also reversed: its 82,332-row `training-set` file is
byte-identical to this project's testing file, and its 175,341-row `testing-set` file is
byte-identical to this project's training file. Training on the raw archive and reporting results
on the current test CSV would therefore leak test records. The archive is retained as a data-source
reference only and is not merged into either trained model.

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

## Validation permutation importance

Permuting one original input column at a time on the internal validation split produced the
following largest balanced-accuracy decreases:

| Feature | Mean decrease |
| --- | ---: |
| `service` | 0.013867 |
| `sbytes` | 0.010127 |
| `ct_dst_src_ltm` | 0.009323 |
| `synack` | 0.008119 |
| `ct_srv_dst` | 0.008068 |
| `tcprtt` | 0.007885 |
| `smean` | 0.007693 |
| `dmean` | 0.007657 |
| `ct_srv_src` | 0.006893 |
| `dload` | 0.005417 |

`service` being most important is consistent with the high error count for records whose service
is unnamed. Historical connection counters and TCP timing also contribute materially, so a later
deployable model cannot replace them with aggregate link QoS values without a controlled retraining
experiment.

## Multi-seed development stability

Seeds `42`, `123`, and `2026` independently repeated grouped splitting, candidate selection,
calibration, and threshold tuning using only the official training CSV. Random Forest was selected
for all three seeds.

| Seed | Selected threshold | Validation balanced accuracy | Tuning recall | Tuning false-positive rate |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.631777 | 0.946179 | 0.953719 | 0.065041 |
| 123 | 0.735327 | 0.944001 | 0.950756 | 0.051607 |
| 2026 | 0.634135 | 0.946533 | 0.953338 | 0.054387 |

Mean validation balanced accuracy was `0.945571` with standard deviation `0.001119`, indicating
stable model selection performance. Thresholds ranged from `0.631777` to `0.735327`, so the exact
operating threshold is less stable than the selected model and must remain part of the saved model
artifact rather than becoming a universal constant.

## Leaf-size experiment

A smoother Random Forest using `min_samples_leaf=4` was compared with the selected leaf-size-2
model across the same three development seeds. It improved mean initial validation balanced
accuracy from `0.945571` to `0.947638`, but performed worse on every independent threshold-tuning
split. Mean tuning balanced accuracy fell from `0.947796` to `0.946388`, recall fell from `0.952604`
to `0.951557`, and false-positive rate increased from `0.057012` to `0.058781`. The leaf-size-2
Random Forest therefore remains the baseline, and the official test set was not consulted.

## Local inference benchmark

The saved artifact is 32.64 MB. Ten warmed runs on this development machine produced these median
model-pipeline times, including its categorical encoding, imputation, and classification:

| Batch rows | Median time | Time per row | Throughput | Peak Python allocation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 42.58 ms | 42.58 ms | 23 rows/s | 0.26 MB |
| 100 | 56.42 ms | 0.564 ms | 1,773 rows/s | 0.38 MB |
| 1,000 | 85.58 ms | 0.086 ms | 11,685 rows/s | 1.80 MB |

These timings are machine-specific and favor batch inference. The allocation measurement comes from
Python's `tracemalloc`; it does not include the interpreter, native libraries, or the loaded model's
full process memory. It also excludes CSV loading and input-schema validation. Re-run
`python benchmark_security_inference.py` on deployment hardware before setting latency or capacity
requirements.
