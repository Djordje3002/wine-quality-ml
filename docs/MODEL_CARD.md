# Model card — Wine Quality Lab

Version: 1.0.0 · Experiment seed: 42 · Selected model: `random_forest_leaf5`

## Intended use

Estimate the sensory quality of a Portuguese red wine from eleven laboratory measurements. The engineering goal is a small, inspectable regression pipeline that improves on predicting the mean score and prevents repeated measurements from inflating evaluation results.

The intended audience is engineers reviewing a reproducible ML portfolio and learners inspecting model selection, evaluation and inference. The project has no claimed deployment history or customer results.

## Training and evaluation

Exact duplicate rows are removed first. Identical feature vectors share a hash group, including cases with different target scores. A seeded group split reserves the test set; five-fold grouped cross-validation on the training partition chooses the lowest mean MAE. Scaling is fitted inside each pipeline and fold. The selected model is fitted once on training data, then evaluated on the reserved test partition.

| Model | mae | rmse | r2 |
| --- | ---: | ---: | ---: |
| `mean_baseline` | 0.7185 | 0.8581 | -0.0274 |
| `scaled_ridge` | 0.5549 | 0.7215 | 0.2737 |
| `random_forest_leaf2` | 0.5341 | 0.6904 | 0.3349 |
| `random_forest_leaf5` | 0.5366 | 0.6914 | 0.3330 |


## Interpretation

The forest with leaf size 5 was selected by training CV. The leaf-size-2 forest has slightly lower test MAE, but the selection stays unchanged. That difference is a useful example of why test results must not decide hyperparameters. Quality is an ordinal sensory judgment, and a fractional regression estimate is not a certification.

## Limitations

One regional wine population and a small dataset limit external validity. No vineyard, batch, or vintage identifiers are available to validate unseen producers or future vintages. Grouping identical measurements removes a specific leakage route; it does not establish independence between all samples. Permutation importance describes this fitted model and holdout, not causal effects.

## Artifacts and maintenance

Run the training command to rebuild local artifacts and reports. Load only your own trusted local artifacts. For any data or model change, rerun training, compare with the baseline under the same split protocol, and inspect failure cases. A changed dataset hash or split constitutes a different experiment and must be documented as such. Fresh labeled data is required to estimate real deployment performance.
