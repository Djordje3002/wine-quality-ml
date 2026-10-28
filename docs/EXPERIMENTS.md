# Experiment protocol

## Reproduce

1. Install the project in a fresh Python 3.11 environment with `python -m pip install -e ".[dev]"`.
2. Run `python -m wine_quality train` from the repository root.
3. Inspect `reports/metrics.json` and the generated figures. The report records seed, Python version, installed library versions and dataset source information.
4. Run `python -m wine_quality predict --input examples/sample.json` to exercise the trained model.
5. Run `make check test` to verify the contracts.

## Selection rule

Exact duplicate rows are removed first. Identical feature vectors share a hash group, including cases with different target scores. A seeded group split reserves the test set; five-fold grouped cross-validation on the training partition chooses the lowest mean MAE. Scaling is fitted inside each pipeline and fold. The selected model is fitted once on training data, then evaluated on the reserved test partition.

## Reading the result

The forest with leaf size 5 was selected by training CV. The leaf-size-2 forest has slightly lower test MAE, but the selection stays unchanged. That difference is a useful example of why test results must not decide hyperparameters. Quality is an ordinal sensory judgment, and a fractional regression estimate is not a certification.

The committed results are one seeded benchmark run. They do not establish confidence intervals, statistical significance, robustness across random seeds or production behavior. CV standard deviations summarize folds, not uncertainty on future production metrics. Do not tune settings against the final test set; use new validation experiments and reserve a fresh holdout if making repeated test-driven changes.

## Notebook

`notebooks/01_review_results.ipynb` reads committed reports and displays the evaluation figure. It is executed and can be viewed directly on GitHub without training. To rerun it locally, install `python -m pip install -e ".[notebook]"` and select that environment in your notebook editor. Training logic remains in the package so notebook execution cannot silently diverge from the command-line experiment.

## Resource boundaries

All examples run on CPU without cloud accounts or paid APIs. The default dataset archives are small, and raw data is cached locally. The CNN uses two CPU threads; the wine forest uses two workers. Reports and small images are committed, while model binaries, raw archives and environment folders are ignored. CI has a 15-minute execution limit.
