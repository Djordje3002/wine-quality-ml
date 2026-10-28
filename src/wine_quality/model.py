"""Grouped validation prevents identical measurements crossing split boundaries."""

import io
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import archive, figure, parity, regression, save_report


def split_data(frame):
    frame = frame.drop_duplicates().reset_index(drop=True)
    X, y = frame.drop(columns="quality"), frame.quality
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups)
    )
    return X, y, groups, train, test


def train(root):
    with archive(root) as source:
        frame = pd.read_csv(io.BytesIO(source.read("winequality-red.csv")), sep=";")
    X, y, groups, fit, test = split_data(frame)
    candidates = {
        "mean_baseline": DummyRegressor(),
        "scaled_ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "random_forest_leaf2": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=2
        ),
        "random_forest_leaf5": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=5, random_state=42, n_jobs=2
        ),
    }
    cv = {}
    for name, model in candidates.items():
        scores = -cross_val_score(
            model,
            X.iloc[fit],
            y.iloc[fit],
            groups=groups[fit],
            cv=GroupKFold(5),
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        cv[name] = {"mean_mae": float(scores.mean()), "std_mae": float(scores.std())}
    selected = min(cv, key=lambda name: cv[name]["mean_mae"])
    results = {}
    for name, model in candidates.items():
        model.fit(X.iloc[fit], y.iloc[fit])
        results[name] = regression(y.iloc[test], model.predict(X.iloc[test]))
    model = candidates[selected]
    importance = permutation_importance(
        model,
        X.iloc[test],
        y.iloc[test],
        scoring="neg_mean_absolute_error",
        n_repeats=8,
        random_state=42,
    )
    order = np.argsort(importance.importances_mean)
    plt.figure(figsize=(7, 5))
    plt.barh(X.columns[order], importance.importances_mean[order], color="#2563eb")
    plt.xlabel("Test MAE increase after feature permutation")
    plt.title("Descriptive feature analysis; no model selection")
    figure(root, "feature_importance.png")
    parity(root, y.iloc[test], model.predict(X.iloc[test]))
    result = save_report(
        root,
        {
            "raw_rows": len(frame),
            "unique_rows": len(X),
            "split": {
                "train": len(fit),
                "test": len(test),
                "method": "feature-group holdout + 5-fold grouped CV",
            },
            "selected": selected,
            "cross_validation": cv,
            "test": results,
        },
    )
    joblib.dump({"model": model, "features": X.columns.tolist()}, root / "artifacts/model.joblib")
    (root / "reports/split.json").write_text(
        json.dumps({"train": fit.tolist(), "test": test.tolist()}, indent=2)
    )
    return result


def predict(root, payload):
    artifact = joblib.load(root / "artifacts/model.joblib")
    if set(payload) != set(artifact["features"]):
        raise ValueError("Provide exactly the feature names in examples/sample.json")
    X = pd.DataFrame([payload], columns=artifact["features"]).astype(float)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("All features must be finite numbers")
    return {
        "predicted_quality": float(artifact["model"].predict(X)[0]),
        "scale": "0–10 sensory score; regression estimate",
    }
