import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from wine_quality.model import predict, split_data


def test_identical_features_stay_together_even_with_conflicting_targets():
    frame = pd.DataFrame({"alcohol": list(range(30)) * 2, "quality": [5] * 30 + [6] * 30})
    X, y, groups, fit, test = split_data(frame)
    assert set(groups[fit]).isdisjoint(groups[test])
    assert len(fit) + len(test) == len(X) == 60
    assert len(y) == len(groups)


def test_exact_duplicate_rows_removed_before_split():
    frame = pd.DataFrame({"alcohol": list(range(30)) * 2, "quality": [5] * 60})
    assert len(split_data(frame)[0]) == 30


@pytest.fixture
def trained_root(tmp_path):
    X = pd.DataFrame({"alcohol": [8.0, 9.0, 10.0]})
    model = DummyRegressor().fit(X, [4.0, 5.0, 6.0])
    (tmp_path / "artifacts").mkdir()
    joblib.dump({"model": model, "features": ["alcohol"]}, tmp_path / "artifacts/model.joblib")
    return tmp_path


def test_prediction_uses_named_columns(trained_root):
    assert predict(trained_root, {"alcohol": 9.5})["predicted_quality"] == pytest.approx(5.0)


@pytest.mark.parametrize("payload", [{}, {"alcohol": 10.0, "quality": 8}, {"alcohol": np.nan}])
def test_invalid_feature_contract_rejected(trained_root, payload):
    with pytest.raises(ValueError):
        predict(trained_root, payload)
