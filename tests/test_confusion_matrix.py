import pandas as pd
import pytest
from sklearn import linear_model
from sklearn.datasets import load_iris

from reporter.metrics.confusion_matrix import ConfusionMatrix
from reporter.model import Model


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_iris(return_X_y=True, as_frame=True)
    return X, y


@pytest.fixture(name='classifier')
def _classifier(data: tuple[pd.DataFrame, pd.Series]) -> Model:
    X, y = data
    clf = linear_model.LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    model = Model(clf, X, y, labels=["setosa", "versicolor", "virginica"])
    return model


@pytest.fixture
def metric(classifier: Model) -> ConfusionMatrix:
    return ConfusionMatrix(model=classifier)


def test_confusion_matrix_draws_bokeh_chart(metric: ConfusionMatrix):
    assert metric.draw()


def test_confusion_matrix_creates_a_dataframe(metric: ConfusionMatrix):
    matrix = metric.calculate()
    assert isinstance(matrix, pd.DataFrame)


def test_confusion_matrix_has_correct_shape(metric: ConfusionMatrix):
    matrix = metric.calculate()
    assert matrix.shape == (3 * 3, 3)  # 3 predicted * 3 actual,
