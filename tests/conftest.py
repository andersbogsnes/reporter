import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.utils import Bunch

from reporter.model import Model


@pytest.fixture()
def iris_data() -> Bunch:
    return load_iris(as_frame=True)



@pytest.fixture
def estimator(iris_data: Bunch) -> LogisticRegression:
    X, y = iris_data.data, iris_data.target
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf


@pytest.fixture
def model(estimator: LogisticRegression, iris_data: Bunch) -> Model:
    X, y = iris_data.data, iris_data.target
    return Model(estimator, X, y, labels=iris_data.target_names)
