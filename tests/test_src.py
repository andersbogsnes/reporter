import pytest
import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_iris
from src.metric import generate_confusion_matrix
import pandas as pd


@pytest.fixture(name='regressor')
def _regressor():
    x = np.linspace(0, 100).reshape(-1, 1)
    y = np.logspace(0, 5).reshape(-1, 1)
    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    return clf


@pytest.fixture(name='classifier')
def _classifier():
    X, y = load_iris(return_X_y=True)
    clf = linear_model.LogisticRegression()
    clf.fit(X, y)
    return clf


def test_model_returns_fitted_model(regressor):
    assert hasattr(regressor, 'coef_')


def test_generates_report(regressor):
    pass


def test_confusion_matrix_generates_correct_dataframe(classifier):
    X, y = load_iris(return_X_y=True)
    labels = ['setosa', 'versicolor', 'virginica']
    matrix = generate_confusion_matrix(classifier, X, y, labels=labels)
    assert isinstance(matrix, pd.DataFrame)
    assert labels == list(matrix.columns)
    assert labels == list(matrix.index)
    assert 3 == len(matrix)
