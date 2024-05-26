import pandas as pd

from reporter.metrics import ClassificationReport
from reporter.model import Model
from tests.conftest import model


def test_can_draw_model(model: Model):
    metric = ClassificationReport(model)
    assert metric.draw()


def test_metrics_are_a_pandas_dataframe(model: Model):
    metric = ClassificationReport(model)
    data = metric.calculate()
    assert isinstance(data, pd.DataFrame)


def test_metrics_have_correct_shape(model: Model):
    metric = ClassificationReport(model)
    data = metric.calculate()
    assert data.shape == (3, 5)  # 3 labels * 4 scores + 1 label
