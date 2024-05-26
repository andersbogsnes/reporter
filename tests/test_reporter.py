import pathlib

from reporter import Reporter
from reporter.metrics import ConfusionMatrix, ClassificationReport
from reporter.model import Model


def test_reporter_writes_report(model: Model, tmp_path: pathlib.Path):
    reporter = Reporter(model,
                        output_path=tmp_path,
                        metrics=[ConfusionMatrix(model), ClassificationReport(model)])
    reporter.output()
    assert pathlib.Path(tmp_path / "report.html").exists()
