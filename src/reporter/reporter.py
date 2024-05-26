import pathlib
from typing import Any

import attrs
import pandas as pd
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Environment, PackageLoader

from reporter.metrics import Metric, ClassificationReport, ConfusionMatrix
from reporter.model import Model, ScikitModel
from reporter.settings import COLOR_SCHEME

DEFAULT_SIZE = 400


@attrs.define
class Reporter:
    model: Model
    metrics: list[Metric] = attrs.field(default=list)
    output_path: pathlib.Path = attrs.field(factory=lambda: pathlib.Path.cwd() / 'report')
    _env: Environment = Environment(loader=PackageLoader('reporter', 'template'))

    def output(self) -> None:
        html = self._render_html()
        self._write_report(html)

    def _create_bokeh_html(self) -> tuple[str, list[dict[str, Any]]]:
        plots = [metric.draw() for metric in self.metrics]
        script, htmls = components(plots)
        metrics = [{"name": metric.name, "html": html} for html, metric in zip(htmls, self.metrics)]
        return script, metrics

    def _render_html(self) -> str:
        js = INLINE.render_js()
        css = INLINE.render_css()
        script, metrics = self._create_bokeh_html()
        template = self._env.get_template('report.html')
        html = template.render(metrics=metrics,
                               css=css,
                               js=js,
                               script=script,
                               name=self.model.name,
                               color_scheme=COLOR_SCHEME)
        return html

    def _write_report(self, html: str):
        if not self.output_path.exists():
            self.output_path.mkdir()

        self.output_path.joinpath('Report.html').write_text(html)

    @classmethod
    def from_model(cls, estimator: ScikitModel, x: pd.DataFrame, y: pd.Series, labels: list[str] | None = None):
        model = Model(estimator, x, y, labels)
        metrics = [ClassificationReport(model), ConfusionMatrix(model)]
        return cls(model, metrics=metrics)
