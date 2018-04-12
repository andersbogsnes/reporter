import pathlib

from bokeh.embed import components
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from jinja2 import Environment, PackageLoader
from bokeh.resources import INLINE

from reporter.settings import COLOR_SCHEME, REPORT_PATH
from reporter.metrics.confusion_matrix import ConfusionMatrix
from reporter.metrics.classification_report import ClassificationReport
import numpy as np
DEFAULT_SIZE = 400


class Model:
    def __init__(self, model, x, y, labels=None):
        self.model = model
        self.x = x
        self.y = y
        self.labels = np.unique(y) if labels is None else labels
        self.y_pred = model.predict(x)
        self.y_proba = None
        self.name = model.__class__.__name__

        if hasattr(model, 'pred_proba'):
            self.y_proba = model.pred_proba(x)


class Render:
    def __init__(self, model, metrics=None):
        self.env = Environment(loader=PackageLoader('reporter', 'template'))
        self.model = model
        self.metrics = self.init_metrics(metrics)

    def init_metrics(self, metrics):
        if metrics is None:
            raise ValueError("Must pass metrics")
        for metric in metrics:
            metric.init_model(self.model)
        return metrics

    def create_bokeh_html(self):
        plot_metrics = [metric for metric in self.metrics if metric.plot]
        plots = [metric.plot for metric in plot_metrics]
        script, htmls = components(plots)
        for metric, html in zip(plot_metrics, htmls):
            metric.add_html(html)
        return script

    def render_html(self):
        js = INLINE.render_js()
        css = INLINE.render_css()
        script = self.create_bokeh_html()
        metrics = [metric.components for metric in self.metrics]
        template = self.env.get_template('report.html')
        html = template.render(metrics=metrics,
                               css=css,
                               js=js,
                               script=script,
                               name=model.name,
                               color_scheme=COLOR_SCHEME)
        self.write_html(html)

    def write_html(self, html):
        current_path = pathlib.Path(REPORT_PATH).joinpath('report')
        if not current_path.exists():
            current_path.mkdir()

        current_path.joinpath('Report.html').write_text(html)

    @classmethod
    def from_model(cls, model, x, y, labels=None):
        metrics = [ClassificationReport, ConfusionMatrix]
        model = Model(model, x, y, labels)
        return cls(model, metrics)


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression()
    clf.fit(X, y)

    labels = ['setosa', 'virginica', 'versace']
    model = Model(clf, X, y, labels)
    cr = ClassificationReport()
    cr.add_text('This is a test')
    metrics = [cr, ConfusionMatrix()]

    Render(model, metrics).render_html()
