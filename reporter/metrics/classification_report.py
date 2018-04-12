import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from reporter.metrics import Metric


class ClassificationReport(Metric):
    """Generates a table of classification metrics"""

    def __init__(self):
        super().__init__('classification-report')
        self.param_names = ['Precision', 'Recall', 'F1-score', 'Support']

    def generate_data(self) -> pd.DataFrame:
        params = list(precision_recall_fscore_support(self.y, self.y_pred))
        return pd.DataFrame(params, index=self.param_names, columns=self.labels).round(2).T

    def draw(self):
        table = self.generate_data()
        self.components['html'] = table.to_html(classes=['table'])
        return self
