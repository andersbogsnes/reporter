import attrs
import pandas as pd
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from sklearn.metrics import precision_recall_fscore_support

from reporter.model import Model


@attrs.define
class ClassificationReport:
    """Generates a table of classification metrics"""

    model: Model
    name: str = "Classification Report"

    def calculate(self) -> pd.DataFrame:
        param_names = ['Precision', 'Recall', 'F1-score', 'Support']
        params = list(precision_recall_fscore_support(self.model.y, self.model.y_pred))
        return (pd.DataFrame(params, columns=self.model.labels)
                .round(2)
                .T
                .set_axis(param_names, axis='columns')
                .reset_index()
                .rename(columns={"index": "Label"}))

    def draw(self):
        source = ColumnDataSource(self.calculate())
        columns = [
            TableColumn(field="Label"),
            TableColumn(field='Precision'),
            TableColumn(field='Recall'),
            TableColumn(field='F1-score'),
            TableColumn(field='Support'),

        ]
        table = DataTable(source=source, columns=columns, index_position=None)
        return table
