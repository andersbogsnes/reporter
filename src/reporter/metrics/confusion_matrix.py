import attrs
import pandas as pd
from bokeh.models import ColumnDataSource, LinearColorMapper, HoverTool
from bokeh.plotting import figure
from bokeh.transform import transform
from sklearn.metrics import confusion_matrix

from reporter.model import Model
from reporter.settings import COLORS


@attrs.define
class ConfusionMatrix:
    model: Model
    name: str = "Confusion Matrix"

    def calculate(self):
        matrix = confusion_matrix(self.model.y, self.model.y_pred)
        matrix = pd.DataFrame(matrix, index=self.model.labels, columns=self.model.labels)
        matrix.index.name = 'Predicted'
        matrix.columns.name = 'Actual'
        return pd.DataFrame(matrix.stack(), columns=['Value']).reset_index()

    def draw(self, size=400):
        index_label = 'Predicted'
        column_label = 'Actual'

        matrix = self.calculate()
        min_val, max_val = matrix.Value.min(), matrix.Value.max()
        source = ColumnDataSource(matrix)
        mapper = LinearColorMapper(palette=COLORS, low=min_val, high=max_val)

        hover = HoverTool(tooltips=[
            ('Number', f"@Value")
        ])

        p = figure(width=size,
                   height=size,
                   title='Confusion Matrix',
                   tools=[hover],
                   toolbar_location=None,
                   x_range=self.model.labels,
                   y_range=list(reversed(self.model.labels)))

        p.yaxis.axis_label = index_label
        p.xaxis.axis_label = column_label

        p.rect(x=column_label,
               y=index_label,
               width=1,
               height=1,
               source=source,
               fill_color=transform('Value', mapper))
        return p
