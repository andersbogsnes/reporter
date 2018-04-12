import pandas as pd
from bokeh.models import ColumnDataSource, LinearColorMapper, HoverTool
from bokeh.plotting import figure
from bokeh.transform import transform
from sklearn.metrics import confusion_matrix
from reporter.settings import COLORS
from reporter.metrics import Metric


class ConfusionMatrix(Metric):
    def __init__(self):
        super().__init__('confusion-matrix')

    def generate_data(self):
        matrix = confusion_matrix(self.y, self.y_pred)
        matrix = pd.DataFrame(matrix, index=self.labels, columns=self.labels)
        matrix.index.name = 'Predicted'
        matrix.columns.name = 'Actual'
        return pd.DataFrame(matrix.stack(), columns=['Value']).reset_index()

    def draw(self, size=400):
        index_label = 'Predicted'
        column_label = 'Actual'

        matrix = self.generate_data()
        min_val, max_val = matrix.Value.min(), matrix.Value.max()
        source = ColumnDataSource(matrix)
        mapper = LinearColorMapper(palette=COLORS, low=min_val, high=max_val)

        hover = HoverTool(tooltips=[
            ('Number', f"@Value")
        ])

        p = figure(plot_width=size,
                   plot_height=size,
                   title='Confusion Matrix',
                   tools=[hover],
                   toolbar_location=None,
                   x_range=self.labels,
                   y_range=list(reversed(self.labels)))

        p.yaxis.axis_label = index_label
        p.xaxis.axis_label = column_label

        p.rect(x=column_label,
               y=index_label,
               width=1,
               height=1,
               source=source,
               fill_color=transform('Value', mapper))
        self.plot = p
        return p
