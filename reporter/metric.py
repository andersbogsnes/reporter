# TODO Get classification report, confusion matrix into HTML

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LinearColorMapper, LabelSet, HoverTool
import pandas as pd
import numpy as np
from reporter.colors import COLORS
from bokeh.transform import transform
from jinja2 import Environment, PackageLoader

env = Environment(loader=PackageLoader('reporter', 'template'))


def generate_confusion_matrix_data(model, x, y, labels=None):
    labels = np.unique(y) if labels is None else labels

    y_pred = model.predict(x)
    matrix = confusion_matrix(y, y_pred)
    matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix.index.name = 'Predicted'
    matrix.columns.name = 'Actual'
    return pd.DataFrame(matrix.stack(), columns=['value']).reset_index()


def draw_heatmap(matrix, labels, index_label='Predicted', column_label='Actual', value_label='value', size=400):
    min_val, max_val = matrix[value_label].min(), matrix[value_label].max()
    source = ColumnDataSource(matrix)
    mapper = LinearColorMapper(palette=COLORS, low=min_val, high=max_val)
    hover = HoverTool(tooltips=[
        ('Number', f"@{value_label}")
    ])

    p = figure(plot_width=size,
               plot_height=size,
               title='Confusion Matrix',
               tools=[hover],
               toolbar_location=None,
               x_range=labels,
               y_range=list(reversed(labels)))

    p.yaxis.axis_label = index_label
    p.xaxis.axis_label = column_label

    p.rect(x=column_label,
           y=index_label,
           width=1,
           height=1,
           source=source,
           fill_color=transform(value_label, mapper))

    return p


def create_confusion_matrix(model, x, y, labels=None):
    matrix = generate_confusion_matrix_data(model, x, y, labels)
    return draw_heatmap(matrix, labels)


def generate_classification_report_data(model, x, y, labels=None):
    y_pred = model.predict(x)
    params = precision_recall_fscore_support(y, y_pred)
    param_names = ['precision', 'recall', 'f1-score', 'support']
    return pd.DataFrame(params, index=param_names, columns=labels).T



def render_report_html(model, x, y, labels, confusion_matrix=True):
    template_components = {}
    if confusion_matrix:
        conf_p = create_confusion_matrix(model, x, y, labels)
        script_confusion, div_confusion = components(conf_p)
        template_components['script_confusion'] = script_confusion
        template_components['div_confusion'] = div_confusion

    template = env.get_template('report.html')
    html = template.render(**template_components)
    with open('new_report.html', 'w') as f:
        f.write(html)


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression()
    clf.fit(X, y)

    labels = ['setosa', 'virginica', 'versace']

    render_report_html(clf, X, y, ['setosa', 'versicolor', 'virginica'])
