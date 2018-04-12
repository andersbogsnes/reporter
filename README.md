# Reporter Machine Learning reports
Reporter autogenerates machine learning reports with the metrics you define.
We use Bokeh to embed graphs into HTML documents, so you get beautiful, interactive graphs for free!

## Usage
>>> from reporter.metrics import ClassificationReport, ConfusionMatrix
>>> from reporter import Render