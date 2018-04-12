class Metric:
    def __init__(self, name):
        self.model = None
        self.y = None
        self.y_pred = None
        self.y_proba = None
        self.components = {"name": name}
        self.labels = None
        self.plot = None

    def init_model(self, model, **kwargs):
        self.model = model
        self.y = model.y
        self.y_pred = model.y_pred
        self.y_proba = model.y_proba
        self.labels = model.labels

        self.draw(**kwargs)
        return self

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def generate_data(self):
        raise NotImplementedError

    def draw(self, **kwargs):
        raise NotImplementedError

    def add_text(self, text):
        self.components['text'] = text
        return self

    def add_html(self, html):
        if 'html' not in self.components:
            self.components['html'] = html
        return self
