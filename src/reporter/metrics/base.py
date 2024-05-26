from typing import Protocol


class Metric(Protocol):
    def calculate(self):
        ...

    def draw(self):
        ...