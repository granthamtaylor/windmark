from math import isclose

class LabelBalancer:
    def __init__(self, **kwargs: float):
        self.labels = list(kwargs.keys())
        self.values = list(kwargs.values())

        assert isclose(sum(self.values), 1.0)
        assert len(self.labels) == len(self.values)

        self.size = len(self.labels)

    def interpolate(self, kappa: float) -> tuple[list[float], list[float]]:
        assert 0.0 <= kappa <= 1.0

        null = 1 / self.size

        interpolation = [kappa * value + (1 - kappa) * null for value in self.values]

        assert isclose(sum(interpolation), 1.0)

        ratio = [interpol / value for interpol, value in zip(self.values, interpolation)]

        thresholds = list(map(lambda x: x / max(ratio)))
        weights = list(map(lambda x: sum(ratio) / (x * self.size), interpolation))

        return thresholds, weights
