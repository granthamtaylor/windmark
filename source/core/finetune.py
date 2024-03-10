from math import isclose

class LabelBalancer:

    def __init__(self, labels: list[str], counts: list[int]):
        
        for count in counts:
            assert count > 0

        assert len(labels) == len(counts)

        self.size = len(labels)

        self.labels = labels
        self.counts = counts
        self.total = total = sum(counts)
        self.values = [count / total for count in counts]

        assert isclose(sum(self.values), 1.0)

    def interpolate(self, kappa: float):

        assert 0.0 <= kappa <= 1.0

        size = self.size
        null = 1 / size

        interpolation = [kappa * value + (1 - kappa) * null for value in self.values]

        assert isclose(sum(interpolation), 1.0)

        ratio = [interpol / value for interpol, value in zip(self.values, interpolation)]

        self.thresholds = list(map(lambda x: x / max(ratio), ratio))
        self.weights = list(map(lambda x: sum(ratio) / (x * size * size), interpolation))