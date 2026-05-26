from typing import List

class AverageMeter:
    def __init__(self, max_samples = None):
        self.max_samples = max_samples
        self.samples: List[float] = []

    def add_sample(self, value: float):
        self.samples.append(value)
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def average(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    def window_average(self, window_size: int) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples[-window_size:]) / min(window_size, len(self.samples))
