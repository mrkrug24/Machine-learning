import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray) -> None:
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min) / (self.max - self.min)


class StandardScaler:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit(self, data: np.ndarray) -> None:
        self.std = data.std(axis=0)
        self.mean = data.mean(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
