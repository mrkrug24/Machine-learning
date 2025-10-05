import numpy as np


def evaluate_measures(sample):
    sample.sort()
    x = np.array(sample)
    c = np.unique(x)
    p = np.array(
        [(float(len(np.where(x == c[i])[0])) / len(sample)) for i in range(len(c))]
    )
    return {
        "gini": 1 - np.sum(p * p),
        "entropy": -np.sum(p * np.log(p)),
        "error": 1 - np.amax(p),
    }
