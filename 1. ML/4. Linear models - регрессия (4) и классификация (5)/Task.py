import numpy as np


class Preprocesser:
    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocesser):
    def __init__(self, dtype=np.float64):
        super().__init__()
        self.dtype = dtype
        self.d = {}

    def fit(self, X, Y=None):
        x = np.asarray(X)
        self.d = {i: np.unique(x[:, i], return_inverse=True) for i in range(x.shape[1])}

    def transform(self, X):
        assert X.shape[1] == len(self.d), "Size error"
        x = np.asarray(X)
        return np.concatenate(
            [np.eye(self.d[i][0].shape[0])[self.d[i][1]] for i in range(x.shape[1])],
            axis=1,
        )

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:
    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.d = {}

    def fit(self, X, Y):
        assert len(Y.shape) == 1, "size error"
        assert X.shape == np.array(X).shape, "size error"
        x, y = np.asarray(X), np.asarray(Y)
        self.d = {
            i: {
                j: np.array(
                    [
                        np.sum(y[col == j]) / len(x[col == j]),
                        len(x[col == j]) / x.shape[0],
                        0,
                    ]
                )
                for j in np.unique(col)
            }
            for i, col in enumerate(x.T)
        }

    def transform(self, X, a=1e-5, b=1e-5):
        x = np.asarray(X)
        res = np.zeros((x.shape[0], x.shape[1] * 3))

        for i, j in enumerate(x.T):
            for k in np.unique(j):
                z = j == k
                self.d[i][k][2] = (self.d[i][k][0] + a) / (self.d[i][k][1] + b)
                res[z, 3 * i: 3 * (1 + i)] = self.d[i][k]
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack(
            (idx[: i * n_], idx[(i + 1) * n_:])
        )
    yield idx[(n_splits - 1) * n_:], idx[: (n_splits - 1) * n_]


class FoldCounters:
    def __init__(self, n_folds=3, dtype=np.float64):
        self.X = None
        self.section = []
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        assert len(Y.shape) == 1, "size error"
        for i in group_k_fold(X.shape[0], self.n_folds, seed):
            self.section.append(i)
        self.X = np.zeros((X.shape[0], X.shape[1] * 2), dtype=np.float64)
        list = X.values
        for i in self.section:
            for j in range(list.shape[1]):
                val = list[i[0], j]
                info = list[i[1], j]
                for k in np.unique(val):
                    p = np.where(info == k)[0]
                    self.X[i[0][np.where(val == k)[0]], j * 2: j * 2 + 2] = (
                        Y.values[i[1]][p].sum() / p.shape[0],
                        p.shape[0] / i[1].shape[0],
                    )

    def transform(self, X, a=1e-5, b=1e-5):
        res = np.zeros((self.X.shape[0], 0), dtype=np.float64)
        for i in range(0, self.X.shape[1], 2):
            res = np.hstack(
                (
                    res,
                    self.X[:, i].reshape(-1, 1),
                    self.X[:, i + 1].reshape(-1, 1),
                    ((self.X[:, i] + a) / (self.X[:, i + 1] + b)).reshape(-1, 1),
                )
            )
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    res = np.zeros(np.unique(x).shape[0])
    for i, j in enumerate(np.unique(x)):
        res[i] = len(y[(y == 1) & (x == j)]) / (
            len(y[(y == 1) & (x == j)]) + len(y[(y == 0) & (x == j)])
        )
    return res
