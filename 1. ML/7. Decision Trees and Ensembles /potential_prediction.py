import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor


class PotentialTransformer:
    def fit(self, x, y):
        return self

    def fit_transform(self, x, y):
        return self.transform(x)

    def transform(self, x):
        arr_1, arr_2, res = x.shape[1], x.shape[2], []

        for iter in x:
            f_x, f_y, l_x, l_y = arr_1, arr_2, 0, 0

            for i in range(arr_1):
                for j in range(arr_2):
                    if iter[i][j] != 20:
                        f_x, l_x = min(i, f_x), max(i, l_x)
                        f_y, l_y = min(j, f_y), max(j, l_y)

            new_pic = np.full((arr_1, arr_2), 20)
            new_pic[
                int((arr_1 + f_x - l_x - 1) / 2): int((arr_1 + l_x + 1 - f_x) / 2),
                int((arr_2 + f_y - l_y - 1) / 2): int((arr_2 + l_y + 1 - f_y) / 2),
            ] = np.array(iter[f_x: l_x + 1, f_y: l_y + 1])
            res.append(new_pic)

        return np.sum(res, axis=1).reshape((len(res), -1))


def load_dataset(data_dir):
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    regressor = Pipeline(
        [
            ("trsfm", PotentialTransformer()),
            (
                "ETR",
                ExtraTreesRegressor(
                    max_depth=14,
                    random_state=39,
                    n_estimators=280,
                    max_features="sqrt",
                    criterion="friedman_mse",
                ),
            ),
        ]
    )
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
