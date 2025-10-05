import numpy as np
import typing
from collections import defaultdict


def kfold_split(
    num_objects: int, num_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_size = num_objects // num_folds
    fold_remainder = num_objects % num_folds
    fold_indices = np.arange(num_objects)
    folds = []

    for i in range(num_folds):
        start = i * fold_size
        end = (
            (i + 1) * fold_size
            if i < num_folds - 1
            else (i + 1) * fold_size + fold_remainder
        )
        val_indices = fold_indices[start:end]
        train_indices = np.concatenate((fold_indices[:start], fold_indices[end:]))
        folds.append((train_indices, val_indices))

    return folds


def knn_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    parameters: dict[str, list],
    score_function: callable,
    folds: list[tuple[np.ndarray, np.ndarray]],
    knn_class: object,
) -> dict[str, float]:
    results = {}

    for normalizer, normalizer_name in parameters["normalizers"]:
        for n_neighbors in parameters["n_neighbors"]:
            for metric in parameters["metrics"]:
                for weight in parameters["weights"]:
                    avg_score = 0.0

                    for train_indices, test_indices in folds:
                        X_train = X[train_indices]
                        X_test = X[test_indices]
                        y_train = y[train_indices]
                        y_test = y[test_indices]
                        knn_model = knn_class(
                            n_neighbors=n_neighbors, metric=metric, weights=weight
                        )

                        if normalizer is not None:
                            normalizer.fit(X_train)
                            X_train, X_test = normalizer.transform(
                                X_train
                            ), normalizer.transform(X_test)

                        knn_model.fit(X_train, y_train)
                        pred = knn_model.predict(X_test)
                        score = score_function(y_test, pred)
                        avg_score += score

                    avg_score /= len(folds)
                    key = (normalizer_name, n_neighbors, metric, weight)
                    results[key] = avg_score

    return results
