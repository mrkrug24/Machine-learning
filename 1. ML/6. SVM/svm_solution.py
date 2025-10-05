from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def train_svm_and_predict(train_features, train_target, test_features):
    grid_search = GridSearchCV(
        SVC(),
        {
            "C": [0.5, 1, 1.5, 2, 2.5, 3],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
        },
        cv=2,
    )
    grid_search.fit(train_features, train_target)
    return grid_search.best_estimator_.predict(test_features)
