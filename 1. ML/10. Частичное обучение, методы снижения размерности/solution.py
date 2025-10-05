import sklearn
import numpy as np
from sklearn.cluster import KMeans


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        super().__init__()
        self.mapping_ = None
        self.n_clusters = n_clusters

    def fit(self, data, labels):
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(data)
        self.mapping_, predicted_labels = self._best_fit_classification(
            kmeans.labels_, labels
        )
        return self

    def predict(self, data):
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(data)
        cluster_labels = kmeans.predict(data)
        predictions = np.empty(data.shape[0], dtype=int)
        for i in range(self.n_clusters):
            predictions[cluster_labels == i] = self.mapping_[i]
        return predictions

    def _best_fit_classification(self, cluster_labels, true_labels):
        n_clusters = self.n_clusters - (1 if -1 in cluster_labels else 0)
        map = np.zeros(self.n_clusters, dtype=int) - 1
        pred = np.copy(cluster_labels)
        x = list(set(cluster_labels))
        x.sort()
        for cluster in range(n_clusters):
            mask = cluster_labels == cluster
            y = np.arange(n_clusters)
            mask1 = y == cluster
            cluster_true_labels = true_labels[mask != 0]
            if -1 in cluster_true_labels:
                cluster_true_labels = cluster_true_labels[cluster_true_labels != -1]
            uniq, cnt = np.unique(cluster_true_labels, return_counts=True)
            if len(uniq) == 0 or (len(uniq) == 1 and uniq[0] == -1):
                z = np.copy(true_labels)
                if -1 in true_labels:
                    z = true_labels[true_labels != -1]
                uniq, cnt = np.unique(z, return_counts=True)
                max_cnt = cnt.max()
                max_labels = uniq[cnt == max_cnt]
                if len(max_labels) == 1:
                    pred[mask] = max_labels[0]
                    map[mask1] = max_labels[0]
                else:
                    min_label = max_labels.min()
                    pred[mask] = min_label
                    map[mask1] = min_label
                continue
            elif len(uniq) == 1:
                pred[mask] = uniq[0]
                map[mask1] = uniq[0]
            else:
                max_cnt = cnt.max()
                max_labels = uniq[cnt == max_cnt]
                if -1 in max_labels:
                    max_labels = max_labels[max_labels != -1]
                if len(max_labels) == 1:
                    pred[mask] = max_labels[0]
                    map[mask1] = max_labels[0]
                else:
                    min_label = max_labels.min()
                    pred[mask] = min_label
                    map[mask1] = min_label
        return map, pred
