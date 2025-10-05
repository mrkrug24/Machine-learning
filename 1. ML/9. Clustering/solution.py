import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances_chunked


def silhouette_score(x, labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    freq = np.bincount(labels)

    def _silhouette_reduce(d, start):
        dist = np.zeros((len(d), len(freq)), dtype=d.dtype)
        idx = (np.arange(len(d)), labels[start: start + len(d)])
        for i in range(len(d)):
            dist[i] += np.bincount(labels, weights=d[i], minlength=len(freq))

        dist_i, dist[idx] = dist[idx], np.inf
        return dist_i, (dist / freq).min(axis=1)

    dist_wc, dist_oc = zip(
        *pairwise_distances_chunked(x, reduce_func=_silhouette_reduce)
    )
    dist_wc, dist_oc = np.concatenate(dist_wc), np.concatenate(dist_oc)

    n = (freq - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_wc /= n

    sil = dist_oc - dist_wc
    with np.errstate(divide="ignore", invalid="ignore"):
        sil /= np.maximum(dist_wc, dist_oc)

    return np.mean(np.nan_to_num(sil))


def bcubed_score(true_labels, predicted_labels):
    corr = (predicted_labels[None, :] == predicted_labels[:, None]) & (
        true_labels[None, :] == true_labels[:, None]
    )
    prec = (
        np.sum(
            np.sum(corr, axis=1)
            / np.sum(predicted_labels[None, :] == predicted_labels[:, None], axis=1)
        )
        / true_labels.shape[0]
    )
    recall = (
        np.sum(
            np.sum(corr, axis=1)
            / np.sum(true_labels[None, :] == true_labels[:, None], axis=1)
        )
        / true_labels.shape[0]
    )
    return 2 * prec * recall / (prec + recall)
