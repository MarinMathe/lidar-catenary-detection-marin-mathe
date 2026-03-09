"""
Clustering methods for separating individual wires.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def estimate_number_of_wires(projection: np.ndarray, k_range=range(2, 15)) -> int:
    """
    Estimate optimal number of clusters using silhouette score.

    Parameters
    ----------
    projection : ndarray
        1D projection of points.

    Returns
    -------
    int
        Estimated number of wires.
    """

    scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(projection)

        score = silhouette_score(projection, labels)
        scores.append(score)

    best_k = k_range[scores.index(max(scores))]

    return best_k


def cluster_wires(projection: np.ndarray, k: int) -> np.ndarray:
    """
    Cluster projected points into individual wires.

    Parameters
    ----------
    projection : ndarray
    k : int

    Returns
    -------
    labels : ndarray
    """

    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(projection)

    return labels
