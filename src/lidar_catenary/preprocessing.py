"""
Preprocessing utilities for LiDAR point clouds.
"""

import numpy as np
from sklearn.decomposition import PCA


def estimate_wire_directions(points: np.ndarray):
    """
    Estimate the main cable direction using PCA.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N,3) containing xyz coordinates.

    Returns
    -------
    direction : np.ndarray
        Main cable direction.
    perpendicular_direction : np.ndarray
        Direction perpendicular to cables.
    """

    xy = points[:, :2]

    pca = PCA(n_components=2)
    pca.fit(xy)

    direction = pca.components_[0]
    perp_direction = pca.components_[1]

    return direction, perp_direction


def project_perpendicular(points: np.ndarray, perp_direction: np.ndarray) -> np.ndarray:
    """
    Project points on the direction perpendicular to wires.

    Used to separate wires via clustering.

    Parameters
    ----------
    points : ndarray (N,3)

    perp_direction : ndarray (2,)

    Returns
    -------
    projection : ndarray (N,1)
    """

    xy = points[:, :2]
    projection = xy @ perp_direction

    return projection.reshape(-1, 1)
