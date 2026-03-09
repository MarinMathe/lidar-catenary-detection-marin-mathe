"""
Catenary model fitting utilities.
"""

import numpy as np
from scipy.optimize import curve_fit

from .catenary import catenary


def fit_wire_catenary(points: np.ndarray, direction: np.ndarray):
    """
    Fit a catenary model to a single wire.

    Parameters
    ----------
    points : ndarray (N,3)

    direction : ndarray (2,)
        Estimated cable direction.

    Returns
    -------
    params : tuple
        (z0, c, s0)

    p0 : ndarray
        Reference point of the cable.

    s : ndarray
        Distance along cable for each point.
    """

    xy = points[:, :2]
    z = points[:, 2]

    p0 = xy.mean(axis=0)

    s = (xy - p0) @ direction

    params, _ = curve_fit(catenary, s, z)

    return params, p0, s
