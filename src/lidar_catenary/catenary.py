"""
Catenary mathematical model.
"""

import numpy as np


def catenary(s: np.ndarray, z0: float, c: float, s0: float) -> np.ndarray:
    """
    Catenary function.

    Parameters
    ----------
    s : ndarray
        Distance along the cable.

    z0 : float
        Lowest height.

    c : float
        Curvature parameter.

    s0 : float
        Horizontal offset of the lowest point.

    Returns
    -------
    ndarray
        Height values.
    """

    return z0 + c * (np.cosh((s - s0) / c) - 1)
