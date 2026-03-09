"""
Input / output utilities for LiDAR datasets.
"""

from pathlib import Path

import pandas as pd


def load_point_cloud(path: str) -> pd.DataFrame:
    """
    Load LiDAR point cloud stored in a parquet file.

    Parameters
    ----------
    path : str
        Path to the parquet file containing the point cloud.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['x', 'y', 'z'].
    """
    return pd.read_parquet(Path(path))
