"""
Visualization utilities for LiDAR point cloud and wire modeling.
"""

import matplotlib.pyplot as plt
import numpy as np

from .catenary import catenary


def plot_point_cloud_3d(points: np.ndarray) -> None:
    """
    Plot the full LiDAR point cloud in 3D.

    Parameters
    ----------
    points : np.ndarray (N,3)
        Array containing XYZ coordinates.
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("3D Point Cloud")
    plt.show()


def plot_projections(points: np.ndarray) -> None:
    """
    Plot 2D projections of the point cloud.

    XY, XZ, and YZ projections help understand
    the structure of the dataset.
    """

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].scatter(points[:, 0], points[:, 2], s=1)
    axs[0].set_title("X-Z projection")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Z")

    axs[1].scatter(points[:, 0], points[:, 1], s=1)
    axs[1].set_title("X-Y projection")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")

    axs[2].scatter(points[:, 1], points[:, 2], s=1)
    axs[2].set_title("Y-Z projection")
    axs[2].set_xlabel("Y")
    axs[2].set_ylabel("Z")

    plt.show()


def plot_height_distribution(points: np.ndarray) -> None:
    """
    Plot distribution of height values.

    Useful to identify cable altitude ranges.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(points[:, 2], bins=100)

    plt.title("Distribution of height (Z)")
    plt.xlabel("Height")
    plt.ylabel("Frequency")

    plt.show()


def plot_clusters_2d(points: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot wire clusters in the XY plane.

    Parameters
    ----------
    points : ndarray (N,3)
    labels : ndarray (N,)
    """

    plt.figure(figsize=(6, 6))

    plt.scatter(points[:, 0], points[:, 1], c=labels, s=2, cmap="tab10")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Wire clusters (2D)")

    plt.show()


def plot_clusters_3d(points: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot clustered wires in 3D.
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap="tab10", s=3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("Wire clusters in 3D")

    plt.show()


def plot_wire_fit(
    wire_points: np.ndarray,
    s_vals: np.ndarray,
    z_vals: np.ndarray,
    direction: np.ndarray,
    p0: np.ndarray,
    color="red",
) -> None:
    """
    Plot one wire with fitted catenary model.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(wire_points[:, 0], wire_points[:, 1], wire_points[:, 2], s=2)

    xy_line = p0 + np.outer(s_vals, direction)

    ax.plot(xy_line[:, 0], xy_line[:, 1], z_vals, color=color, linewidth=3)

    plt.title("Wire with fitted catenary")

    plt.show()


def plot_catenary_fit_planes(
    points: np.ndarray,
    labels: np.ndarray,
    wire_models: list,
    list_p0: list,
    list_s: list,
    direction: np.ndarray,
    colors: list,
) -> None:
    """
    Visualize the best-fit (s,z) plane used to fit the catenary model.

    For each wire, this function plots:
    - the wire points
    - the plane defined by the cable direction and vertical axis

    Parameters
    ----------
    points : ndarray (N,3)
        Point cloud.

    labels : ndarray
        Cluster labels for each point.

    wire_models : list
        Catenary parameters.

    list_p0 : list
        Reference points.

    list_s : list
        Cable coordinate values.

    direction : ndarray
        Cable direction.

    colors : array-like
        Colors for each wire.
    """

    vertical = np.array([0, 0, 1])

    for i in range(len(wire_models)):

        wire = points[labels == i]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # cable points
        ax.scatter(wire[:, 0], wire[:, 1], wire[:, 2], s=2, color=colors[i])

        p0 = list_p0[i]

        xy = wire[:, :2]
        z = wire[:, 2]

        s = list_s[i]

        s_vals = np.linspace(np.min(s), np.max(s), 20)
        z_vals = np.linspace(np.min(z), np.max(z), 20)

        S, Z = np.meshgrid(s_vals, z_vals)

        XY = p0 + np.outer(S.flatten(), direction)

        X = XY[:, 0].reshape(S.shape)
        Y = XY[:, 1].reshape(S.shape)

        ax.plot_surface(X, Y, Z, alpha=0.25, color=colors[i])

        ax.set_title(f"Cable {i} - best fit plane (s,z)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()


def plot_all_catenaries(
    points: np.ndarray,
    wire_models: list,
    list_p0: list,
    list_s: list,
    direction: np.ndarray,
    colors: list,
) -> None:
    """
    Plot the original LiDAR point cloud together with all fitted catenary models.

    Parameters
    ----------
    points : ndarray (N,3)
        Original LiDAR point cloud.

    wire_models : list
        List of fitted catenary parameters (z0, c, s0).

    list_p0 : list
        Reference point of each wire.

    list_s : list
        List of s coordinates for each wire.

    direction : ndarray (2,)
        Estimated cable direction.

    colors : array-like
        Colors used for each wire.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # original point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    for i in range(len(wire_models)):

        params = wire_models[i]
        z0, c, s0 = params

        s = list_s[i]

        s_vals = np.linspace(np.min(s), np.max(s), 200)
        z_vals = catenary(s_vals, z0, c, s0)

        p0 = list_p0[i]

        xy_line = p0 + np.outer(s_vals, direction)

        x = xy_line[:, 0]
        y = xy_line[:, 1]
        z = z_vals

        ax.plot(x, y, z, color=colors[i], linewidth=3, label=f"Wire {i}")

    ax.legend()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("Fitted catenary models")

    plt.show()
