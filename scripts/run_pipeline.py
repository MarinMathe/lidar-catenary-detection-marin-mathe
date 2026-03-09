"""
Main pipeline for LiDAR wire detection and catenary fitting.

Pipeline steps
--------------
1. Load LiDAR point cloud
2. Visualize raw data
3. Estimate cable direction (PCA)
4. Project points perpendicular to cables
5. Estimate number of wires
6. Cluster wires
7. Fit catenary model for each wire
8. Visualize individual fits
9. Visualize fitting planes
10. Visualize final reconstructed cables
"""

import sys
from pathlib import Path

# Add src/ to path dynamically (works on any OS)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import numpy as np

from lidar_catenary.catenary import catenary
from lidar_catenary.clustering import cluster_wires, estimate_number_of_wires
from lidar_catenary.io import load_point_cloud
from lidar_catenary.modeling import fit_wire_catenary
from lidar_catenary.preprocessing import estimate_wire_directions, project_perpendicular
from lidar_catenary.visualization import (
    plot_all_catenaries,
    plot_catenary_fit_planes,
    plot_clusters_2d,
    plot_clusters_3d,
    plot_height_distribution,
    plot_point_cloud_3d,
    plot_projections,
    plot_wire_fit,
)


def main():

    # ============================================================
    # 1 Load LiDAR dataset
    # ============================================================

    DATA_PATH = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "lidar_cable_points_extrahard.parquet"
    )
    df = load_point_cloud(DATA_PATH)

    points = df[["x", "y", "z"]].values

    print(f"Loaded {points.shape[0]} points")

    # ============================================================
    # 2 Exploratory visualizations
    # ============================================================

    plot_point_cloud_3d(points)

    plot_projections(points)

    plot_height_distribution(points)

    # ============================================================
    # 3 Estimate wire direction with PCA
    # ============================================================

    direction, perp_direction = estimate_wire_directions(points)

    print("Estimated cable direction:", direction)

    # ============================================================
    # 4 Project points to separate wires
    # ============================================================

    projection = project_perpendicular(points, perp_direction)

    plt.figure()

    plt.scatter(projection, points[:, 2], s=1)

    plt.xlabel("Perpendicular projection")
    plt.ylabel("Height (z)")
    plt.title("Projection used for clustering")

    plt.show()

    # ============================================================
    # 5 Estimate number of wires
    # ============================================================

    k = estimate_number_of_wires(projection)

    print("Estimated number of wires:", k)

    # ============================================================
    # 6 Cluster wires
    # ============================================================

    labels = cluster_wires(projection, k)

    plot_clusters_2d(points, labels)

    plot_clusters_3d(points, labels)

    # ============================================================
    # 7 Fit catenary models
    # ============================================================

    wire_models = []
    list_p0 = []
    list_s = []

    unique_labels = np.unique(labels)

    colors = ["orange", "red", "green", "pink", "purple", "yellow", "cyan"]

    for i, label in enumerate(unique_labels):

        wire_points = points[labels == label]

        params, p0, s_vals = fit_wire_catenary(wire_points, direction)

        wire_models.append(params)
        list_p0.append(p0)
        list_s.append(s_vals)

        z0, c, s0 = params

        s_plot = np.linspace(np.min(s_vals), np.max(s_vals), 200)
        z_plot = catenary(s_plot, z0, c, s0)

        # 3D visualization of fit
        plot_wire_fit(wire_points, s_plot, z_plot, direction, p0, color=colors[i])

        # 2D diagnostic plot
        plt.figure()

        plt.scatter(s_vals, wire_points[:, 2], s=3, label="points")

        plt.plot(s_plot, z_plot, color=colors[i], linewidth=3, label="catenary fit")

        plt.xlabel("Distance along cable (s)")
        plt.ylabel("Height (z)")
        plt.title(f"Wire {i} – Catenary fit")

        plt.legend()

        plt.show()

    # ============================================================
    # 8 Visualize fitting planes
    # ============================================================

    plot_catenary_fit_planes(
        points, labels, wire_models, list_p0, list_s, direction, colors
    )

    # ============================================================
    # 9 Final visualization
    # ============================================================

    plot_all_catenaries(points, wire_models, list_p0, list_s, direction, colors)

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
