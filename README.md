# LiDAR Powerline Catenary Detection

This project implements a Python pipeline to automatically detect powerline wires in LiDAR point clouds and fit 3D catenary models to each cable.

The goal is to identify the number of wires present in a LiDAR dataset and reconstruct their physical shape using the catenary equation.


# Problem

Electricity network operators collect LiDAR point clouds using drones to inspect power lines.

The objective is to automatically:

1. Detect how many wires are present
2. Separate points belonging to each wire
3. Fit a **catenary model** describing the cable shape

A catenary describes the curve formed by a cable suspended under its own weight.

The equation used is:

$$
z = z_0 + c \left( \cosh \frac{s - s_0}{c} - 1 \right)
$$

Where:

- **s** = distance along the cable
- **z0** = lowest point of the cable
- **c** = curvature parameter
- **s0** = horizontal offset



# Approach

The pipeline follows several steps.

## 1. Load LiDAR data

Input datasets are stored as `.parquet` files representing the 3D coordinates of each LiDAR point.

## 2. Exploratory data analysis

Several visualizations are used to understand the dataset:

- 3D point cloud visualization
- XY / XZ / YZ projections
- height distribution

These help validate assumptions about wire geometry.



## 3. Estimate wire orientation

Powerlines are almost straight in the horizontal plane.

We estimate their direction using **Principal Component Analysis (PCA)** on the XY coordinates.

This provides:

- main cable direction
- perpendicular direction



## 4. Separate individual wires

Points are projected onto the **perpendicular direction**.

Since wires are separated laterally, this produces distinct clusters.

We then apply **K-Means clustering** to identify individual wires.

The number of wires is automatically estimated using the **silhouette score**.



## 5. Fit catenary models

For each wire cluster:

1. Points are projected along the cable direction
2. The catenary equation is fitted using **non-linear least squares**
3. Parameters `(z0, c, s0)` are estimated using `scipy.optimize.curve_fit`



## 6. Reconstruct 3D wire models

The fitted catenary is projected back into 3D space to reconstruct the cable geometry.

The final result is visualized together with the original LiDAR points.


# Project Structure
```bash
lidar-catenary-detection
│
├── data
│
├── notebooks
│ └── exploration.ipynb
│
├── scripts
│ └── run_pipeline.py
│
├── src
│ └── lidar_catenary
│ ├── init.py
│ ├── io.py
│ ├── preprocessing.py
│ ├── clustering.py
│ ├── catenary.py
│ ├── modeling.py
│ └── visualization.py
│
├── requirements.txt
└── README.md
```



# Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/lidar-catenary-detection
cd lidar-catenary-detection
```


Install dependencies:
```bash
pip install -r requirements.txt
```


# Running the pipeline

To run the full pipeline:
```bash
python scripts/run_pipeline.py
```


This will:

1. Load the LiDAR dataset
2. Visualize the point cloud
3. Estimate wire orientation
4. Cluster points into wires
5. Fit catenary models
6. Visualize the final reconstructed cables



# Exploratory Notebook

The notebook located in:
```bash
notebooks/exploration.ipynb
```
contains the exploratory analysis used to develop the pipeline.

It reproduces the full workflow step-by-step using the functions implemented in the Python package.









