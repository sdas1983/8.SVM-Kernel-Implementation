# SVM Kernel Implementation

This repository contains a Python implementation of Support Vector Machine (SVM) using different kernel functions. The project demonstrates the classification of data points using the Radial Basis Function (RBF) and Linear kernels.

## Overview

The code generates two sets of circular data points, representing two different classes. The data points are classified using SVM with RBF and Linear kernels. The project also includes the creation of additional features to support polynomial kernel implementation, although the primary focus is on RBF and Linear kernels.

## Files

- `svm_kernel_implementation.py`: The main script containing the SVM implementation and data visualization.

## Requirements

To run the code, you need the following Python libraries installed:

- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `plotly`

You can install the required packages using pip:

```bash
pip install numpy matplotlib pandas scikit-learn plotly
```

## Data Visualization
The project includes the following visualizations:

- **Scatter Plots**: Displaying the circular data points for two different classes.
- **3D Scatter Plots**: Visualizing the feature space for polynomial kernels using Plotly.

## Key Functions
- **Data Generation**: Creates two circular data sets representing different classes.
- **Feature Engineering**: Generates additional features for polynomial kernel implementation.
- **SVM Training**: Trains SVM models using RBF and Linear kernels.
- **Visualization**: Plots the data points and feature space using Matplotlib and Plotly.

## Results
The accuracy of the SVM models is displayed after training. The visualizations provide insights into the distribution of data points and the decision boundaries formed by the SVM models.
