# LightGBM

## Data Interface

- LibSVM (zero-based) / TSV / CSV format text file

- NumPy 2D array(s), pandas DataFrame, H2O DataTable’s Frame, SciPy sparse matrix

  - I tried DataFrame, but somehow it failed.

- LightGBM binary file

- LightGBM Sequence object(s)

## Setting Parameters

- LightGBM can use a dictionary to set Parameters.

## Core Parameters

- objective, boosting, num_iterations, learning_rate, num_leaves...

## Learning Control Parameters

- max_depth, lambda_l1, lambda_l2

## IO Parameters

- linear_tree(default = false), max_bin

## Metric Parameters

- metric

## Training

## CV

- Training with n-fold CV

## Early Stopping

## Prediction

## Features

- Fast training speed and high efficiency

- Lower memory usage

- Better accuracy

- Parallel learning supported

- Capability of handling large-scalling data

- Support categorical feature directly
