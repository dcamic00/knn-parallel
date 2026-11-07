# Parallel k-NN Classifier (OpenMP + SIMD)

## Description
This repository contains a parallel implementation of the exact k-Nearest Neighbors (k-NN) classification algorithm using OpenMP and SIMD vectorization in C++. This project is part of the Mid-term assignments for the *Parallel Programming for Machine Learning* course.

To maximize performance, facilitate SIMD vectorization, and optimize cache locality, the data structure has been transformed into a Structure of Arrays (SoA) memory layout.

## Dataset
The classifier is evaluated on the MNIST dataset of handwritten digits (784 dimensions).
- **Source**: [OpenML MNIST](https://www.openml.org/d/554)
- **Required files**: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`.

## Compilation & Execution
To compile the source code applying performance optimizations (`-O3`), multithreading support (`-fopenmp`), and architecture-specific vectorization (`-march=native`), run:

```bash
g++ -O3 -fopenmp -march=native knn_mnist.cpp -o knn
```

To run the compiled program and generate the performance CSVs:
```bash
./knn
```

To generate the performance and accuracy plots (requires Python, pandas, and matplotlib):
```bash
python plot_results.py
```

## Parameter Table
The following parameters are evaluated within the execution of the program:

| Parameter | Values Tested | Description |
| :--- | :--- | :--- |
| **k (Neighbors)** | `1, 5, 10` | Number of nearest neighbors used for majority voting. |
| **Distance Metric** | `Euclidean`, `Cosine` | Mathematical metrics used to calculate vector similarity. |
| **Threads** | `1, 2, 4, 8, 16` | OpenMP thread counts used to measure strong scaling. |
| **Schedule** | `static`, `dynamic`, `guided` | OpenMP loop scheduling strategies evaluated. |