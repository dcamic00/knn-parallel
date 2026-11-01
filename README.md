# Parallel k-NN Classifier (OpenMP + SIMD)

## Description
This repository contains a parallel implementation of the exact k-Nearest Neighbors (k-NN) classification algorithm using OpenMP and SIMD vectorization in C++. This project is part of the Mid-term assignments for the *Parallel Programming for Machine Learning* course.

To maximize performance, facilitate SIMD vectorization, and optimize cache locality, the data structure has been transformed into a Structure of Arrays (SoA) memory layout.

## Dataset
The classifier is evaluated on the MNIST dataset of handwritten digits (784 dimensions).
- **Source**: [OpenML MNIST](https://www.openml.org/d/554)
- **Required files**: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`.

## Requirements
- GCC Compiler (`g++`) with OpenMP support.
- Operating System: Windows (via Anaconda/MinGW) or Linux.

## Compilation
To compile the source code applying performance optimizations (`-O3`), multithreading support (`-fopenmp`), and architecture-specific vectorization (`-march=native`), run:

```bash
g++ -O3 -fopenmp -march=native knn_mnist.cpp -o knn