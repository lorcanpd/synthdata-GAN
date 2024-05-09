# GAN for Synthetic Data Creation

## Overview

This project was developed for the Wellcome Sanger Institute Machine Learning Hackathon 2024. This is a draft framework for creating a synthetic data generator.
Data from the Kaggle competition ["Open Problems - Multimodal Single-Cell Integration"](https://www.kaggle.com/competitions/open-problems-multimodal/data) was adopted for the development of the framework as it is a reasonably sized, clean, dataset.
The Python script here trains a Generative Adversarial Network (GAN) architecture to generate synthetic single cell expression data corresponding to various cell types.

## Background

The data originates from mobilized peripheral CD34+ hematopoietic stem and progenitor cells (HSPCs) isolated from four healthy human donors. These cells were observed over a ten-day period, during which they were cultured under controlled conditions. The dataset includes two types of single-cell assays:

- **Multiome**: Measures chromatin accessibility and gene expression.
- **CITEseq**: Measures gene expression and surface protein levels.

The GAN training script focuses on CITEseq gene expression data. The GAN is trained to generate synthetic gene expression profiles based on cell type labels. 
However, as this a draft framework the models do not learn to model the cell types on different time points or donors. Instead, they treat all cell types as one.

## Model Description

The GAN consists of three main components:
1. **ConditionalGenerator**: Generates synthetic gene expression data based on input noise modulated by cell type-specific parameters. Uses embeddings for mean and deviation, which are modulated through input noise to produce data reflective of specific cell types.
2. **ClassDiscriminator**: Classifies the authenticity of generated data based on cell type, enhancing the generator's ability to produce type-specific expressions.
3. **AdversarialDiscriminator**: Validates the authenticity of generated samples, refining the training process by providing feedback on the realism of synthetic data.

## Key Features
- **Sparsity Enforcement**: Employs mechanisms to ensure the sparsity of output data, mimicking the real distribution of gene expressions where many genes are not expressed.

## Features to be added
- **Label Mixing**: Introduces interpolations between class decision boundaries to create a smoother gradient of cell type characteristics.
- **GPU Support**: Designed for acceleration using CUDA, allowing training on modern GPUs for enhanced performance.

### Input Data Files

- **`metadata.csv`**: Contains metadata about each cell, including cell ID, donor, day of experiment, technology used, and cell type.
- **HDF5 Datasets**: Large arrays containing the experimental measurements for Multiome and CITEseq technologies.
  - Specifically this work employs the CITEseq data. 

### Requirements
`python==3.11.9`
`pandas==2.2.2`
`torch==2.3.0`