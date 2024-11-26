# Preprocessing and Analysis Codes for Task fMRI

This repository contains preprocessing and analysis codes for task-based functional MRI (tfMRI) data. The structure of the repository is organized into several subdirectories, each dedicated to a specific aspect of data processing and analysis. Below is a description of each directory and its purpose.

## Repository Structure

### 1. `preprocessing`
This folder contains all the scripts needed for preprocessing tfMRI data. The preprocessing includes steps such as data organization into the BIDS format, motion correction, and alignment. It provides the foundation for subsequent analyses, ensuring data quality and consistency across subjects.

### 2. `encoding`
The `encoding` folder contains codes for constructing encoding models, which map stimulus features to brain activity. These models are typically used to predict brain activity in response to specific stimuli, allowing you to understand how the brain represents external information during the task.

### 3. `decoding`
The `decoding` folder provides codes for decoding models that infer stimulus characteristics or behavioral responses based on observed brain activity. Decoding is useful for understanding how much information about a specific variable is present in the brain's activity patterns.

### 4. `prf`
The `prf` folder contains scripts for population receptive field (pRF) analysis. pRF analysis is typically used to determine the visual field properties of neurons based on tfMRI data, which can be particularly useful in mapping retinotopic areas in visual cortex regions.

### 5. `quality_validation`
The `quality_validation` folder includes scripts for validating data quality. This is an essential step to ensure that the preprocessing procedures yield reliable data. The scripts perform quality checks such as motion inspection, signal-to-noise ratio (SNR) assessment, and identification of outliers.

### 6. `rsa`
The `rsa` (Representational Similarity Analysis) folder contains scripts to perform representational similarity analysis. RSA is used to compare the similarity structure of neural patterns with behavioral or theoretical models, providing insights into how information is represented in different regions of the brain.
