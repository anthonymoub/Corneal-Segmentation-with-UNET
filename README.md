# Corneal Demarcation Line Detection using U-Net

## Introduction
This repository contains the implementation and findings of our study on automated demarcation line detection in corneal scans using a deep learning approach. The project focuses on addressing keratoconus, a corneal condition, leveraging advanced Convolutional Neural Networks (CNNs) for post-surgical evaluation. Our model is based on the U-Net architecture, renowned for its effectiveness in biomedical image segmentation.
Our study is driven by the need for precise and efficient identification of the demarcation line in post-CXL (Corneal Cross-Linking) surgery scans for keratoconus treatment. Traditional methods are often subjective and time-consuming, hence, the application of deep learning offers a promising alternative.

## Data Collection
In collaboration with medical institutions in Switzelrand and Lebanon, we compiled a dataset of 939 corneal scans from 61 patients. These scans exhibit varying visibility of the demarcation line, adding complexity to the segmentation task.

## Methods

### U-Net model
Our approach employs U-Net, a specialized deep CNN for image segmentation. The architecture comprises a contracting path for context capture and an expansive path for precise localization. Key features include:

Skip connections for detailed segmentation.
A bottleneck layer for spatial dimension reduction.
Model checkpointing and learning rate scheduling for efficient training.
The dataset was divided into training (80%), validation (10%), and testing (10%) sets. The training process involves multiple epochs with rigorous validation to ensure model accuracy and generalizability.

### Preprocessing
The scans were pre-processed to optimize the model's performance, involving steps such as:

- Grayscale conversion.
- Standardization and normalization.
- Data augmentation techniques like rotations and flips to increase dataset robustness.


## Training
The model training was executed with the following specifics:

Loss function: Binary cross-entropy with Logits.
Optimizer: ADAM, SGD,
Regularization techniques: Dropout and data augmentation.
Hardware: Training was conducted on T4 GPU with 16 GB RAM on Google Colab
Results and Evaluation
The model was evaluated on a separate test set to ensure unbiased performance assessment. We used metrics like accuracy, precision, recall, and F1-score for comprehensive evaluation. The results indicated a significant improvement in identifying the demarcation line compared to traditional methods.

## Repository Structure
/data: Contains sample data for demonstration.
/models: U-Net model architecture and trained models.
/notebooks: Jupyter notebooks for training and evaluation.
/results: Results and visualizations of the model performance.
requirements.txt: List of Python dependencies for replication.
Usage
Instructions on how to set up the environment, train the model, and perform evaluations are provided in the /docs directory.


## Citations
U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger et al., 2015.
[Additional relevant papers and datasets used]
License
This project is licensed under [License Name]. For more details, see LICENSE file.

