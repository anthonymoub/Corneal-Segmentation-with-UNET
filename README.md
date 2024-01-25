# Corneal Demarcation Line Detection using U-Net

<div style="text-align:center">
    <img width="848" alt="Screenshot" src="https://github.com/anthonymoub/Corneal-Segmentation-with-UNET/assets/112438562/b2b964f8-dadb-45e1-8a8a-d5a73f8c8bc0">
</div>

## Introduction
This project centers around the application of Convolutional Neural Networks (CNN) for the identification of a specific type of line in corneal scans, which is crucial in post-surgical analysis for a particular eye disease. The chosen model architecture is U-Net. The primary objective of our model is to streamline the detection process of this line, with the potential for integration into medical devices employed for scanning patients' corneas, thereby introducing automation to this diagnostic procedure.

## Disclaimer 
As this project is currently in progress and we plan on producing a publication from it, certain aspects such as data collection, the actual dataset, data cleaning, and splitting have been kept external to this repository, with details intentionally left vague. The repository serves as an illustrative snapshot of the work accomplished thus far. We anticipate updating this README with a link to the research paper once it undergoes publication.

## Data
In collaboration with medical institutions in Switzerland and Lebanon, we compiled a dataset of 939 corneal scans from 61 patients. These scans exhibit varying visibility of the demarcation line, adding complexity to the segmentation task. The basic task the model needs to learn is illustrated below, in the input (X) and output (Y) image. Input images consist of corneal scans, which contain a thin line in cases where the surgery is successful. The ground-truth Y or output image is simply a mask of the line, which is traced manually by opthalmologists.

<img width="759" alt="Screenshot 2024-01-23 at 4 55 09 PM" src="https://github.com/anthonymoub/Corneal-Segmentation-with-UNET/assets/112438562/0bf6dab2-c1c5-42ad-928d-dcfde987fcc3">


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
data.py: Creates a train and test loader using the original data. <br>
model.py: U-Net model architecture. <br>
train.py: Script used to train the model. <br>
resume_training.py: Since model training was interrupted often due to large number of images, a resume training script deals with this. <br>
requirements.txt: List of Python dependencies for replication. <br>
 
## Results

Preliminary results show the model is able to segment demarcation lines correctly on unseen data.
![image](https://github.com/anthonymoub/Corneal-Segmentation-with-UNET/assets/103491240/d040f714-b767-412d-bd39-e0c0d009a2a4)


## Citations
U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger et al., 2015.

