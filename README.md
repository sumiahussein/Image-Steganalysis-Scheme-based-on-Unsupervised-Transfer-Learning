# README

## Introduction
This MATLAB code is designed for implementing a Convolutional Neural Network (CNN) for image classification and transfer learning. It includes training and testing phases along with evaluation metrics.

## Requirements
- MATLAB environment
- X.mat and LB.mat files containing image data and corresponding labels respectively.

## Usage
1. Ensure that X.mat and LB.mat files are present in the same directory as the script.
2. Run the main_Unsupervised.m script in MATLAB.

## Description
1. **Data Preprocessing**: The code loads image data and labels, normalizes the images, and reshapes them for model input.
2. **Model Definition**: Two CNN models are defined: one for regression and one for classification. Each model consists of convolutional layers, activation functions, pooling layers, fully connected layers, and output layers.
3. **Training**: The regression model is trained using the provided data. Transfer learning is performed on the output of the regression model to train the classification model.
4. **Testing and Evaluation**: The trained models are tested using separate test data. Evaluation metrics such as accuracy, precision, and Dice similarity coefficient (DSC) are computed.

## Files
- `X.mat`: Contains image data.
- `LB.mat`: Contains corresponding labels.

## Results
The script outputs the following evaluation metrics:
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Accuracy
- Precision
- Dice Similarity Coefficient (DSC)

## Contributors
- Sumia Al-Obaidi/University of Tabriz (https://github.com/sumiahussein)

## License
This project is licensed under the MIT License.


A main dataset can be downloaded from http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip

X.mat can be accessed via Google Drive https://drive.google.com/drive/folders/1V6it3IH_JAcz0Vvr5vlQeph_rDKYFxvM?usp=sharing 
