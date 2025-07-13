# Brain-Tumor-Image-Classification-with-VGG19

## Overview
This project focuses on developing a robust deep learning model for the classification of brain tumor images. Utilizing transfer learning with a pre-trained VGG19 model, this solution aims to assist in the early and accurate detection of brain tumors from MRI scans, potentially aiding medical diagnosis.

## Features
* **Brain Tumor Classification:** Accurately classifies MRI brain images into different tumor categories (e.g., Meningioma, Glioma, Pituitary Tumor, No Tumor).
* **Transfer Learning with VGG19:** Leverages the powerful feature extraction capabilities of the pre-trained VGG19 Convolutional Neural Network.
* **Data Preprocessing & Augmentation:** Implements robust image preprocessing techniques and data augmentation to enhance model generalization and prevent overfitting.
* **Performance Evaluation:** Comprehensive evaluation using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
* **Model Visualization:** Provides insights into model training history (loss, accuracy curves) and predictions.

## Dataset
* **Source:** [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
* **Classes:** This dataset contains 7022 images of human brain MRI images which are classified into 4 classes:
    * Glioma
    * Meningioma
    * No tumor
    * Pituitary
* **Preprocessing:** Images are resized to 224x224 pixels
* **Split:** The dataset is split into training and test sets (90% training 10% test]).

## Model Architecture
* **Base Model:** VGG19, pre-trained on ImageNet.
* **Custom Layers:**
    * Flatten layer to convert the output of the convolutional base into a 1D feature vector.
    * Dense layers with ReLU activation for feature learning.
    * Dropout layers for regularization to prevent overfitting.
    * Output layer with Softmax activation for multi-class classification.
* **Optimizer:** Adam optimizer.
* **Loss Function:** Categorical Cross-Entropy.

## Performance
The model achieved the following performance metrics on the test set:
* **Accuracy:** 99.2%
* **F1-Score:** 99.2%
  
| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| glioma      | 1.00      | 0.98   | 0.99     | 163     |
| meningioma  | 0.97      | 1.00   | 0.98     | 164     |
| notumor     | 1.00      | 1.00   | 1.00     | 198     |
| pituitary   | 1.00      | 0.99   | 1.00     | 175     |
| **Accuracy**    |               |       | 0.99     | 700     |
| **Macro avg**   | 0.99          | 0.99  | 0.99     | 700     |
| **Weighted avg**| 0.99          | 0.99  | 0.99     | 700     |


### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/fahriasalsabilla/Brain-Tumor-Image-Classification-with-VGG19.git](https://github.com/fahriasalsabilla/Brain-Tumor-Image-Classification-with-VGG19.git)
    cd Brain-Tumor-Image-Classification-with-VGG19
    ```
