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
* **Source:** [Sebutkan Sumber Dataset Anda, Contoh: Kaggle, atau nama folder jika Anda membuatnya sendiri].
* **Classes:** [Sebutkan Jumlah dan Nama Kelas, Contoh: 4 classes: Meningioma, Glioma, Pituitary Tumor, and No Tumor].
* **Image Count:** [Sebutkan Jumlah Total Gambar, Contoh: Approximately X images (e.g., 7000 images)].
* **Distribution:** [Sebutkan Distribusi Gambar per Kelas, Contoh: Fairly balanced distribution across classes].
* **Preprocessing:** Images are resized to [Sebutkan Ukuran Resizing, Contoh: 224x224 pixels] and normalized.
* **Split:** The dataset is split into training, validation, and test sets ([Sebutkan Persentase Split, Contoh: e.g., 80% training, 10% validation, 10% test]).

## Model Architecture
* **Base Model:** VGG19, pre-trained on ImageNet.
* **Custom Layers:**
    * Flatten layer to convert the output of the convolutional base into a 1D feature vector.
    * Dense layers with ReLU activation for feature learning.
    * Dropout layers for regularization to prevent overfitting.
    * Output layer with [Sebutkan Aktivasi Output, Contoh: Softmax activation] for multi-class classification.
* **Optimizer:** [Sebutkan Optimizer yang Digunakan, Contoh: Adam optimizer].
* **Loss Function:** [Sebutkan Loss Function, Contoh: Categorical Cross-Entropy].

## Performance
The model achieved the following performance metrics on the test set:
* **Accuracy:** [Sebutkan Akurasi, Contoh: 98.5%]
* **Precision:** [Sebutkan Presisi (bisa rata-rata atau per kelas), Contoh: 0.98]
* **Recall:** [Sebutkan Recall (bisa rata-rata atau per kelas), Contoh: 0.97]
* **F1-Score:** [Sebutkan F1-Score (bisa rata-rata atau per kelas), Contoh: 0.97]

[Opsional: Sertakan Confusion Matrix atau Classification Report dalam format teks atau link ke gambar jika Anda menyediakannya.]

## Getting Started

### Prerequisites
* Python 3.x
* Jupyter Notebook (recommended)

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/fahriasalsabilla/Brain-Tumor-Image-Classification-with-VGG19.git](https://github.com/fahriasalsabilla/Brain-Tumor-Image-Classification-with-VGG19.git)
    cd Brain-Tumor-Image-Classification-with-VGG19
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    (Pastikan Anda memiliki file `requirements.txt` yang berisi daftar semua library Python yang digunakan, contoh: `tensorflow`, `keras`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `opencv-python`)

### Usage
1.  Download the dataset from [Sebutkan Sumber Dataset Anda dan Link Download jika ada].
2.  Place the dataset in the appropriate directory (e.g., `data/`).
3.  Open and run the Jupyter Notebook: `brain_tumor_classification.ipynb` (Sesuaikan nama file `.ipynb` Anda).
4.  Follow the steps in the notebook to preprocess data, build, train, and evaluate the model.

## File Structure
