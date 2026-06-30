# Handwritten Character Recognition using CNN

## Overview
This Deep Learning project implements a Convolutional Neural Network (CNN) to recognize handwritten characters. It uses the standard MNIST dataset (digits 0-9) to demonstrate end-to-end deep learning workflows, including data preprocessing, model building, training with callbacks, and comprehensive evaluation.

## Dataset
We are using the **MNIST Dataset**, directly loaded from `tensorflow.keras.datasets`.
- **Training Samples**: 60,000 grayscale images (28x28 pixels)
- **Testing Samples**: 10,000 grayscale images (28x28 pixels)
- **Classes**: 10 (Digits 0 through 9)

## Project Structure
- `analysis.ipynb`: The Jupyter Notebook containing the full pipeline (CNN architecture explanation, data loading, preprocessing, model training, evaluation, and future improvements).
- `charts/`: Contains generated visualizations (Sample images, Training/Validation Loss & Accuracy curves, Confusion Matrix, and Sample Predictions).
- `trained_model.keras`: The saved weights and architecture of the trained CNN model.
- `outputs/`: Contains the predictions CSV file.
- `summary.docx`: A professional Word document summarizing the CNN architecture and key findings.
- `requirements.txt`: Python dependencies.

## CNN Architecture
The CNN model features:
- Convolutional layers with ReLU activation to extract spatial features.
- MaxPooling layers to downsample feature maps.
- **Batch Normalization** to stabilize and accelerate training.
- **Dropout** layers to prevent overfitting.
- Fully connected Dense layers for classification.
- **EarlyStopping** callback to halt training when validation loss stops improving.

## Setup and Execution
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Jupyter Notebook `analysis.ipynb` to view the analysis, training, and evaluation pipeline.

## Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Future Improvements
- Adapting the architecture to the broader EMNIST dataset (including letters) for full alphanumeric recognition.
- Applying Data Augmentation to further improve model robustness.
- Tuning hyperparameters for optimal performance.
