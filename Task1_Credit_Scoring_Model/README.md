# Credit Scoring Model using Machine Learning

## Overview
This project presents an end-to-end Machine Learning solution for predicting credit scoring. It evaluates whether a customer is creditworthy (good credit) or a default risk (bad credit). The project is structured and formatted for a Machine Learning Internship submission, demonstrating industry-standard practices.

## Dataset
We are using the **German Credit Data** (fetched via `scikit-learn.datasets.fetch_openml`), a standard dataset in credit scoring. 
- **Target Variable**: `class` (good vs. bad credit risk)
- **Features**: Various demographic, financial, and credit history attributes of the applicants.

## Project Structure
- `analysis.ipynb`: A comprehensive Jupyter Notebook containing Exploratory Data Analysis (EDA), Data Preprocessing, Feature Engineering, Model Training, Evaluation, and Business Recommendations.
- `charts/`: Contains generated visualizations (Target distribution, Correlation heatmap, Feature importance, ROC curves, Confusion matrices).
- `models/`: Stores the best trained model as a serialized `.pkl` file.
- `outputs/`: Contains the predictions and model comparison metrics in CSV format.
- `summary.docx`: A professional Word document summarizing the methodology, findings, and conclusions.
- `requirements.txt`: List of required Python dependencies.

## Models Evaluated
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

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
- ROC-AUC

## Conclusion
The results of the analysis provide actionable business recommendations for financial institutions to minimize default risk while maximizing the approval of creditworthy applicants.
