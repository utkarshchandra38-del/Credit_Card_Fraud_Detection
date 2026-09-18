# Credit Card Fraud Detection using Machine Learning

## Overview

This project is a Machine Learning case study focused on detecting fraudulent credit card transactions.

Credit card fraud detection is a highly imbalanced classification problem, where fraudulent transactions represent a very small proportion of the total transactions. The objective of this project is to build a machine learning model capable of identifying fraudulent transactions while evaluating its performance using appropriate classification metrics.

## Dataset

The project uses a Credit Card Fraud Detection dataset containing transaction information and a binary target variable:

- `Class = 0` → Legitimate transaction
- `Class = 1` → Fraudulent transaction

The dataset is not included in this repository because the dataset file is approximately 144 MB, which exceeds GitHub's regular file size limit.

**Dataset Source:** 
MLG-ULB Credit Card Dataset from Kaggle
[original dataset link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Jupyter Notebook / Google Colab

## Machine Learning Workflow

The notebook follows these steps:

1. Load and inspect the dataset
2. Analyze the class distribution
3. Separate features and target variable
4. Perform a stratified train-test split
5. Handle class imbalance using **SMOTE**
6. Train an **XGBoost Classifier**
7. Generate predictions and prediction probabilities
8. Evaluate the model using:
   - Confusion Matrix
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
9. Perform classification threshold tuning
10. Analyze feature importance

## Handling Class Imbalance

Since fraudulent transactions are significantly fewer than legitimate transactions, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training data.

SMOTE is applied only after splitting the dataset into training and testing sets to prevent information from the test set from being used during training.

## Model

The project uses **XGBoost Classifier** for binary classification.

The model predicts whether a transaction is:

- Legitimate
- Fraudulent

## Evaluation

Because this is an imbalanced classification problem, accuracy alone is not sufficient to evaluate the model.

The following metrics are considered:

- **Precision** – proportion of transactions predicted as fraud that are actually fraudulent.
- **Recall** – proportion of actual fraudulent transactions detected by the model.
- **F1-Score** – harmonic mean of precision and recall.
- **ROC-AUC** – measures the model's ability to distinguish between legitimate and fraudulent transactions.
- **Confusion Matrix** – provides the counts of true positives, true negatives, false positives, and false negatives.

The notebook also evaluates different probability thresholds and selects a threshold based on the F1-score.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── MLE Case Study 2.ipynb
├── README.md
└── .gitignore
