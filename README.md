# Customer Churn Prediction for a Gaming Platform

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Machine Learning Workflow](#machine-learning-workflow)
  - [1. Data Cleaning and Churn Definition](#1-data-cleaning-and-churn-definition)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Modeling Pipeline](#3-modeling-pipeline)
  - [4. Model Training and Comparison](#4-model-training-and-comparison)
- [Results and Analysis](#results-and-analysis)
- [How to Run This Project](#how-to-run-this-project)
- [Conclusion and Future Work](#conclusion-and-future-work)

## Introduction

This project focuses on building a machine learning model to predict customer churn for **AB Gaming**, a digital gaming platform with a monthly subscription service. By analyzing user activity and sales data, the model aims to identify customers who are likely to cancel their subscriptions.

The primary business goal is to proactively identify at-risk customers so that retention strategies can be implemented, ultimately reducing revenue loss and improving customer lifetime value.

## Datasets

The analysis is based on two primary data sources:

- `sales.csv`: Contains the transaction history for clients acquired since 2019-01-01.
- `user_activity.csv`: Contains user characteristics and engagement metrics.

Key features include:

- **User Info:** `gender`, `age`, `account_id`
- **Subscription Info:** `plan` (SMALL, MEDIUM, LARGE), `currency` (USD, EUR)
- **Engagement Metrics:** `hours` (mean weekly hours played), `games` (median weekly games played), `genre1` (most played genre), `genre2` (second most played genre)
- **Platform Info:** `type` (device type used)

## Project Structure

```
.
├── 20231215_CaseStudy_ML_Neural_Network.ipynb  # Main notebook for EDA, feature engineering, and initial model building.
├── src/
│   ├── data_loader.py                          # Module for loading and preprocessing the data.
│   └── models.py                               # Module for training and evaluating all models.
├── README.md                                   # Project documentation.
```

## Machine Learning Workflow

The project follows a structured, end-to-end machine learning pipeline.

### 1. Data Cleaning and Churn Definition

- **Churn Definition:** A customer is defined as a **churner** if they have made **fewer than 7 monthly payments**.
- **Data Filtering:** To ensure model quality, two groups of users were excluded from the training data:
  1.  Users with only 1 or 2 payments, as their short history provides insufficient data for reliable prediction.
  2.  Users whose subscription period was still active near the data extraction date, as their final churn status is ambiguous.

### 2. Feature Engineering

To enhance the model's predictive power, new features were created from the existing data:

- `hours_per_game`: Calculated as `hours / games`, this feature represents the average time a user dedicates to a single game, potentially indicating their level of deep engagement.
- `age_group`: The numerical `age` feature was binned into categorical groups (e.g., "18-21", "22-25"). This helps the model capture non-linear relationships between age and churn that a raw numerical feature might miss.

### 3. Modeling Pipeline

A robust data preparation pipeline was built to prepare the data for various machine learning algorithms.

- **One-Hot Encoding:** Categorical features like `gender`, `plan`, and `genre` were converted into a numerical format using one-hot encoding.
- **Handling Class Imbalance:** The dataset is imbalanced, with significantly more non-churners than churners. To prevent the model from becoming biased, **Random Oversampling** was applied to the training data. This technique balances the dataset by creating synthetic copies of the minority class (churners).
- **Feature Scaling:** For models sensitive to the scale of input features (like Neural Networks and SVM), `StandardScaler` was used to standardize the data, giving each feature a mean of 0 and a standard deviation of 1.

### 4. Model Training and Comparison

A key part of this project was to systematically train and compare a suite of different models to find the best performer.

- **Models Trained:**
  - **Scaled Data Models:** Logistic Regression, Support Vector Machine (SVM), and a Neural Network.
  - **Unscaled Data Models:** Random Forest and XGBoost.
- **Hyperparameter Tuning:** For the Random Forest model, `RandomizedSearchCV` was used to automatically find the optimal hyperparameters, ensuring the model's performance was maximized.
- **Neural Network Architecture:** A Keras-based sequential Neural Network was built with `Dropout` layers and an `EarlyStopping` callback to prevent overfitting and ensure robust generalization.

## Results and Analysis

The performance of all models was collected and can be compared to identify the most effective one for this specific problem. The final evaluation was based on a comprehensive set of metrics, including:

- **Accuracy:** Overall percentage of correct predictions.
- **Precision:** The proportion of predicted churners who actually churned.
- **Recall (Sensitivity):** The proportion of actual churners that the model correctly identified.
- **F1-Score:** The harmonic mean of precision and recall, providing a single metric that balances both concerns.
- **Confusion Matrix:** A table visualizing the performance, showing true positives, true negatives, false positives, and false negatives.

| Model               | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| ------------------- | -------- | ----------------- | -------------- | ---------------- |
| Logistic Regression | 0.7430   | 0.4632            | 0.9167         | 0.6154           |
| SVM                 | 0.7944   | 0.5313            | 0.7083         | 0.6071           |
| Neural Network      | 0.7897   | 0.5224            | 0.7292         | 0.6087           |
| Random Forest       | 0.8458   | 0.6829            | 0.5833         | 0.6292           |
| XGBoost             | 0.8224   | 0.5735            | 0.8125         | 0.6724           |

## Conclusion and Future Work

This project successfully demonstrates an end-to-end machine learning workflow for predicting customer churn. By systematically cleaning data, engineering relevant features, and comparing multiple models, a robust predictive solution can be developed.
