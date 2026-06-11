# Financial Risk Analytics for Credit Card Fraud Detection

## Project Overview

Developed an end-to-end Machine Learning-based Fraud Detection System to identify suspicious financial transactions and reduce financial risk. The solution leverages advanced classification algorithms, automated data pipelines, and real-time prediction capabilities to detect fraudulent activities within large-scale transaction datasets.

## Problem Statement

Financial institutions process millions of transactions daily, making it challenging to identify fraudulent activities using traditional rule-based systems. Undetected fraud can lead to significant financial losses, security breaches, and reduced customer trust. This project aims to build an intelligent fraud detection system capable of accurately identifying high-risk transactions while minimizing false alarms.

## Objectives

* Detect fraudulent transactions with high accuracy.
* Reduce financial losses caused by unauthorized activities.
* Automate fraud monitoring and risk assessment processes.
* Handle highly imbalanced financial transaction data effectively.
* Enable real-time fraud prediction and continuous performance monitoring.

## Solution Approach

The project follows a complete Machine Learning pipeline:

1. Data Extraction from SQLite Database
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering and Selection
5. Class Imbalance Analysis
6. Model Training using Random Forest and XGBoost
7. Hyperparameter Optimization using RandomizedSearchCV
8. Feature Importance Analysis
9. Model Deployment and Real-Time Prediction

## Technologies Used

* Python
* SQL (SQLite)
* Pandas
* NumPy
* Scikit-Learn
* Random Forest
* XGBoost
* RandomizedSearchCV
* Matplotlib
* Seaborn
* Joblib

## Model Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 100%  |
| Precision | 1.00  |
| Recall    | 0.97  |
| F1-Score  | 0.98  |

The final Random Forest model achieved excellent fraud detection performance while maintaining a near-zero false positive rate.

## Key Findings

The most influential factors contributing to fraud detection include:

* Transaction Amount
* Sender Account Balance
* Receiver Account Balance
* Transaction Type
* Balance Consistency Indicators

Feature importance analysis helped identify critical behavioral patterns associated with fraudulent transactions.

## Business Impact

* Reduced financial liability caused by fraudulent activities.
* Improved transaction monitoring efficiency.
* Enabled proactive fraud prevention and risk mitigation.
* Enhanced customer trust and account security.
* Supported scalable fraud analytics for high-volume transaction environments.

## Future Enhancements

* Real-time streaming fraud detection.
* Integration with banking transaction systems.
* Deep Learning-based anomaly detection models.
* Explainable AI dashboards for fraud investigation.
* Cloud deployment for large-scale production environments.

## Repository Structure

```text
Financial-Risk-Analytics-Credit-Card-Fraud-Detection/
│
├── model_training.py
├── predict_live.py
├── Fraud Detection Project Report.docx
├── best_random_forest_model.pkl
└── README.md
```

## Project Summary

Built a Machine Learning-based Fraud Detection System using Random Forest, XGBoost, and SQL to identify suspicious financial transactions. Implemented automated data preprocessing, feature engineering, anomaly detection, hyperparameter tuning, and real-time prediction workflows. Achieved 98% F1-score and 97% fraud recall, enabling proactive fraud prevention and reducing financial risk.
