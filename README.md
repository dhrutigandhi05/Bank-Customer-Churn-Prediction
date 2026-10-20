# Bank Customer Churn Prediction
An ML project that predicts whether a bank customer is likely to leave based on factors like age, account balance, credit score, and banking activity. It includes an interactive terminal app where users can enter customer information and get a churn prediction.

## Tech Stack
Language: Python
Data Analysis: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn
Terminal UI: Textual
Model Serialization: Joblib
Testing: Pytest

## Models & Results
Trained and compared Logistic Regression and Decision Tree classifiers using 5-fold cross-validation.

The Decision Tree  with max depth 10 got the best F1 score.

| Metric | Test Score |
|---|---|
| Accuracy | 83.95% |
| Precision | 64.73% |
| Recall | 46.44% |
| F1 Score | 54.08% |

This project applies concepts including descriptive statistics, correlation, logistic regression, decision trees, entropy, and information gain.