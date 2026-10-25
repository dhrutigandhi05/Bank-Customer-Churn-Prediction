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
Trained and compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting classifiers using 5-fold cross-validation.
Gradient Boosting got the highest cross-validation F1 score (0.5873).

| Metric | Test Score |
|---|---|
| Accuracy | 86.70% |
| Precision | 76.80% |
| Recall | 49.63% |
| F1 Score | 60.30% |