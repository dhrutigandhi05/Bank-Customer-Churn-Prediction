from pathlib import Path
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.preprocess import (
    load_data,
    split_features_target,
    split_data
)

def evaluate_model():
    project_root = Path(__file__).resolve().parent.parent
    model = joblib.load(project_root / "models" / "churn_model.joblib")
    df = load_data(project_root / "data" / "Churn_Modelling.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    y_pred = model.predict(X_test)

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test Precision:", precision_score(y_test, y_pred))
    print("Test Recall:", recall_score(y_test, y_pred))
    print("Test F1:", f1_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Churned"]
    ))

if __name__ == "__main__":
    evaluate_model()