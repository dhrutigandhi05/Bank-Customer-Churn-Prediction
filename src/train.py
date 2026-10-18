from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.preprocess import (
    load_data,
    split_features_target,
    split_data,
    create_preprocessor
)

def create_models(X_train):
    models = {
        "logistic_regression": Pipeline([
            ("preprocessor", create_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),

        "decision_tree": Pipeline([
            ("preprocessor", create_preprocessor(X_train)),
            ("classifier", DecisionTreeClassifier(
                max_depth=10,
                criterion="entropy",
                random_state=42
            ))
        ])
    }

    return models

def train_models():
    project_root = Path(__file__).resolve().parent.parent
    df = load_data(project_root / "data" / "Churn_Modelling.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = create_models(X_train)

    crossValidation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    best_model = None
    best_score = -1
    best_name = None

    for name, model in models.items():
        results = cross_validate(
            model,
            X_train,
            y_train,
            cv=crossValidation,
            scoring="f1"
        )

        mean_f1 = results["test_score"].mean()
        print(f"{name}: Cross-validation F1 = {mean_f1:.4f}")

        if mean_f1 > best_score:
            best_score = mean_f1
            best_model = model
            best_name = name

    best_model.fit(X_train, y_train)
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "churn_model.joblib")

    print(f"\nBest model: {best_name}")
    print(f"Cross-validation F1: {best_score:.4f}")
    print("Model saved successfully.")

if __name__ == "__main__":
    train_models()