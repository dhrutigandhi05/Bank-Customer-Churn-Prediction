from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

def get_feature_importance():
    model = joblib.load(MODEL_PATH)
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    results = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    return results.sort_values(
        by="Importance",
        ascending=False
    )

if __name__ == "__main__":
    importance = get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))