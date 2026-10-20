from src.preprocess import (
    load_data,
    split_features_target,
    split_data,
    create_preprocessor
)

def test_dataset_loading():
    df = load_data("data/Churn_Modelling.csv")
    assert df.shape == (10000, 14)
    assert "Exited" in df.columns
    assert df.isnull().sum().sum() == 0

def test_feature_selection():
    df = load_data("data/Churn_Modelling.csv")
    X, y = split_features_target(df)
    assert X.shape[1] == 10
    assert "Exited" not in X.columns
    assert len(X) == len(y)

def test_data_split():
    df = load_data("data/Churn_Modelling.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert len(y_train) == 8000
    assert len(y_test) == 2000
    assert set(X_train.index).isdisjoint(set(X_test.index))

def test_preprocessor():
    df = load_data("data/Churn_Modelling.csv")
    X, y = split_features_target(df)
    preprocessor = create_preprocessor(X)
    assert len(preprocessor.transformers) == 2