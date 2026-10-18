from src.preprocess import (
    load_data,
    split_features_target,
    split_data,
    create_preprocessor
)

df = load_data("data/Churn_Modelling.csv")
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)
preprocessor = create_preprocessor(X_train)
print("Dataset:", df.shape)
print("Training:", X_train.shape)
print("Testing:", X_test.shape)
print("Preprocessor:", preprocessor)