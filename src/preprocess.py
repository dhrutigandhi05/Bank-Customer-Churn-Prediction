import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_features_target(df):
    X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    y = df["Exited"]
    return X, y

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def create_preprocessor(X):
    categorical_features = ["Geography", "Gender"]

    numerical_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor