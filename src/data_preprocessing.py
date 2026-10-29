from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def split_data(df, test_size=0.2, random_state=42):
    """Razdvaja karakteristike i cenu, pa deli podatke na trening i test skup."""

    X = df.drop(columns=["priceUSD"])
    y = df["priceUSD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def get_column_types(X):
    """Razdvaja numeričke i kategorijske karakteristike."""

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return numerical_columns, categorical_columns


def create_numerical_pipeline():
    """Pravi pipeline za obradu numeričkih karakteristika."""

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return numerical_pipeline


def create_categorical_pipeline():
    """Pravi pipeline za obradu kategorijskih karakteristika."""

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return categorical_pipeline

def create_preprocessor(numerical_columns, categorical_columns):
    """Povezuje numerički i kategorijski pipeline."""

    numerical_pipeline = create_numerical_pipeline()
    categorical_pipeline = create_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns)
        ]
    )

    return preprocessor