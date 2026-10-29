from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


def create_linear_regression_model(preprocessor):
    """Pravi pipeline za pretprocesiranje i linearnu regresiju."""

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    return model

def train_model(model, X_train, y_train):
    """Trenira regresioni model i čuva ga na disku."""

    model.fit(X_train, y_train)

    model_path = Path("models/linear_regression_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    return model