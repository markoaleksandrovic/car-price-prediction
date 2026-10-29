
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_preprocessing import create_preprocessor
import joblib
from pathlib import Path


def train_comparison_models(
    X_train,
    y_train,
    numerical_columns,
    categorical_columns
):
    """Trenira tri regresiona modela radi poređenja."""

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    trained_models = {}

    for name, regressor in models.items():

        preprocessor = create_preprocessor(
            numerical_columns,
            categorical_columns
        )

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} — uspešno istreniran!")

    return trained_models


def evaluate_comparison_models(models, X_test, y_test):
    """Evaluira modele i poredi njihove regresione metrike."""

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

    results_df = pd.DataFrame(results)

    return results_df.sort_values(by="MAE")


def save_best_model(models, results_df):
    """Čuva model sa najmanjom MAE vrednošću."""

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    model_path = Path("models/car_price_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)

    print(f"Najbolji model: {best_model_name}")
    print(f"Model sačuvan u: {model_path}")

    return best_model