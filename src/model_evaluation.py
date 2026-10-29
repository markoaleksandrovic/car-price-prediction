import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model, X_test, y_test):
    """Evaluira regresioni model pomoću standardnih metrika."""

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return metrics


def create_prediction_examples(model, X_test, y_test, n=10):
    """Prikazuje primere stvarnih i predviđenih cena automobila."""

    y_pred = model.predict(X_test)

    examples = pd.DataFrame({
        "actual_price": y_test.values,
        "predicted_price": y_pred
    })

    examples["error"] = abs(
        examples["actual_price"] - examples["predicted_price"]
    )

    return examples.head(n).round(2)