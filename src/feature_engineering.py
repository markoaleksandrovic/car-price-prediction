def add_car_age(df, reference_year=2026):
    """Dodaje starost automobila na osnovu godine proizvodnje."""

    df = df.copy()

    df["car_age"] = reference_year - df["year"]

    return df


def add_mileage_per_year(df):
    """Dodaje prosečnu godišnju kilometražu automobila."""

    df = df.copy()

    df["mileage_per_year"] = (
        df["mileage(kilometers)"] / df["car_age"]
    )

    return df


def engineer_features(df):
    """Kreira nove karakteristike za predviđanje cene automobila."""

    df = add_car_age(df)
    df = add_mileage_per_year(df)

    return df