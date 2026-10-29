import pandas as pd


def load_data(file_path):
    """Učitava skup podataka iz CSV fajla."""
    return pd.read_csv(file_path)

def remove_duplicates(df):
    """Uklanja potpuno duplirane redove iz skupa podataka."""
    df_cleaned = df.drop_duplicates().copy()
    return df_cleaned

def check_invalid_values(df):
    """Proverava broj potencijalno nevalidnih numeričkih vrednosti."""
    invalid_values = {
        "priceUSD <= 0": (df["priceUSD"] <= 0).sum(),
        "mileage(kilometers) < 0": (df["mileage(kilometers)"] < 0).sum(),
        "volume(cm3) <= 0": (df["volume(cm3)"] <= 0).sum(),
        "year > 2026": (df["year"] > 2026).sum()
    }

    return invalid_values


def clean_data(df):
    """Sprovodi osnovno čišćenje podataka."""

    df_cleaned = remove_duplicates(df)

    return df_cleaned