# tests/validate_data.py
"""
Data validation using pandas (replaces great_expectations which is
incompatible with Python 3.14).
"""
import pandas as pd


def validate_crop_data(csv_path: str = "Crop_recommendation.csv"):
    df = pd.read_csv(csv_path)

    results = {}
    failures = []

    # --- Schema check ---
    expected_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    results["columns"] = list(df.columns) == expected_cols
    if not results["columns"]:
        failures.append(f"columns: expected {expected_cols}, got {list(df.columns)}")

    # --- No nulls ---
    for col in expected_cols:
        has_nulls = df[col].isnull().any()
        results[f"no_null_{col}"] = not has_nulls
        if has_nulls:
            failures.append(f"no_null_{col}: found {int(df[col].isnull().sum())} nulls")

    # --- Value ranges ---
    range_checks = {
        "N":           (0, 140),
        "P":           (5, 145),
        "K":           (5, 205),
        "temperature": (8, 44),
        "humidity":    (14, 100),
        "ph":          (3, 10),
        "rainfall":    (20, 300),
    }
    for col, (lo, hi) in range_checks.items():
        out_of_range = ~df[col].between(lo, hi)
        results[f"{col}_range"] = not out_of_range.any()
        if out_of_range.any():
            failures.append(f"{col}_range: {int(out_of_range.sum())} values outside [{lo}, {hi}]")

    # --- Valid crop labels ---
    valid_crops = [
        "rice", "maize", "chickpea", "banana", "mango", "grapes",
        "watermelon", "apple", "orange", "cotton", "jute", "coffee"
    ]
    invalid = ~df["label"].isin(valid_crops)
    results["valid_labels"] = not invalid.any()
    if invalid.any():
        failures.append(f"valid_labels: {int(invalid.sum())} invalid labels found")

    # --- Class balance: each crop should have >= 80 samples ---
    counts = df["label"].value_counts()
    under_represented = counts[counts < 80]
    results["class_balance"] = len(under_represented) == 0
    if len(under_represented) > 0:
        failures.append(f"class_balance: under-represented crops: {dict(under_represented)}")

    # --- Report ---
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
        raise ValueError(f"Data validation failed: {failures}")

    print("✅ All data validations passed!")
    return True


if __name__ == "__main__":
    validate_crop_data()