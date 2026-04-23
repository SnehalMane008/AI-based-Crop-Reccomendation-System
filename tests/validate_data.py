# tests/validate_data.py
import pandas as pd
import great_expectations as ge

def validate_crop_data(csv_path: str = "Crop_recommendation.csv"):
    df = ge.read_csv(csv_path)
    
    results = {}
    
    # Schema checks
    results["columns"] = df.expect_table_columns_to_match_ordered_list(
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    )
    
    # No nulls
    for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]:
        results[f"no_null_{col}"] = df.expect_column_values_to_not_be_null(col)
    
    # Value ranges
    results["N_range"] = df.expect_column_values_to_be_between("N", 0, 140)
    results["P_range"] = df.expect_column_values_to_be_between("P", 5, 145)
    results["K_range"] = df.expect_column_values_to_be_between("K", 5, 205)
    results["temp_range"] = df.expect_column_values_to_be_between("temperature", 8, 44)
    results["humidity_range"] = df.expect_column_values_to_be_between("humidity", 14, 100)
    results["ph_range"] = df.expect_column_values_to_be_between("ph", 3, 10)
    results["rainfall_range"] = df.expect_column_values_to_be_between("rainfall", 20, 300)
    
    # Class balance check - each crop should have >= 80 samples
    results["class_counts"] = df.expect_column_value_lengths_to_be_between("label", 1, 20)
    
    # Check valid crop labels
    valid_crops = [
        "rice", "maize", "chickpea", "banana", "mango", "grapes",
        "watermelon", "apple", "orange", "cotton", "jute", "coffee"
    ]
    results["valid_labels"] = df.expect_column_values_to_be_in_set("label", valid_crops)
    
    # Report failures
    failed = [k for k, v in results.items() if not v["success"]]
    if failed:
        print(f"FAILED validations: {failed}")
        raise ValueError(f"Data validation failed: {failed}")
    
    print("✅ All data validations passed!")
    return True

if __name__ == "__main__":
    validate_crop_data()