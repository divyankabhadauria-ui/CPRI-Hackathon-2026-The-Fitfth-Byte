import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

BASE_FEATURES = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4"
]

REGRESSION_FEATURES = BASE_FEATURES + [
    "V_times_I"
]

CLASSIFICATION_FEATURES = BASE_FEATURES + [
    "V_times_I",
    "S1_minus_S2",
    "S1_minus_S3",
    "S2_minus_S3",
    "Sensor_S1_missing",
    "Sensor_S2_missing",
    "Sensor_S3_missing",
    "Sensor_S4_missing"
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):
    df = df.copy()

    # Missing-value indicators
    for sensor in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]:
        df[sensor + "_missing"] = df[sensor].isna().astype(int)

    # Physical / sensor relationship features
    df["V_times_I"] = (
        df["Applied_Voltage_kV"] *
        df["Load_Current_A"]
    )

    df["S1_minus_S2"] = (
        df["Sensor_S1"] -
        df["Sensor_S2"]
    )

    df["S1_minus_S3"] = (
        df["Sensor_S1"] -
        df["Sensor_S3"]
    )

    df["S2_minus_S3"] = (
        df["Sensor_S2"] -
        df["Sensor_S3"]
    )

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(train, test):

    train = train.copy()
    test = test.copy()

    # Create features before filling missing values
    train = add_features(train)
    test = add_features(test)

    # Fill numeric missing values using TRAINING medians only
    for column in BASE_FEATURES:

        median_value = train[column].median()

        train[column] = train[column].fillna(median_value)
        test[column] = test[column].fillna(median_value)

    return train, test


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("Loading data...")

    train = pd.read_excel(
        DATA_FILE,
        sheet_name="Training_Data"
    )

    test = pd.read_excel(
        DATA_FILE,
        sheet_name="Test_Data"
    )

    print("Training rows:", len(train))
    print("Test rows:", len(test))

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    train, test = clean_data(train, test)

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    y_reg = train["Reference_Parameter"]

    y_class = train["Validity_Label"]

    # --------------------------------------------------------
    # REGRESSION MODEL
    # Gradient Boosting
    # --------------------------------------------------------

    X_reg = train[REGRESSION_FEATURES]
    X_test_reg = test[REGRESSION_FEATURES]

    regression_model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.10,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=2,
        random_state=42
    )

    regression_model.fit(
        X_reg,
        y_reg
    )

    predicted_reference = regression_model.predict(
        X_test_reg
    )

    # --------------------------------------------------------
    # CLASSIFICATION MODEL
    # Random Forest
    # --------------------------------------------------------

    X_class = train[CLASSIFICATION_FEATURES]
    X_test_class = test[CLASSIFICATION_FEATURES]

    classification_model = RandomForestClassifier(
        n_estimators=700,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    classification_model.fit(
        X_class,
        y_class
    )

    predicted_validity = classification_model.predict(
        X_test_class
    )

    # Probability of Invalid
    probabilities = classification_model.predict_proba(
        X_test_class
    )

    class_names = list(classification_model.classes_)

    if "Invalid" in class_names:
        invalid_index = class_names.index("Invalid")
        invalid_probability = probabilities[:, invalid_index]
    else:
        invalid_probability = np.zeros(len(test))

    # --------------------------------------------------------
    # ATTENTION SCORE
    # --------------------------------------------------------

    sensor_differences = (
        test["S1_minus_S2"].abs() +
        test["S1_minus_S3"].abs() +
        test["S2_minus_S3"].abs()
    ) / 3

    max_difference = sensor_differences.max()

    if max_difference > 0:
        normalized_difference = (
            sensor_differences /
            max_difference
        )
    else:
        normalized_difference = np.zeros(len(test))

    missing_proportion = (
        test[
            [
                "Sensor_S1_missing",
                "Sensor_S2_missing",
                "Sensor_S3_missing",
                "Sensor_S4_missing"
            ]
        ].sum(axis=1) / 4
    )

    attention_score = (
        0.70 * invalid_probability +
        0.20 * normalized_difference +
        0.10 * missing_proportion
    )

    # --------------------------------------------------------
    # FINAL SUBMISSION
    # --------------------------------------------------------

    submission = pd.DataFrame({
        "Test_ID": test["Test_ID"],
        "Predicted_Reference_Parameter": predicted_reference,
        "Validity_Label": predicted_validity
    })

    submission_file = OUTPUT_DIR / "TheFifthByte.csv"

    submission.to_csv(
        submission_file,
        index=False
    )

    # --------------------------------------------------------
    # TOP 3 TEST IDs
    # --------------------------------------------------------

    attention_table = pd.DataFrame({
        "Test_ID": test["Test_ID"],
        "attention_score": attention_score,
        "invalid_probability": invalid_probability
    })

    top_3 = (
        attention_table
        .sort_values(
            "attention_score",
            ascending=False
        )
        .head(3)
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "number_of_records": int(len(test)),

        "number_valid": int(
            (predicted_validity == "Valid").sum()
        ),

        "number_invalid": int(
            (predicted_validity == "Invalid").sum()
        ),

        "minimum_predicted_reference_parameter": float(
            predicted_reference.min()
        ),

        "maximum_predicted_reference_parameter": float(
            predicted_reference.max()
        ),

        "average_predicted_reference_parameter": float(
            predicted_reference.mean()
        ),

        "top_3_test_ids_needing_attention": (
            top_3["Test_ID"].tolist()
        ),

        "regression_model": "Gradient Boosting",

        "classification_model": "Random Forest"
    }

    summary_file = OUTPUT_DIR / "summary.json"

    with open(summary_file, "w") as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL CHECKS
    # --------------------------------------------------------

    print("\n========== FINAL CHECK ==========")

    print("Submission rows:", len(submission))
    print("Unique Test IDs:", submission["Test_ID"].nunique())
    print(
        "Missing reference predictions:",
        submission["Predicted_Reference_Parameter"].isna().sum()
    )
    print(
        "Missing validity predictions:",
        submission["Validity_Label"].isna().sum()
    )

    print("\nValidity counts:")
    print(submission["Validity_Label"].value_counts())

    print("\nTop 3 IDs needing attention:")
    print(top_3)

    print("\nFiles created:")
    print(submission_file)
    print(summary_file)

    print("\nFINAL PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
