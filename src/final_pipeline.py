from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, f1_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# BASE FEATURES
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


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):

    df = df.copy()

    # Electrical relationship
    df["V_times_I"] = (
        df["Applied_Voltage_kV"]
        * df["Load_Current_A"]
    )

    # Sensor difference features
    df["S1_minus_S2"] = (
        df["Sensor_S1"]
        - df["Sensor_S2"]
    )

    df["S1_minus_S3"] = (
        df["Sensor_S1"]
        - df["Sensor_S3"]
    )

    df["S2_minus_S3"] = (
        df["Sensor_S2"]
        - df["Sensor_S3"]
    )

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(train, test):

    train = train.copy()
    test = test.copy()

    # --------------------------------------------------------
    # 1. Create missing-value indicators FIRST
    # --------------------------------------------------------

    for sensor in [
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Sensor_S4"
    ]:

        train[sensor + "_missing"] = (
            train[sensor].isna().astype(int)
        )

        test[sensor + "_missing"] = (
            test[sensor].isna().astype(int)
        )

    # --------------------------------------------------------
    # 2. Fill missing values using TRAINING medians
    # --------------------------------------------------------

    for column in BASE_FEATURES:

        median_value = train[column].median()

        train[column] = train[column].fillna(
            median_value
        )

        test[column] = test[column].fillna(
            median_value
        )

    # --------------------------------------------------------
    # 3. Create engineered features AFTER imputation
    # --------------------------------------------------------

    train = add_features(train)
    test = add_features(test)

    return train, test


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("Loading data...")

    # --------------------------------------------------------
    # LOAD ORIGINAL DATASET
    # --------------------------------------------------------

    train = pd.read_excel(
        DATA_FILE,
        sheet_name="Training_Data"
    )

    test = pd.read_excel(
        DATA_FILE,
        sheet_name="Test_Data"
    )

    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    # --------------------------------------------------------
    # CLEAN + FEATURE ENGINEERING
    # --------------------------------------------------------

    train, test = clean_data(train, test)

    # ========================================================
    # REGRESSION
    # ========================================================

    REGRESSION_FEATURES = BASE_FEATURES + [
        "V_times_I"
    ]

    X_reg = train[REGRESSION_FEATURES]
    y_reg = train["Reference_Parameter"]

    X_test_reg = test[REGRESSION_FEATURES]

    regression_models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.10,
                max_depth=2,
                min_samples_split=20,
                min_samples_leaf=2,
                random_state=42
            ),

        "Hist Gradient Boosting":
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42
            )
    }

    print("\n========== REGRESSION ==========")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    regression_results = []

    for name, model in regression_models.items():

        fold_mae = []

        for train_idx, val_idx in kfold.split(X_reg):

            X_train = X_reg.iloc[train_idx]
            X_val = X_reg.iloc[val_idx]

            y_train = y_reg.iloc[train_idx]
            y_val = y_reg.iloc[val_idx]

            model.fit(X_train, y_train)

            prediction = model.predict(X_val)

            mae = mean_absolute_error(
                y_val,
                prediction
            )

            fold_mae.append(mae)

        average_mae = np.mean(fold_mae)

        regression_results.append({
            "Model": name,
            "CV_MAE": average_mae
        })

        print(
            f"{name}: CV MAE = "
            f"{average_mae:.6f}"
        )

    regression_results_df = pd.DataFrame(
        regression_results
    )

    regression_results_df = regression_results_df.sort_values(
        "CV_MAE"
    )

    best_regression_name = (
        regression_results_df.iloc[0]["Model"]
    )

    best_regression_mae = float(
        regression_results_df.iloc[0]["CV_MAE"]
    )

    print(
        f"\nBest regression model: "
        f"{best_regression_name}"
    )

    # --------------------------------------------------------
    # TRAIN BEST REGRESSION MODEL ON ALL DATA
    # --------------------------------------------------------

    best_regression_model = regression_models[
        best_regression_name
    ]

    best_regression_model.fit(
        X_reg,
        y_reg
    )

    reference_predictions = (
        best_regression_model.predict(
            X_test_reg
        )
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

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

    X_cls = train[CLASSIFICATION_FEATURES]

    y_cls = (
        train["Validity_Label"]
        .map({
            "Valid": 0,
            "Invalid": 1
        })
    )

    X_test_cls = test[CLASSIFICATION_FEATURES]

    # --------------------------------------------------------
    # RANDOM FOREST CLASSIFIER
    # --------------------------------------------------------

    classifier = RandomForestClassifier(
        n_estimators=700,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\n========== CLASSIFICATION ==========")

    skfold = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    f1_scores = []

    for train_idx, val_idx in skfold.split(
        X_cls,
        y_cls
    ):

        X_train = X_cls.iloc[train_idx]
        X_val = X_cls.iloc[val_idx]

        y_train = y_cls.iloc[train_idx]
        y_val = y_cls.iloc[val_idx]

        classifier.fit(
            X_train,
            y_train
        )

        prediction = classifier.predict(
            X_val
        )

        score = f1_score(
            y_val,
            prediction
        )

        f1_scores.append(score)

    classification_f1 = float(
        np.mean(f1_scores)
    )

    print(
        f"Random Forest CV F1 = "
        f"{classification_f1:.6f}"
    )

    # --------------------------------------------------------
    # TRAIN CLASSIFIER ON ALL TRAINING DATA
    # --------------------------------------------------------

    classifier.fit(
        X_cls,
        y_cls
    )

    validity_numeric = classifier.predict(
        X_test_cls
    )

    invalid_probability = (
        classifier.predict_proba(
            X_test_cls
        )[:, 1]
    )

    validity_predictions = np.where(
        validity_numeric == 1,
        "Invalid",
        "Valid"
    )

    # ========================================================
    # ATTENTION SCORE
    # ========================================================

    sensor_difference = (
        test["S1_minus_S2"].abs()
        + test["S1_minus_S3"].abs()
        + test["S2_minus_S3"].abs()
    ) / 3

    if sensor_difference.max() != sensor_difference.min():

        normalized_sensor_difference = (
            sensor_difference
            - sensor_difference.min()
        ) / (
            sensor_difference.max()
            - sensor_difference.min()
        )

    else:

        normalized_sensor_difference = (
            sensor_difference * 0
        )

    missing_proportion = (
        test[
            [
                "Sensor_S1_missing",
                "Sensor_S2_missing",
                "Sensor_S3_missing",
                "Sensor_S4_missing"
            ]
        ].mean(axis=1)
    )

    attention_score = (
        0.70 * invalid_probability
        + 0.20 * normalized_sensor_difference
        + 0.10 * missing_proportion
    )

    attention_df = pd.DataFrame({

        "Test_ID": test["Test_ID"],

        "attention_score":
            attention_score,

        "invalid_probability":
            invalid_probability

    })

    top_3 = (
        attention_df
        .sort_values(
            "attention_score",
            ascending=False
        )
        .head(3)
    )

    # ========================================================
    # FINAL SUBMISSION
    # ========================================================

    submission = pd.DataFrame({

        "Test_ID":
            test["Test_ID"],

        "Predicted_Reference_Parameter":
            reference_predictions,

        "Validity_Label":
            validity_predictions
    })

    output_csv = (
        OUTPUT_DIR
        / "TheFifthByte.csv"
    )

    submission.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "number_of_records":
            int(len(submission)),

        "number_valid":
            int(
                (submission["Validity_Label"] == "Valid")
                .sum()
            ),

        "number_invalid":
            int(
                (submission["Validity_Label"] == "Invalid")
                .sum()
            ),

        "minimum_predicted_reference_parameter":
            float(
                submission[
                    "Predicted_Reference_Parameter"
                ].min()
            ),

        "maximum_predicted_reference_parameter":
            float(
                submission[
                    "Predicted_Reference_Parameter"
                ].max()
            ),

        "average_predicted_reference_parameter":
            float(
                submission[
                    "Predicted_Reference_Parameter"
                ].mean()
            ),

        "top_3_test_ids_needing_attention":
            top_3["Test_ID"].tolist(),

        "regression_model":
            best_regression_name,

        "regression_cv_mae":
            best_regression_mae,

        "classification_model":
            "Random Forest",

        "classification_cv_f1":
            classification_f1
    }

    output_summary = (
        OUTPUT_DIR
        / "summary.json"
    )

    with open(
        output_summary,
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    print("\n========== FINAL CHECK ==========")

    print(
        f"Submission rows: "
        f"{len(submission)}"
    )

    print(
        f"Unique Test IDs: "
        f"{submission['Test_ID'].nunique()}"
    )

    print(
        f"Missing reference predictions: "
        f"{submission['Predicted_Reference_Parameter'].isna().sum()}"
    )

    print(
        f"Missing validity predictions: "
        f"{submission['Validity_Label'].isna().sum()}"
    )

    print("\nValidity counts:")

    print(
        submission["Validity_Label"].value_counts()
    )

    print("\nTop 3 IDs needing attention:")

    print(top_3)

    print("\nFiles created:")

    print(output_csv)
    print(output_summary)

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert len(submission) == len(test)

    assert (
        submission["Test_ID"].nunique()
        == len(test)
    )

    assert (
        submission[
            "Predicted_Reference_Parameter"
        ].notna().all()
    )

    assert (
        submission[
            "Validity_Label"
        ].notna().all()
    )

    assert set(
        submission["Validity_Label"].unique()
    ).issubset({
        "Valid",
        "Invalid"
    })

    print(
        "\nFINAL PIPELINE COMPLETED SUCCESSFULLY!"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
