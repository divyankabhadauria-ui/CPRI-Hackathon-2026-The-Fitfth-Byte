import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. FILE PATHS
# ============================================================

TRAIN_FILE = "data/training_data_for_person3_validity (1).csv"

TEST_FILE = "data/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"

OUTPUT_FOLDER = "outputs/person3"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. BASE FEATURES
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

TARGET = "Validity_Label"


# ============================================================
# 3. FEATURE ENGINEERING
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
# 4. LOAD TRAINING DATA
# ============================================================

print("\nLoading Person 3 training data...")

train = pd.read_csv(TRAIN_FILE)

print("Training shape:", train.shape)


# Check required columns

required_columns = BASE_FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in train.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. CREATE MISSING-VALUE INDICATORS
# ============================================================

sensor_columns = [
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4"
]

for sensor in sensor_columns:

    train[sensor + "_missing"] = (
        train[sensor].isna().astype(int)
    )


# ============================================================
# 6. IMPUTE BASE FEATURES
# ============================================================

print("\nHandling missing values...")

imputer = SimpleImputer(strategy="median")

train[BASE_FEATURES] = imputer.fit_transform(
    train[BASE_FEATURES]
)


# ============================================================
# 7. CREATE ENGINEERED FEATURES
# ============================================================

train = add_features(train)


# ============================================================
# 8. CLASSIFICATION FEATURES
# ============================================================

FEATURES = BASE_FEATURES + [
    "V_times_I",
    "S1_minus_S2",
    "S1_minus_S3",
    "S2_minus_S3",
    "Sensor_S1_missing",
    "Sensor_S2_missing",
    "Sensor_S3_missing",
    "Sensor_S4_missing"
]


X = train[FEATURES].copy()

y = train[TARGET].copy()


# ============================================================
# 9. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# 10. MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=700,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)

print("MODEL COMPARISON")

print("=" * 60)


results = []

best_model_name = None

best_f1 = -1

best_model = None


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_val
    )

    f1 = f1_score(
        y_val,
        predictions,
        pos_label="Invalid"
    )

    results.append({

        "Model": name,

        "F1_Invalid": f1

    })

    print(
        f"{name}: F1 = {f1:.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1

        best_model_name = name

        best_model = model


print("\nBest Model:", best_model_name)

print(
    "Best F1 Score:",
    round(best_f1, 4)
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

validation_predictions = best_model.predict(
    X_val
)

print("\n" + "=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(
    classification_report(
        y_val,
        validation_predictions
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        validation_predictions,
        labels=["Valid", "Invalid"]
    )
)


# ============================================================
# 14. SAVE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)

print("FEATURE IMPORTANCE")

print("=" * 60)


if hasattr(
    best_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({

        "Feature": FEATURES,

        "Importance":
            best_model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "feature_importance.csv"
        ),
        index=False
    )


# ============================================================
# 16. TRAIN BEST MODEL ON ALL TRAINING DATA
# ============================================================

print(
    "\nTraining best model on complete training data..."
)

best_model.fit(
    X,
    y
)


# ============================================================
# 17. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

test = pd.read_excel(
    TEST_FILE,
    sheet_name="Test_Data"
)

print(
    "Test shape:",
    test.shape
)


# ============================================================
# 18. CREATE TEST MISSING INDICATORS
# ============================================================

for sensor in sensor_columns:

    test[sensor + "_missing"] = (
        test[sensor].isna().astype(int)
    )


# ============================================================
# 19. IMPUTE TEST DATA
# ============================================================

test[BASE_FEATURES] = imputer.transform(
    test[BASE_FEATURES]
)


# ============================================================
# 20. CREATE TEST ENGINEERED FEATURES
# ============================================================

test = add_features(test)


X_test = test[FEATURES].copy()


# ============================================================
# 21. PREDICT VALID / INVALID
# ============================================================

test_predictions = best_model.predict(
    X_test
)


# ============================================================
# 22. SAVE PREDICTIONS
# ============================================================

prediction_output = pd.DataFrame({

    "Test_ID":
        test["Test_ID"],

    "Predicted_Validity_Label":
        test_predictions

})


prediction_file = os.path.join(

    OUTPUT_FOLDER,

    "validity_predictions_final.csv"

)


prediction_output.to_csv(

    prediction_file,

    index=False

)


# ============================================================
# 23. FINAL COUNTS
# ============================================================

print("\n" + "=" * 60)

print("FINAL TEST PREDICTIONS")

print("=" * 60)


print(
    prediction_output[
        "Predicted_Validity_Label"
    ].value_counts()
)


print(
    "\nSaved:",
    prediction_file
)


print(
    "\nPerson 3 validity detection completed successfully! ✅"
)
