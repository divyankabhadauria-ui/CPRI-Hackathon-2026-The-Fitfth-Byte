"""

  - Compares Logistic Regression, Random Forest, Extra Trees, Gradient Boosting
  - Evaluates each with Precision, Recall, F1-score, Confusion Matrix
  - Investigates WHY records are Invalid (not just "the model says so")
  - Prints the final Output block: Best Model / F1 Score / Important Features /
    Observed abnormal patterns
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_FILE = "data/training_data_for_person3_validity (1).csv"

TEST_FILE = "data/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"

OUTPUT_FOLDER = "outputs/person3"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("CPRI HACKATHON - PERSON 3")
print("VALID / INVALID DETECTION")
print("=" * 70)

print("\nLoading Person 3 training data...")

train = pd.read_csv(TRAIN_FILE)

print("Training data shape:", train.shape)

print("\nTraining columns:")
print(train.columns.tolist())


# ============================================================
# 3. FEATURES
# ============================================================

feature_cols = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]

TARGET = "Validity_Label"

X = train[feature_cols].copy()
y = train[TARGET].copy()


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before imputation:")
print(X.isnull().sum())

medians = X.median(numeric_only=True)

X = X.fillna(medians)

print("\nMissing values after imputation:")
print(X.isnull().sum())


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. DEFINE MODELS
# ============================================================

candidates = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

scores = {}
fitted_models = {}
comparison_results = []


for name, model in candidates.items():

    print(f"\nEvaluating {name}...")

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    f1 = f1_score(
        y_val,
        preds,
        pos_label="Invalid"
    )

    precision = precision_score(
        y_val,
        preds,
        pos_label="Invalid"
    )

    recall = recall_score(
        y_val,
        preds,
        pos_label="Invalid"
    )

    scores[name] = f1
    fitted_models[name] = model

    comparison_results.append({
        "Model": name,
        "Precision_Invalid": precision,
        "Recall_Invalid": recall,
        "F1_Invalid": f1
    })

    print(f"\n--- {name} ---")

    print(classification_report(y_val, preds))

    print(
        "Confusion matrix "
        "[rows=actual, cols=predicted], "
        "order=[Valid, Invalid]:"
    )

    print(
        confusion_matrix(
            y_val,
            preds,
            labels=["Valid", "Invalid"]
        )
    )


# ============================================================
# 8. SAVE MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(comparison_results)

comparison_df = comparison_df.sort_values(
    "F1_Invalid",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_name = max(
    scores,
    key=scores.get
)

best_model = fitted_models[best_name]

best_f1 = scores[best_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"\nBest Model: {best_name}")
print(f"Invalid F1 Score: {best_f1:.4f}")


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT FEATURES")
print("=" * 70)

if hasattr(best_model, "feature_importances_"):

    importance = pd.Series(
        best_model.feature_importances_,
        index=feature_cols
    )

    important_features = (
        importance
        .sort_values(ascending=False)
    )

elif hasattr(best_model, "coef_"):

    importance = pd.Series(
        best_model.coef_[0],
        index=feature_cols
    ).abs()

    important_features = (
        importance
        .sort_values(ascending=False)
    )

else:

    important_features = None


if important_features is not None:

    print(
        important_features.to_string()
    )

    importance_df = important_features.reset_index()

    importance_df.columns = [
        "Feature",
        "Importance"
    ]

    importance_df.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "feature_importance.csv"
        ),
        index=False
    )

else:

    print("Feature importance not available.")


# ============================================================
# 11. OBSERVED ABNORMAL PATTERNS
# ============================================================

print("\n" + "=" * 70)
print("OBSERVED ABNORMAL PATTERNS")
print("=" * 70)

comparison = (
    train.groupby(TARGET)[feature_cols]
    .mean()
    .T
)

comparison["difference"] = (
    comparison["Invalid"]
    - comparison["Valid"]
)

comparison["pct_difference"] = (
    comparison["difference"]
    / comparison["Valid"]
    * 100
).round(1)

comparison = comparison.sort_values(
    "pct_difference",
    key=abs,
    ascending=False
)

print(comparison.to_string())

comparison.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "abnormal_patterns.csv"
    )
)


# ============================================================
# 12. LOAD REAL TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("PREDICTING REAL TEST DATA")
print("=" * 70)

raw_train = pd.read_excel(
    TEST_FILE,
    sheet_name="Training_Data"
)

test = pd.read_excel(
    TEST_FILE,
    sheet_name="Test_Data"
)


# ============================================================
# 13. IMPUTE TEST DATA
# ============================================================

sensor_cols = [
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4"
]

test[sensor_cols] = test[sensor_cols].fillna(
    raw_train[sensor_cols].median()
)


# ============================================================
# 14. TRAIN BEST MODEL ON ALL TRAINING DATA
# ============================================================

print("\nTraining best model on all training data...")

final_model = type(best_model)(
    **best_model.get_params()
)

final_model.fit(X, y)

print("Training complete.")


# ============================================================
# 15. PREDICT TEST DATA
# ============================================================

test_predictions = final_model.predict(
    test[feature_cols]
)


# ============================================================
# 16. CREATE RESULTS
# ============================================================

results = pd.DataFrame({
    "Test_ID": test["Test_ID"],
    "Predicted_Validity_Label": test_predictions
})


print("\nPredicted counts on test set:")

print(
    results["Predicted_Validity_Label"].value_counts()
)

invalid_percentage = (
    results["Predicted_Validity_Label"]
    .eq("Invalid")
    .mean()
    * 100
)

print(
    f"\n% Invalid: {invalid_percentage:.2f}%"
)


# ============================================================
# 17. DISPLAY FIRST 15
# ============================================================

print("\nFirst 15 predictions:")

print(
    results
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 18. SAVE PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    OUTPUT_FOLDER,
    "validity_predictions_final.csv"
)

results.to_csv(
    prediction_file,
    index=False
)

print(
    f"\nSaved predictions -> {prediction_file}"
)


# ============================================================
# 19. SAVE SUMMARY
# ============================================================

summary = {
    "task": "Validity Detection",
    "training_records": int(len(train)),
    "test_records": int(len(test)),
    "best_model": best_name,
    "invalid_f1": float(best_f1),
    "test_valid": int(
        (
            results["Predicted_Validity_Label"]
            == "Valid"
        ).sum()
    ),
    "test_invalid": int(
        (
            results["Predicted_Validity_Label"]
            == "Invalid"
        ).sum()
    ),
    "test_invalid_percentage": float(
        invalid_percentage
    )
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "person3_summary.csv"
    ),
    index=False
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PERSON 3 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
