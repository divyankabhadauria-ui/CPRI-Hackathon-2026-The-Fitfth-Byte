import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression


# 1. SETTINGS
INPUT_FILE = "data/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
OUTPUT_FOLDER = "outputs/person2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# 2. LOAD DATA
print("=" * 70)
print("CPRI HACKATHON - PERSON 2")
print("REFERENCE PARAMETER PREDICTION")
print("=" * 70)

print("\nLoading Excel file...")

excel_file = pd.ExcelFile(INPUT_FILE)

print("\nAvailable sheets:")
print(excel_file.sheet_names)


train = pd.read_excel(
    INPUT_FILE,
    sheet_name="Training_Data"
)

test = pd.read_excel(
    INPUT_FILE,
    sheet_name="Test_Data"
)

print("\nTraining data shape:", train.shape)
print("Test data shape:", test.shape)


# 3. DISPLAY COLUMNS
print("\nTraining columns:")
print(train.columns.tolist())

print("\nTest columns:")
print(test.columns.tolist())


# 4. TARGET AND FEATURES
TARGET = "Reference_Parameter"

DROP_COLUMNS = [
    "Test_ID",
    "Validity_Label",
    "Reference_Parameter"
]

if TARGET not in train.columns:
    raise ValueError(
        "Reference_Parameter column was not found in Training_Data."
    )


X = train.drop(columns=DROP_COLUMNS)
y = train[TARGET]

X_test = test[X.columns]


print("\nTarget:")
print(TARGET)

print("\nFeatures used for prediction:")
for feature in X.columns:
    print(" -", feature)


# 5. HANDLE MISSING VALUES
print("\nMissing values in training features:")

missing_values = X.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:

    print("\nMissing values detected. Applying median imputation.")

    medians = X.median(numeric_only=True)

    X = X.fillna(medians)
    X_test = X_test.fillna(medians)

else:
    print("No missing feature values detected.")


# 6. DEFINE MODELS
models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.10,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=2,
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    )
}


# 7. 5-FOLD CROSS VALIDATION
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():

    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error"
        },
        n_jobs=-1
    )

    mae = -scores["test_MAE"]
    rmse = -scores["test_RMSE"]

    results.append({
        "Model": name,
        "CV_MAE": mae.mean(),
        "MAE_STD": mae.std(),
        "CV_RMSE": rmse.mean(),
        "RMSE_STD": rmse.std()
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "CV_MAE"
).reset_index(drop=True)


print("\n")
print(results_df.to_string(index=False))


# Save comparison
results_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "model_comparison.csv"
    ),
    index=False
)


# 8. SELECT BEST MODEL
BEST_MODEL_NAME = results_df.iloc[0]["Model"]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("\nBest model:", BEST_MODEL_NAME)

best_model = models[BEST_MODEL_NAME]


# 9. TRAIN BEST MODEL
print("\nTraining best model on all training data...")

best_model.fit(X, y)

print("Training complete.")


# 10. FEATURE IMPORTANCE
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

if hasattr(best_model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    importance_df["Importance_Percent"] = (
        importance_df["Importance"] * 100
    )

    print(importance_df.to_string(index=False))

    importance_df.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "feature_importance.csv"
        ),
        index=False
    )

else:

    print(
        "Feature importance is not directly available "
        "for this model."
    )


# 11. PREDICT TEST DATA
print("\n" + "=" * 70)
print("PREDICTING TEST DATA")
print("=" * 70)

test_predictions = best_model.predict(X_test)


# 12. CREATE PREDICTION FILE
prediction_df = pd.DataFrame({
    "Test_ID": test["Test_ID"],
    "Predicted_Reference_Parameter": test_predictions
})


prediction_file = os.path.join(
    OUTPUT_FOLDER,
    "Person2_Reference_Parameter_Predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\nPrediction file created:")
print(prediction_file)


# 13. PREDICTION STATISTICS
minimum_prediction = test_predictions.min()
maximum_prediction = test_predictions.max()
average_prediction = test_predictions.mean()


print("\n" + "=" * 70)
print("PREDICTION STATISTICS")
print("=" * 70)

print(f"\nNumber of test records: {len(test_predictions)}")

print(
    f"Minimum predicted Reference Parameter: "
    f"{minimum_prediction:.4f}"
)

print(
    f"Maximum predicted Reference Parameter: "
    f"{maximum_prediction:.4f}"
)

print(
    f"Average predicted Reference Parameter: "
    f"{average_prediction:.4f}"
)


# 14. FIRST 20 PREDICTIONS
print("\n" + "=" * 70)
print("FIRST 20 TEST PREDICTIONS")
print("=" * 70)

print(
    prediction_df.head(20).to_string(index=False)
)


# 15. SAVE SUMMARY
summary = {
    "task": "Reference Parameter Prediction",
    "training_records": int(len(train)),
    "test_records": int(len(test)),
    "target": TARGET,
    "best_model": BEST_MODEL_NAME,
    "cv_mae": float(results_df.iloc[0]["CV_MAE"]),
    "cv_rmse": float(results_df.iloc[0]["CV_RMSE"]),
    "minimum_prediction": float(minimum_prediction),
    "maximum_prediction": float(maximum_prediction),
    "average_prediction": float(average_prediction)
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "person2_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("PERSON 2 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
