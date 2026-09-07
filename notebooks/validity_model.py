"""
Produces exactly what the brief asks for:
  - Compares Logistic Regression, Random Forest, Extra Trees, Gradient Boosting
  - Evaluates each with Precision, Recall, F1-score, Confusion Matrix
  - Investigates WHY records are Invalid (not just "the model says so")
  - Prints the final Output block: Best Model / F1 Score / Important Features /
    Observed abnormal patterns
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report,
)

DATA_PATH = "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"

train = pd.read_csv("cleaned_training_with_missing_indicators_imputed.csv")

feature_cols = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
    "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]

X = train[feature_cols]
y = train["Validity_Label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

print("=== Comparing all 4 models (evaluated on held-out validation data) ===\n")
scores = {}
fitted_models = {}

for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    f1 = f1_score(y_val, preds, pos_label="Invalid")
    precision = precision_score(y_val, preds, pos_label="Invalid")
    recall = recall_score(y_val, preds, pos_label="Invalid")

    scores[name] = f1
    fitted_models[name] = model

    print(f"--- {name} ---")
    print(classification_report(y_val, preds))
    print("Confusion matrix [rows=actual, cols=predicted], order=[Valid, Invalid]:")
    print(confusion_matrix(y_val, preds, labels=["Valid", "Invalid"]))
    print()

best_name = max(scores, key=scores.get)
best_model = fitted_models[best_name]
best_f1 = scores[best_name]

print(f"=== BEST MODEL: {best_name} (F1 = {best_f1:.3f}) ===\n")

if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=feature_cols)
    important_features = importance.sort_values(ascending=False)
elif hasattr(best_model, "coef_"):
    importance = pd.Series(best_model.coef_[0], index=feature_cols).abs()
    important_features = importance.sort_values(ascending=False)
else:
    important_features = None

print("Important Features:")
print(important_features.to_string() if important_features is not None else "N/A for this model type")


print("\n=== Observed abnormal patterns (Valid vs Invalid comparison) ===")
comparison = train.groupby("Validity_Label")[feature_cols].mean().T
comparison["difference"] = comparison["Invalid"] - comparison["Valid"]
comparison["pct_difference"] = (comparison["difference"] / comparison["Valid"] * 100).round(1)
print(comparison.sort_values("pct_difference", key=abs, ascending=False).to_string())



print("=== Predicting on the real TEST set using the best model ===\n")

sensor_cols = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
raw_train = pd.read_excel(DATA_PATH, sheet_name="Training_Data")
test = pd.read_excel(DATA_PATH, sheet_name="Test_Data")
medians = raw_train[sensor_cols].median()
test[sensor_cols] = test[sensor_cols].fillna(medians)


best_model_class = type(best_model)
final_model = best_model_class(**best_model.get_params())
final_model.fit(X, y)

test_predictions = final_model.predict(test[feature_cols])

results = pd.DataFrame({
    "Test_ID": test["Test_ID"],
    "Predicted_Validity_Label": test_predictions,
})

print("Predicted counts on the test set:")
print(results["Predicted_Validity_Label"].value_counts())
print(f"\n% Invalid: {(results['Predicted_Validity_Label']=='Invalid').mean()*100:.2f}%\n")

print("Classification table (first 15 rows):")
print(results.head(15).to_string(index=False))

results.to_csv("validity_predictions_final.csv", index=False)
print(f"\nSaved full table -> validity_predictions_final.csv ({len(results)} rows)")
