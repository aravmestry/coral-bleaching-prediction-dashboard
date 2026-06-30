import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df = pd.read_csv("../../datasets/global_bleaching_environmental.csv", low_memory=False)

predictors = [
    "Latitude_Degrees", "Longitude_Degrees", "Depth_m",
    "SSTA", "SSTA_Mean", "SSTA_Maximum",
    "SSTA_DHW", "SSTA_DHWMean",
    "TSA", "TSA_Mean",
    "Turbidity", "Cyclone_Frequency"
]

target = "Percent_Bleaching"

data = df[predictors + [target]].copy()

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

X = data[predictors]
y = data[target]

print("Clean modeling rows:", len(data))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []

best_model = None
best_score = -999

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    results.append({
        "Model": name,
        "R2": round(r2, 3),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3)
    })

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_pred = pred

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
results_df.to_csv("outputs/model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df)

cv_scores = cross_val_score(
    best_model, X, y, cv=5, scoring="r2", n_jobs=-1
)

pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "R2": cv_scores
}).to_csv("outputs/cross_validation.csv", index=False)

print("\n5-Fold CV R²:")
print(cv_scores)
print("Mean CV R²:", round(cv_scores.mean(), 3))

joblib.dump(best_model, "outputs/best_random_forest_model.pkl")

importance = pd.Series(
    best_model.feature_importances_,
    index=predictors
).sort_values(ascending=False)

importance.to_csv("outputs/feature_importance.csv", header=["Importance"])

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.25)
plt.plot([0, 100], [0, 100], linestyle="--")
plt.xlabel("Observed Percent Bleaching")
plt.ylabel("Predicted Percent Bleaching")
plt.title("Observed vs Predicted Coral Bleaching")
plt.tight_layout()
plt.savefig("figures/01_observed_vs_predicted.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
importance.sort_values().plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("Environmental Drivers of Coral Bleaching")
plt.tight_layout()
plt.savefig("figures/02_feature_importance.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.hist(y, bins=40)
plt.xlabel("Percent Bleaching")
plt.ylabel("Number of Observations")
plt.title("Distribution of Coral Bleaching Severity")
plt.tight_layout()
plt.savefig("figures/03_bleaching_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylabel("R² Score")
plt.title("Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/04_model_comparison.png", dpi=300)
plt.close()

print("\nDone. Outputs saved to outputs/ and figures/")