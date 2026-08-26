import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================================================
# 1. MODEL PERFORMANCE COMPARISON
# ==================================================

results = pd.read_csv("results/model_results.csv")

print("Model results:")
print(results)

plt.figure(figsize=(10, 6))

plt.bar(results["Model"], results["R2 Score"])

plt.xlabel("ML Model")
plt.ylabel("R² Score")
plt.title("ML Model Performance Comparison")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("results/model_comparison.png", dpi=300)




# ==================================================
# 2. ACTUAL VS PREDICTED
# ==================================================

df = pd.read_csv(
    "data/INGRES_Averaged_Groundwater_Dataset.csv"
)

df = df.drop(
    columns=[
        "Major_Quality_Parameters",
        "Other_Quality_Parameters"
    ]
)

target_column = "Net_GW_Availability_Future_Use_ham"

X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Plot

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel("Actual Groundwater Availability (ham)")
plt.ylabel("Predicted Groundwater Availability (ham)")
plt.title("Actual vs Predicted Groundwater Availability")

# Perfect prediction line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=300
)



print("\nActual vs Predicted plot saved.")
print("R²:", round(r2_score(y_test, y_pred), 4))


# ==================================================
# 3. FEATURE IMPORTANCE
# ==================================================

rf_model = model.named_steps["model"]

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

top_features = feature_importance.head(15)

print("\nTop 15 Important Features:")
print(top_features.to_string(index=False))


plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features Affecting Groundwater Availability")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300
)

print("\nFeature importance plot saved.")