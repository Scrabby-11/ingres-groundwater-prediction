import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 1. LOAD DATASET
# ==================================================

file_path = "/Users/shashwatnigam/Desktop/INGRES PROJ/data/INGRES_Averaged_Groundwater_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==================================================
# 2. REMOVE USELESS COLUMNS
# ==================================================

columns_to_remove = [
    "Major_Quality_Parameters",
    "Other_Quality_Parameters"
]

df = df.drop(columns=columns_to_remove)

print("\nAfter removing useless columns:")
print("Dataset shape:", df.shape)


# ==================================================
# 3. DEFINE TARGET
# ==================================================

target_column = "Net_GW_Availability_Future_Use_ham"

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nTarget variable:", target_column)
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ==================================================
# 4. IDENTIFY COLUMN TYPES
# ==================================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ==================================================
# 5. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain-Test Split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ==================================================
# 6. PREPROCESSING PIPELINE
# ==================================================

# Numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)


# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ==================================================
# 7. DEFINE ML MODELS
# ==================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# ==================================================
# 8. TRAIN AND EVALUATE MODELS
# ==================================================

results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("MAE :", round(mae, 4))
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))


# ==================================================
# 9. MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))


# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_df.to_csv(
    "results/model_results.csv",
    index=False
)

print("\nModel results saved successfully!")
print("Location: results/model_results.csv")

# ==================================================
# 11. FEATURE-TO-TARGET CORRELATION CHECK
# ==================================================

print("\nFEATURE-TO-TARGET CORRELATION")
print("=" * 60)

numeric_features = df.select_dtypes(include=["number"])

correlations = (
    numeric_features
    .corr()[target_column]
    .drop(target_column)
    .sort_values(ascending=False)
)

print(correlations)


# ==================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

# Create the final Random Forest pipeline
best_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", best_model)
    ]
)

# Train the model
best_pipeline.fit(X_train, y_train)

# Get transformed feature names
feature_names = (
    best_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

# Get feature importance
importance = (
    best_pipeline
    .named_steps["model"]
    .feature_importances_
)

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 20 FEATURE IMPORTANCES")
print("=" * 60)

print(
    feature_importance_df
    .head(20)
    .to_string(index=False)
)

# ==================================================
# 13. CHECK RELATIONSHIPS BETWEEN GROUNDWATER FEATURES
# ==================================================

important_columns = [
    "Annual_Ground_Water_Recharge_ham",
    "Annual_Extractable_Ground_Water_Resource_ham",
    "Environmental_Flows_ham",
    "Extraction_Domestic_ham",
    "Extraction_Industrial_ham",
    "Extraction_Irrigation_ham",
    "Total_Ground_Water_Extraction_ham",
    "Stage_of_Ground_Water_Extraction_pct",
    "Allocation_Domestic_2025_ham",
    "Net_GW_Availability_Future_Use_ham"
]

print("\nGROUNDWATER FEATURE CORRELATION MATRIX")
print("=" * 70)

correlation_matrix = df[important_columns].corr()

print(correlation_matrix.round(3))


# ==================================================
# 14. REDUCED FEATURE MODEL
# ==================================================

# Remove highly correlated/redundant features
reduced_features = [
    col for col in X.columns
    if col not in [
        "Annual_Extractable_Ground_Water_Resource_ham",
        "Environmental_Flows_ham"
    ]
]

X_reduced = X[reduced_features]

print("\nReduced feature set:")
print("Original features:", X.shape[1])
print("Reduced features:", X_reduced.shape[1])

# Split reduced dataset
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced,
    y,
    test_size=0.20,
    random_state=42
)

# Identify column types
categorical_red = X_reduced.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_red = X_reduced.select_dtypes(
    include=["number"]
).columns.tolist()

# Preprocessing
numerical_pipeline_red = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline_red = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        ))
    ]
)

preprocessor_red = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline_red, numerical_red),
        ("categorical", categorical_pipeline_red, categorical_red)
    ]
)

# Random Forest
rf_reduced = Pipeline(
    steps=[
        ("preprocessor", preprocessor_red),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

# Train
rf_reduced.fit(X_train_red, y_train_red)

# Predict
y_pred_red = rf_reduced.predict(X_test_red)

# Evaluate
mae_red = mean_absolute_error(y_test_red, y_pred_red)
mse_red = mean_squared_error(y_test_red, y_pred_red)
rmse_red = np.sqrt(mse_red)
r2_red = r2_score(y_test_red, y_pred_red)

print("\nREDUCED RANDOM FOREST RESULTS")
print("=" * 50)
print("MAE :", round(mae_red, 4))
print("MSE :", round(mse_red, 4))
print("RMSE:", round(rmse_red, 4))
print("R2  :", round(r2_red, 4))


# ==================================================
# 15. SAVE FINAL RANDOM FOREST MODEL
# ==================================================

import joblib

# Train final model using all training data
final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

final_model.fit(X_train, y_train)

# Save the complete pipeline
model_path = "results/groundwater_random_forest.pkl"

joblib.dump(final_model, model_path)

print("\nFinal Random Forest model saved successfully!")
print("Model location:", model_path)