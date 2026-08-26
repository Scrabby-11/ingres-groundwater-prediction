import pandas as pd

# Load the CSV dataset
file_path = "/Users/shashwatnigam/Desktop/INGRES PROJ/data/INGRES_Averaged_Groundwater_Dataset.csv"

df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# --------------------------------------------------
# STEP 2: DATA QUALITY CHECK
# --------------------------------------------------

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Basic statistical summary
print("\nStatistical summary:")
print(df.describe())

# --------------------------------------------------
# STEP 3: REMOVE USELESS COLUMNS
# --------------------------------------------------

# These columns contain almost entirely missing values
columns_to_remove = [
    "Major_Quality_Parameters",
    "Other_Quality_Parameters"
]

df = df.drop(columns=columns_to_remove)

print("\nColumns removed:")
print(columns_to_remove)

print("\nNew dataset shape:", df.shape)

print("\nRemaining columns:")
print(df.columns.tolist())

# --------------------------------------------------
# STEP 4: IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# --------------------------------------------------

categorical_columns = df.select_dtypes(include=["object"]).columns
numerical_columns = df.select_dtypes(include=["number"]).columns

print("\nCategorical columns:")
print(categorical_columns.tolist())

print("\nNumerical columns:")
print(numerical_columns.tolist())

# --------------------------------------------------
# STEP 5: HANDLE MISSING VALUES
# --------------------------------------------------

print("\nMissing values before treatment:")
print(df.isnull().sum())

# Handle numerical columns using median
for col in numerical_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Handle categorical columns using mode
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after treatment:")
print(df.isnull().sum().sum())

# --------------------------------------------------
# STEP 6: OUTLIER DETECTION USING IQR
# --------------------------------------------------

print("\nOutlier detection:")

outlier_counts = {}

for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    outlier_counts[col] = len(outliers)

# Display outlier counts
for col, count in outlier_counts.items():
    print(f"{col}: {count} outliers")

  # --------------------------------------------------
# STEP 7: OUTLIER TREATMENT USING IQR CAPPING
# --------------------------------------------------

# Define the target variable
target_column = "Net_GW_Availability_Future_Use_ham"

# Apply outlier capping only to input features
feature_columns = [
    col for col in numerical_columns
    if col != target_column
]

for col in feature_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = df[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

print("\nOutlier treatment completed using IQR capping.")
print("Target variable was not capped.")
print("Number of rows:", len(df))


# --------------------------------------------------
# STEP 8: ENCODE CATEGORICAL VARIABLES
# --------------------------------------------------

# One-hot encode categorical columns
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

print("\nCategorical variables encoded successfully.")

print("New dataset shape:", df.shape)

print("\nData types after encoding:")
print(df.dtypes)

# --------------------------------------------------
# STEP 9: SEPARATE FEATURES AND TARGET
# --------------------------------------------------

target_column = "Net_GW_Availability_Future_Use_ham"

# Features
X = df.drop(columns=[target_column])

# Target
y = df[target_column]

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

print("\nTarget variable:")
print(target_column)

# --------------------------------------------------
# STEP 10: TRAIN-TEST SPLIT
# --------------------------------------------------

from sklearn.model_selection import train_test_split

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

# --------------------------------------------------
# STEP 11: FEATURE SCALING
# --------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)