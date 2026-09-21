import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "dataset/used_cars.csv"
MODEL_PATH = "model/car_price_model.pkl"

df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

df = df.drop_duplicates()

numeric_columns = [
    "year", "selling_price", "km_driven",
    "mileage", "engine", "max_power", "seats"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["selling_price"])

features = [
    "name", "year", "km_driven", "fuel", "seller_type",
    "transmission", "owner", "mileage", "engine",
    "max_power", "seats"
]

X = df[features]
y = df["selling_price"]

categorical_features = [
    "name", "fuel", "seller_type", "transmission", "owner"
]

numerical_features = [
    "year", "km_driven", "mileage", "engine", "max_power", "seats"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
