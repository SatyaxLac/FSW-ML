import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score
)

# Load dataset
df=pd.read_csv('../fsw_clean_dataset.csv')

# Define features and target
X = df[['D','V','N','HeatInput']]
y = df['Yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# Calculate regression metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Classification task
threshold = 430

y_train_class = (y_train > threshold).astype(int)
y_test_class = (y_test > threshold).astype(int)

# Train XGBoost Classifier
xgb_clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb_clf.fit(X_train_scaled, y_train_class)

# Make predictions
y_pred_class = xgb_clf.predict(X_test_scaled)

# Calculate classification metrics
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)

print("\n" + "="*50)
print("XGBOOST MODEL RESULTS")
print("="*50)
print("\nRegression Metrics:")
print(f"R2 Score: {r2}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

print("\nClassification Metrics (Threshold = 430):")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print("="*50)
