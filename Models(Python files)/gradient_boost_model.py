import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1 Load augmented dataset
df = pd.read_csv("../augmented_data.csv")
print("Dataset size:", df.shape)

# 2 Define features and target
X = df.drop(columns=["Yield", "Pitch", "ToolVelocity"])
y = df["Yield"]

# 3 Create bins for stratified sampling
bins = pd.qcut(y, q=10, labels=False)

# 4 Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=bins,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 5 Train Gradient Boosting Model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# 6 Model Evaluation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print("GRADIENT BOOSTING MODEL RESULTS")
print("="*50)
print(f"R2 Score: {r2}")
print(f"RMSE: {rmse}")
print("="*50)
