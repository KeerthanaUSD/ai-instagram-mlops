
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Read dummy data
train_path = "/opt/ml/input/data/train/train.csv"
df = pd.read_csv(train_path)

X = df[["feature1","feature2"]]
y = df["label"]

# Train dummy model
model = LogisticRegression()
model.fit(X, y)

# Save model
os.makedirs("/opt/ml/model", exist_ok=True)
joblib.dump(model, "/opt/ml/model/model.joblib")
print("✅ Training complete. Dummy model saved.")
