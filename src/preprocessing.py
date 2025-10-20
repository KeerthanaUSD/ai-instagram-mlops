import pandas as pd
import os

# Dummy data creation
os.makedirs("/opt/ml/processing/output", exist_ok=True)
df = pd.DataFrame({
    "feature1": [1,2,3,4,5],
    "feature2": [10,20,30,40,50],
    "label": [0,1,0,1,0]
})

# Save dummy train data
df.to_csv("/opt/ml/processing/output/train.csv", index=False)
print("✅ Preprocessing complete. Dummy data saved.")

