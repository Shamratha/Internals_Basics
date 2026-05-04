import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

os.makedirs("results", exist_ok=True)

# Load tuned params
with open("models/tuned_model_meta.json") as f:
    meta = json.load(f)

best_params = meta["best_params"]
# max_depth may be stored as string "None"
if best_params.get("max_depth") in (None, "None", "null"):
    best_params["max_depth"] = None

df = pd.read_csv("data/training_data.csv")
X = df.drop("inflation_index", axis=1)
y = df["inflation_index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("econpulse-inflation-index")

REGISTERED_MODEL_NAME = "econpulse-inflation-index-predictor"

with mlflow.start_run(run_name="register-v1") as run:
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_params(best_params)
    mlflow.log_metric("mae", mae)

    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    run_id = run.info.run_id

client = MlflowClient()
# Get the latest version
versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
version_number = max(int(v.version) for v in versions)

output = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "version": version_number,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": round(mae, 6),
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=2)

# Save for task 4
with open("models/registered_meta.json", "w") as f:
    json.dump({
        "version_1": version_number,
        "run_id_v1": run_id,
        "mae_v1": round(mae, 6),
        "registered_model_name": REGISTERED_MODEL_NAME,
    }, f)

print("Task 3 complete. Results saved to results/step3_s6.json")
print(f"Registered: {REGISTERED_MODEL_NAME} v{version_number} | MAE={mae:.4f}")
