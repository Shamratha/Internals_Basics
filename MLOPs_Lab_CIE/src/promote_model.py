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

with open("models/registered_meta.json") as f:
    meta = json.load(f)

with open("models/tuned_model_meta.json") as f:
    tuned = json.load(f)

best_params = tuned["best_params"]
if best_params.get("max_depth") in (None, "None", "null"):
    best_params["max_depth"] = None

REGISTERED_MODEL_NAME = meta["registered_model_name"]
version_1 = meta["version_1"]
mae_v1 = meta["mae_v1"]

df = pd.read_csv("data/training_data.csv")
X = df.drop("inflation_index", axis=1)
y = df["inflation_index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("econpulse-inflation-index")

# Train challenger with random_state=99
with mlflow.start_run(run_name="register-v2-challenger") as run:
    challenger = RandomForestRegressor(**best_params, random_state=99)
    challenger.fit(X_train, y_train)
    preds = challenger.predict(X_test)
    mae_v2 = mean_absolute_error(y_test, preds)

    mlflow.log_params(best_params)
    mlflow.log_metric("mae", mae_v2)

    mlflow.sklearn.log_model(
        challenger,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    run_id_v2 = run.info.run_id

client = MlflowClient()
versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
version_2 = max(int(v.version) for v in versions)

# Set champion alias to version 1 first
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", str(version_1))

# Decide promotion
if mae_v2 < mae_v1:
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", str(version_2))
    champion_version = version_2
    action = "promoted"
    print(f"Challenger (v{version_2}, MAE={mae_v2:.4f}) is better → PROMOTED to champion")
else:
    champion_version = version_1
    action = "kept"
    print(f"Champion (v{version_1}, MAE={mae_v1:.4f}) is still better → KEPT as champion")

output = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "alias_name": "champion",
    "champion_version": champion_version,
    "challenger_version": version_2,
    "action": action,
}

with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 4 complete. Results saved to results/step4_s7.json")