import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/training_data.csv")
X = df.drop("inflation_index", axis=1)
y = df["inflation_index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("econpulse-inflation-index")

results = []

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Log params
        params = model.get_params()
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, artifact_path="model")

        results.append({
            "name": name,
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 6),
            "run_id": run.info.run_id,
        })

        print(f"{name} — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

best = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "econpulse-inflation-index",
    "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]} for r in results],
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"],
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nTask 1 complete. Results saved to results/step1_s1.json")
print(f"Best model: {best['name']} (MAE={best['mae']})")

# Save best model name for next scripts
with open("models/best_model_meta.json", "w") as f:
    json.dump({"best_model": best["name"], "best_run_id": best["run_id"]}, f)