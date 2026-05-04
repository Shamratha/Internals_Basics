import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/training_data.csv")
X = df.drop("inflation_index", axis=1)
y = df["inflation_index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

mlflow.set_experiment("econpulse-inflation-index")

with mlflow.start_run(run_name="tuning-econpulse") as parent_run:
    parent_run_id = parent_run.info.run_id

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=18,          # all combos: 3*3*2 = 18
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    # Log each trial as a nested run
    for i, (params, mean_score, std_score) in enumerate(zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"],
    )):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params(params)
            cv_mae = -mean_score
            mlflow.log_metric("cv_mae", cv_mae)
            mlflow.log_metric("cv_mae_std", std_score)

    best_model = search.best_estimator_
    test_preds = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, test_preds)

    best_cv_mae = -search.best_score_

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_test_mae", best_mae)
    mlflow.log_metric("best_cv_mae", best_cv_mae)
    mlflow.sklearn.log_model(best_model, artifact_path="tuned_model")

output = {
    "search_type": "random",
    "n_folds": 3,
    "total_trials": len(search.cv_results_["params"]),
    "best_params": {k: (v if v is not None else None) for k, v in search.best_params_.items()},
    "best_mae": round(best_mae, 6),
    "best_cv_mae": round(best_cv_mae, 6),
    "parent_run_name": "tuning-econpulse",
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=2)

# Save tuned model meta for next step
with open("models/tuned_model_meta.json", "w") as f:
    json.dump({
        "best_params": search.best_params_,
        "best_mae": round(best_mae, 6),
        "parent_run_id": parent_run_id,
    }, f, default=str)

print("Task 2 complete. Results saved to results/step2_s2.json")
print(f"Best params: {search.best_params_}")
print(f"Best test MAE: {best_mae:.4f} | Best CV MAE: {best_cv_mae:.4f}")