"""
MLOps Training Script with MLflow Integration
Student: Sricharan | Roll No: 2022BCS0111
Experiment Name: sricharan_experiment

Supports multiple models, hyperparameters, feature selection, and dataset versioning.
All runs are logged to MLflow for experiment tracking.
"""

import argparse
import json
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature subsets ────────────────────────────────────────────────────────
ALL_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

SELECTED_FEATURES = [
    "volatile acidity", "citric acid", "total sulfur dioxide",
    "sulphates", "alcohol", "density"
]


def load_dataset(version: str = "v1") -> pd.DataFrame:
    """Load dataset by version. v2 applies outlier removal via IQR."""
    path = os.path.join(DATA_DIR, "winequality-red.csv")
    data = pd.read_csv(path, sep=";")

    if version == "v2":
        # Outlier removal using IQR method
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        Q1 = data[numeric_cols].quantile(0.25)
        Q3 = data[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((data[numeric_cols] < (Q1 - 1.5 * IQR)) |
                 (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        data = data[mask].reset_index(drop=True)
        print(f"Dataset v2: {len(data)} samples after outlier removal")
    else:
        print(f"Dataset v1: {len(data)} samples (original)")

    return data


def get_model(model_type: str, **kwargs):
    """Return a scikit-learn model based on type."""
    models = {
        "lasso": lambda: Lasso(alpha=kwargs.get("alpha", 0.1)),
        "ridge": lambda: Ridge(alpha=kwargs.get("alpha", 1.0)),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 10),
            random_state=42,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=42,
        ),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(models)}")
    return models[model_type]()


def train_and_log(args):
    """Train a model and log everything to MLflow."""

    # ── Load data ──────────────────────────────────────────────────────────
    data = load_dataset(args.dataset_version)

    features = SELECTED_FEATURES if args.features == "selected" else ALL_FEATURES
    X = data[features]
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Model ──────────────────────────────────────────────────────────────
    model = get_model(
        args.model_type,
        alpha=args.alpha,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    # ── MLflow tracking ────────────────────────────────────────────────────
    mlflow.set_experiment("sricharan_experiment")

    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("dataset_version", args.dataset_version)
        mlflow.log_param("features", args.features)
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("feature_names", ", ".join(features))
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)

        if args.model_type in ("lasso", "ridge"):
            mlflow.log_param("alpha", args.alpha)
        if args.model_type in ("random_forest", "gradient_boosting"):
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
        if args.model_type == "gradient_boosting":
            mlflow.log_param("learning_rate", args.learning_rate)

        # Train
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("MSE", round(mse, 4))
        mlflow.log_metric("RMSE", round(rmse, 4))
        mlflow.log_metric("MAE", round(mae, 4))
        mlflow.log_metric("R2", round(r2, 4))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Print summary
        print(f"\n{'='*50}")
        print(f"Run: {args.run_name}")
        print(f"Model: {args.model_type}")
        print(f"Dataset: {args.dataset_version}")
        print(f"Features: {args.features} ({len(features)})")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        print(f"{'='*50}\n")

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(model, model_path)

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "run_name": args.run_name,
        "model_type": args.model_type,
        "dataset_version": args.dataset_version,
        "features": args.features,
        "num_features": len(features),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "student_name": "Sricharan",
        "roll_no": "2022BCS0111",
    }

    results_path = os.path.join(MODELS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # ── Append to aggregate output_metrics.json ────────────────────────────
    aggregate_path = os.path.join(BASE_DIR, "output_metrics.json")
    if os.path.exists(aggregate_path):
        with open(aggregate_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Replace if same run_name exists
    all_results = [r for r in all_results if r.get("run_name") != args.run_name]
    all_results.append(results)

    with open(aggregate_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Results saved to {results_path}")
    print(f"Aggregate metrics updated at {aggregate_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Training with MLflow")
    parser.add_argument("--run_name", type=str, default="run1", help="MLflow run name")
    parser.add_argument("--model_type", type=str, default="lasso",
                        choices=["lasso", "ridge", "random_forest", "gradient_boosting"])
    parser.add_argument("--alpha", type=float, default=0.1, help="Regularization (Lasso/Ridge)")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Max tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate (GB)")
    parser.add_argument("--dataset_version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--features", type=str, default="all", choices=["all", "selected"])
    args = parser.parse_args()

    train_and_log(args)
