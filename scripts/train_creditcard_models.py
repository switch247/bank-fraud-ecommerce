import sys
import os
from pathlib import Path
import pandas as pd
import mlflow

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline.tabular_modeling import build_preprocessor, build_classification_models, split_features_target
from src.pipeline.experiment_tracking import run_experiment, register_best_model


def main():
    raw_path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    if not raw_path.exists():
        print(f"Data file not found at {raw_path}")
        return

    print(f"Loading credit card data from {raw_path}...")
    df = pd.read_csv(raw_path)

    # Optional quick run downsampling via env var QUICK or --quick flag
    if os.environ.get("QUICK") == "1" or "--quick" in sys.argv:
        n = min(80000, len(df))
        df = df.sample(n=n, random_state=42)
        print(f"[quick] Downsampled to {len(df)} rows for faster iteration")

    # Ensure target is int {0,1}
    if df["Class"].dtype != int:
        df["Class"] = df["Class"].astype(int)

    # Feature/target split
    drop_cols = []  # nothing special to drop; all features are numeric already
    X_train, X_test, y_train, y_test = split_features_target(
        df, target="Class", drop_cols=drop_cols, test_size=0.2, random_state=42, stratify=True
    )

    # Determine columns for preprocessor
    # For creditcard.csv everything except target is numeric
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = []

    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    # Build models (LogReg, DT, RF, GBDT, optional XGB)
    models = build_classification_models(preprocessor=preprocessor)

    # MLflow local tracking
    experiment_name = "CreditCard_Fraud_Models"
    mlflow.set_tracking_uri("file:./mlruns")

    for model_name, model_pipeline in models.items():
        print(f"Running experiment for {model_name}...")

        # Small param grids
        param_grid = None
        if "log_reg" in model_name:
            param_grid = {"model__C": [0.1, 1.0, 3.0]}
        elif "decision_tree" in model_name:
            param_grid = {"model__max_depth": [5, 10, None], "model__min_samples_split": [2, 5]}
        elif "random_forest" in model_name:
            param_grid = {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]}
        elif "gradient_boosting" in model_name:
            param_grid = {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]}
        elif "xgb" in model_name:
            param_grid = {"model__n_estimators": [200, 400], "model__learning_rate": [0.05, 0.1]}

        run_experiment(
            experiment_name=experiment_name,
            model_name=model_name,
            model=model_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            param_grid=param_grid,
            search_type="grid",
        )

    print("Registering best model by F1...")
    register_best_model(experiment_name, metric="f1_score")


if __name__ == "__main__":
    main()
