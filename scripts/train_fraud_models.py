import sys
import os
from pathlib import Path
import pandas as pd
import mlflow
from imblearn.over_sampling import SMOTENC

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.fraud_features import merge_ip_country, add_time_features, add_transaction_frequency
from src.pipeline.tabular_modeling import build_preprocessor, build_classification_models, split_features_target
from src.pipeline.experiment_tracking import run_experiment, register_best_model


def _prepare_fraud_df(df: pd.DataFrame, ip_country_df: pd.DataFrame | None = None) -> pd.DataFrame:
    # Apply feature engineering helpers if raw datetime/IP columns exist
    cols = set(df.columns)
    out = df.copy()

    if {"signup_time", "purchase_time"}.issubset(cols):
        out = add_time_features(out)
    if ip_country_df is not None and "ip_address" in cols:
        out = merge_ip_country(out, ip_country_df)
    if {"user_id", "device_id", "ip_address"}.issubset(cols):
        out = add_transaction_frequency(out)

    return out


def main():
    # Prefer processed file; fall back to raw + ip mapping
    processed_path = PROJECT_ROOT / "data" / "processed" / "Fraud_Data_Processed.csv"
    raw_path = PROJECT_ROOT / "data" / "raw" / "Fraud_Data.csv"
    ip_path = PROJECT_ROOT / "data" / "raw" / "IpAddress_to_Country.csv"

    if processed_path.exists():
        print(f"Loading processed fraud data from {processed_path}...")
        df = pd.read_csv(processed_path)
        ip_country_df = None
    else:
        print(f"Loading raw fraud data from {raw_path}...")
        df = pd.read_csv(raw_path)
        ip_country_df = pd.read_csv(ip_path) if ip_path.exists() else None
        df = _prepare_fraud_df(df, ip_country_df)

    # Optional quick run downsampling via env var QUICK or --quick flag
    if os.environ.get("QUICK") == "1" or "--quick" in sys.argv:
        n = min(80000, len(df))
        df = df.sample(n=n, random_state=42)
        print(f"[quick] Downsampled to {len(df)} rows for faster iteration")

    # Target column name is 'class'
    if df["class"].dtype != int:
        df["class"] = df["class"].astype(int)

    # Drop columns that are not suitable for modeling directly
    drop_cols = [c for c in ["signup_time", "purchase_time"] if c in df.columns]

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = split_features_target(
        df, target="class", drop_cols=drop_cols, test_size=0.2, random_state=42, stratify=True
    )

    # Identify numeric/categorical columns for preprocessing
    datetime_like = []  # we've dropped time columns already above
    numeric_cols = [c for c in X_train.select_dtypes(include=["number"]).columns if c not in datetime_like]
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Print training class distribution before resampling
    train_dist = y_train.value_counts().sort_index()
    train_pct = (y_train.value_counts(normalize=True).sort_index() * 100).round(3)
    print("Training class distribution BEFORE resampling:")
    print({int(k): int(v) for k, v in train_dist.items()})
    print({int(k): float(v) for k, v in train_pct.items()})

    # Apply SMOTENC only on training split to handle mixed types
    # Build categorical feature indices relative to X_train columns
    cat_indices = [X_train.columns.get_loc(c) for c in categorical_cols]
    smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_train_res, y_train_res = smotenc.fit_resample(X_train, y_train)

    # Print class distribution after resampling
    res_dist = y_train_res.value_counts().sort_index()
    res_pct = (y_train_res.value_counts(normalize=True).sort_index() * 100).round(3)
    print("Training class distribution AFTER SMOTENC resampling:")
    print({int(k): int(v) for k, v in res_dist.items()})
    print({int(k): float(v) for k, v in res_pct.items()})

    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    # Build models
    models = build_classification_models(preprocessor=preprocessor)

    # MLflow local tracking
    experiment_name = "Ecommerce_Fraud_Models"
    mlflow.set_tracking_uri("file:./mlruns")

    for model_name, model_pipeline in models.items():
        print(f"Running experiment for {model_name}...")

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
            model_name=f"{model_name}__smotenc",
            model=model_pipeline,
            X_train=X_train_res,
            y_train=y_train_res,
            X_test=X_test,
            y_test=y_test,
            param_grid=param_grid,
            search_type="grid",
        )

    print("Registering best model by F1...")
    register_best_model(experiment_name, metric="f1_score")


if __name__ == "__main__":
    main()
