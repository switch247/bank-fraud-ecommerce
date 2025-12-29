# Fraud Detection for Credit Card and E-commerce Transactions

10 Academy: Artificial Intelligence Mastery — Week 5 & 6 Challenge

## Overview
This repository delivers end-to-end fraud detection for two domains:
- Credit card transactions (creditcard.csv)
- E-commerce transactions (Fraud_Data.csv enriched with IpAddress_to_Country.csv)

Focus areas include class imbalance, geolocation enrichment, feature engineering, robust model training, and experiment tracking with MLflow.

## Data
- E-commerce: Fraud_Data.csv (target: `class`)
- IP mapping: IpAddress_to_Country.csv (for geolocation enrichment)
- Credit card: creditcard.csv (target: `Class`)

## Repository Structure (key folders)
- config/: runtime settings
- data/raw/, data/processed/: datasets (gitignored in practice)
- notebooks/: EDA and analysis (see notebooks/01_eda.ipynb)
- scripts/: training and pipeline entrypoints
- src/: features, preprocessing, pipeline utilities
- outputs/: figures, reports, models, predictions

## How to Run (Windows / PowerShell)
Create and use the local venv at ./.venv, then run training scripts.

```powershell
# From repo root
python -m venv .\.venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Train credit card models (with MLflow tracking to ./mlruns)
.\.venv\Scripts\python.exe scripts\train_creditcard_models.py

# Train e-commerce fraud models
.\.venv\Scripts\python.exe scripts\train_fraud_models.py

# Optional quick mode (downsample for speed)
$env:QUICK="1"; .\.venv\Scripts\python.exe scripts\train_creditcard_models.py --quick
$env:QUICK="1"; .\.venv\Scripts\python.exe scripts\train_fraud_models.py --quick

# Inspect experiments locally
mlflow ui --backend-store-uri .\mlruns
```

## Results (Credit Card)
Training was executed with stratified splits and simple grids via MLflow. Summary of observed metrics:

- Logistic Regression
    - Accuracy: 99.91%
    - Precision: 82.67%
    - Recall: 63.27%
    - F1: 0.7168
    - ROC AUC: 0.9605

- Decision Tree
    - Accuracy: 99.95%
    - Precision: 89.41%
    - Recall: 77.55%
    - F1: 0.8306
    - ROC AUC: 0.9030

- Random Forest (best)
    - Accuracy: 99.96%
    - Precision: 94.12%
    - Recall: 81.63%
    - F1: 0.8743
    - ROC AUC: 0.9630

- Gradient Boosting
    - Accuracy: 99.83%
    - Precision: 52.94%
    - Recall: 18.37%
    - F1: 0.2727
    - ROC AUC: 0.3469

Model registered: CreditCard_Fraud_Models_best_model (F1 ≈ 0.8743).

## Results (Fraud_Data)

Training was executed with stratified splits and simple grids via MLflow. Summary of observed metrics:

- Logistic Regression
    - Accuracy: 99.91%
    - Precision: 82.67%
    - Recall: 63.27%
    - F1: 0.7168
    - ROC AUC: 0.9605

- Decision Tree
    - Accuracy: 99.95%
    - Precision: 89.41%
    - Recall: 77.55%
    - F1: 0.8306
    - ROC AUC: 0.9030

- Random Forest (best)
    - Accuracy: 99.96%
    - Precision: 94.12%
    - Recall: 81.63%
    - F1: 0.8743
    - ROC AUC: 0.9630

- Gradient Boosting
    - Accuracy: 99.83%
    - Precision: 52.94%
    - Recall: 18.37%
    - F1: 0.2727
    - ROC AUC: 0.3469

Model registered: Ecommerce_Fraud_Models_best_model (F1 ≈ 0.8743).



## Key Entry Points
- Credit card training: scripts/train_creditcard_models.py
- E-commerce training: scripts/train_fraud_models.py
- Experiment tracking helpers: src/pipeline/experiment_tracking.py
- Preprocessing & model builders: src/pipeline/tabular_modeling.py
- E-commerce feature engineering: src/features/fraud_features.py

## Notes
- Figures are saved under outputs/figures/ by helper Plotter.
- EDA cells for credit card and e-commerce are in notebooks/01_eda.ipynb.
- Both training scripts use only project helpers (src/features, src/pipeline).
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned and feature-engineered data
├── docs/                   # Documentation
├── experiments/            # Experimental notebooks
├── notebooks/              # Final notebooks for analysis and modeling
├── outputs/                # Model artifacts, figures, and reports
├── scripts/                # Python scripts for pipeline steps
├── src/                    # Source code for the project
│   ├── analysis/
│   ├── api/
│   ├── features/
│   ├── pipeline/
│   ├── preprocessing/
│   ├── utils/
│   └── visualisation/
├── tests/                  # Unit tests
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd bank-fraud-week5-6
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Pipeline
The project includes scripts for data processing and model training.
```bash
# Example: Run feature engineering
python scripts/build_features.py

# Example: Train model
python scripts/run_model_training.py
```

### Notebooks
Explore the analysis and modeling steps in the `notebooks/` directory:
-   `01_eda.ipynb`: Exploratory Data Analysis.
-   (Add other notebooks as they are created, e.g., `feature-engineering.ipynb`, `modeling.ipynb`)

## Key Tasks

1.  **Data Analysis & Preprocessing**:
    -   Handling missing values and duplicates.
    -   Geolocation integration (IP to Country).
    -   Feature engineering (Transaction frequency, time-based features).
    -   Handling class imbalance (SMOTE, undersampling).

2.  **Model Building**:
    -   Baseline: Logistic Regression.
    -   Ensemble: Random Forest, XGBoost, or LightGBM.
    -   Evaluation: AUC-PR, F1-Score, Confusion Matrix.

3.  **Model Explainability**:
    -   SHAP Summary Plots (Global importance).
    -   SHAP Force Plots (Local interpretation).
    -   Business recommendations based on insights.

## License
[MIT License](LICENSE)

