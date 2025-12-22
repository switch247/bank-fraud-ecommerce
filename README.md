# Fraud Detection for E-commerce and Banking Transactions

**10 Academy: Artificial Intelligence Mastery - Week 5 & 6 Challenge**

## Project Overview

This project aims to improve the detection of fraud cases for e-commerce transactions and bank credit transactions. As a data scientist at Adey Innovations Inc., the goal is to create accurate and robust fraud detection models that handle the unique challenges of both types of transaction data, including class imbalance, geolocation analysis, and transaction pattern recognition.

Key objectives:
- Analyze and preprocess transaction data.
- Engineer features to identify fraud patterns.
- Build and train machine learning models (Logistic Regression, Random Forest, XGBoost/LightGBM).
- Evaluate models using metrics appropriate for imbalanced data (AUC-PR, F1-Score).
- Interpret model decisions using SHAP (SHapley Additive exPlanations).

## Business Need

Fraud detection is critical for minimizing financial losses and building trust with customers. A key challenge is balancing security with user experience—minimizing false positives (blocking legitimate users) while maximizing the detection of actual fraud (false negatives). This project focuses on building models that optimize this trade-off.

## Data Sources

The project utilizes three main datasets:

1.  **Fraud_Data.csv**: E-commerce transaction data.
    -   Features: `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`.
    -   Target: `class` (1: Fraud, 0: Non-Fraud).
2.  **IpAddress_to_Country.csv**: Maps IP address ranges to countries.
    -   Used to enrich `Fraud_Data.csv` with geolocation information.
3.  **creditcard.csv**: Bank transaction data.
    -   Features: `Time`, `Amount`, and anonymized PCA features `V1` to `V28`.
    -   Target: `Class` (1: Fraud, 0: Non-Fraud).

**Critical Challenge**: Both datasets are highly imbalanced, requiring specialized techniques like SMOTE or undersampling.

## Project Structure

```
├── .vscode/
├── config/                 # Configuration files
├── data/
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

