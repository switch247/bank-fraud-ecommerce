
## How to Run (Windows / PowerShell)

### Step-by-Step Setup & Run Instructions

#### 1. Clone the Repository
```powershell
git clone https://github.com/switch247/bank-fraud-ecommerce.git
cd bank-fraud-ecommerce
```

#### 2. Set Up Python Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### 3. (Optional) Use Docker
```powershell
docker build -t bank-fraud .
docker-compose up
```

#### 4. Configure Settings
Edit files in `config/` (e.g., `settings.py`, `logging.yaml`) as needed for your environment.

#### 5. Run Feature Engineering
```powershell
python scripts/build_features.py
```

#### 6. Train Models
For credit card fraud:
```powershell
python scripts/train_creditcard_models.py
```
For e-commerce fraud:
```powershell
python scripts/train_fraud_models.py
```

#### 7. Evaluate Results
Reports and model comparisons are saved in `outputs/` and `data/processed/`.

#### 8. Run Tests
```powershell
pytest tests/
```

#### 9. Experiment Tracking
To launch the MLflow UI:
```powershell
mlflow ui --backend-store-uri .\mlruns
# Then open http://localhost:5000 in your browser
```


# Fraud Detection for E-commerce and Bank Transactions

## Business Need

Adey Innovations Inc. is a leader in financial technology, providing solutions for e-commerce and banking. The goal of this project is to improve the detection of fraud cases for both e-commerce and bank credit transactions. By leveraging advanced machine learning and geolocation analysis, we aim to build robust models that enhance transaction security, reduce financial losses, and build trust with customers and financial institutions.

Fraud detection must balance security and user experience: false positives can alienate customers, while false negatives lead to direct losses. Our models are evaluated not just on accuracy, but on their ability to balance these competing costs. Real-time monitoring and explainability are also key for business adoption.

## Data and Features

We use three main datasets:

- **Fraud_Data.csv**: E-commerce transaction data with features such as user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, and the target `class` (1 = fraud, 0 = not fraud). Highly imbalanced.
- **IpAddress_to_Country.csv**: Maps IP address ranges to countries for geolocation enrichment.
- **creditcard.csv**: Bank transaction data with anonymized features (V1-V28), Amount, Time, and the target `Class` (1 = fraud, 0 = not fraud). Also highly imbalanced.

Key feature engineering includes:
- Transaction frequency and velocity
- Time-based features (hour_of_day, day_of_week, time_since_signup)
- Geolocation integration (IP to country)
- Handling class imbalance (SMOTE, undersampling)

See the `data/` folder for raw and processed files. Sensitive data should be handled per your organization’s policies.

## Learning Outcomes

**Skills:**
- Data cleaning, preprocessing, and merging
- Feature engineering from raw data
- Handling highly imbalanced datasets
- Model training and evaluation with AUC-PR, F1-Score
- Model explainability with SHAP

**Knowledge:**
- Business and technical challenges of fraud detection
- Importance of explainability (XAI)
- Model selection based on metrics and business context

**Behaviors:**
- Business-centric problem solving
- Systematic, organized workflow

**Communication:**
- Reporting on complex statistical issues

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned and feature-engineered data
├── notebooks/              # EDA, feature engineering, modeling, explainability
├── src/                    # Source code
├── tests/                  # Unit tests
├── models/                 # Saved model artifacts
├── scripts/                # Pipeline scripts
├── requirements.txt
├── README.md
└── .gitignore
```

## Step-by-Step Setup & Run Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/switch247/bank-fraud-ecommerce.git
cd bank-fraud-ecommerce
```

### 2. Set Up Python Environment
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. (Optional) Use Docker
```sh
docker build -t bank-fraud .
docker-compose up
```

### 4. Configure Settings
Edit files in `config/` (e.g., `settings.py`, `logging.yaml`) as needed.

### 5. Data Preprocessing & Feature Engineering
```sh
python scripts/build_features.py
```

### 6. Model Training
For credit card fraud:
```sh
python scripts/train_creditcard_models.py
```
For e-commerce fraud:
```sh
python scripts/train_fraud_models.py
```

### 7. Model Evaluation & Reporting
Reports and model comparisons are saved in `outputs/` and `data/processed/`.

### 8. Run Tests
```sh
pytest tests/
```

### 9. Experiment Tracking
To launch the MLflow UI:
```sh
mlflow ui --backend-store-uri ./mlruns
# Then open http://localhost:5000 in your browser
```


## Key Implementation Entry Points

- Credit card training: scripts/train_creditcard_models.py
- E-commerce training: scripts/train_fraud_models.py
- Experiment tracking helpers: src/pipeline/experiment_tracking.py
- Preprocessing & model builders: src/pipeline/tabular_modeling.py
- E-commerce feature engineering: src/features/fraud_features.py

## Key Tasks

### Task 1: Data Analysis and Preprocessing
- Handle missing values and duplicates
- EDA: distributions, relationships, class imbalance
- Geolocation integration (IP to country)
- Feature engineering (frequency, time-based, etc.)
- Normalize/scale, encode categorical features
- Handle class imbalance (SMOTE/undersampling)

### Task 2: Model Building and Training
- Stratified train-test split
- Baseline: Logistic Regression
- Ensemble: Random Forest, XGBoost, or LightGBM
- Hyperparameter tuning
- Stratified K-Fold cross-validation
- Model comparison and selection (performance + interpretability)

### Task 3: Model Explainability
- Feature importance (built-in and SHAP)
- SHAP summary and force plots
- Business recommendations based on insights

## References
- Kaggle: Credit Card Fraud Dataset
- Kaggle: IEEE Fraud Detection Competition
- Kaggle: Fraud E-commerce Dataset
- imbalanced-learn Documentation
- scikit-learn: Precision-Recall Curves
- GeeksforGeeks: IP Address to Integer Conversion
- pandas.merge_asof Documentation
- Analytics Vidhya, DataCamp, IBM, and others (see project docs)
# Fraud Detection for Credit Card and E-commerce Transactions

## Business Context

Online payment fraud is a major risk for banks and e-commerce platforms, leading to financial losses and eroding customer trust. This project addresses the detection of fraudulent transactions using machine learning, aiming to support real-time prevention and compliance with regulatory standards. The models and pipelines here are designed to help organizations identify suspicious activity, reduce false positives, and improve operational efficiency.

## Data Source Notes

- **creditcard.csv**: Public dataset of anonymized credit card transactions, labeled for fraud, from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- **Fraud_Data.csv**: Synthetic e-commerce transaction data with fraud labels.
- **IpAddress_to_Country.csv**: Maps IP addresses to countries, enabling geolocation-based features.
- **Processed Data**: The `data/processed/` folder contains cleaned and feature-engineered datasets for modeling and evaluation.

See the `data/` folder for all raw and processed files. Sensitive or proprietary data should be handled according to your organization’s policies.

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

### Step-by-Step Setup & Run Instructions

#### 1. Clone the Repository
```powershell
git clone https://github.com/switch247/bank-fraud-ecommerce.git
cd bank-fraud-ecommerce
```

#### 2. Set Up Python Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### 3. (Optional) Use Docker
```powershell
docker build -t bank-fraud .
docker-compose up
```

#### 4. Configure Settings
Edit files in `config/` (e.g., `settings.py`, `logging.yaml`) as needed for your environment.

#### 5. Run Feature Engineering
```powershell
python scripts/build_features.py
```

#### 6. Train Models
For credit card fraud:
```powershell
python scripts/train_creditcard_models.py
```
For e-commerce fraud:
```powershell
python scripts/train_fraud_models.py
```

#### 7. Evaluate Results

### 7. Model Evaluation & Reporting
Reports and model comparisons are saved in `outputs/` and `data/processed/`.

#### 8. Run Tests
```powershell
pytest tests/
```

#### 9. Experiment Tracking
To launch the MLflow UI:
```powershell
mlflow ui --backend-store-uri .\mlruns
# Then open http://localhost:5000 in your browser
```

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

## Task 2b: Cross-Validation and Model Selection

### Cross-Validation with Aggregated Metrics
Both training scripts support **5-fold stratified cross-validation** with the `--use-cv` flag. This provides robust performance estimates with mean ± standard deviation for all metrics.

```powershell
# Run with cross-validation (can combine with --quick for faster iteration)
.\.venv\Scripts\python.exe scripts\train_creditcard_models.py --use-cv --quick
.\.venv\Scripts\python.exe scripts\train_fraud_models.py --use-cv --quick

# Full dataset with CV (takes longer)
.\.venv\Scripts\python.exe scripts\train_creditcard_models.py --use-cv
.\.venv\Scripts\python.exe scripts\train_fraud_models.py --use-cv
```

**Cross-validation features:**
- Stratified 5-fold CV maintains class distribution in each fold
- Computes mean and standard deviation for: accuracy, precision, recall, F1, ROC-AUC
- Logs individual fold results and aggregated statistics to MLflow
- Provides more reliable performance estimates than single train/test split

### Model Comparison and Selection
After training, both scripts automatically:

1. **Generate comparison reports** saved to `outputs/`:
   - `model_comparison_creditcard.csv` - Credit card models comparison
   - `model_comparison_fraud.csv` - E-commerce fraud models comparison

2. **Select best model** using intelligent criteria:
   - **Primary metric**: F1-score (optimal for imbalanced classification)
   - **Interpretability consideration**: When models perform similarly (within 2% F1), selects the more interpretable model
   - **Interpretability ranking**: Logistic Regression (5) > Decision Tree (4) > Random Forest (3) > Gradient Boosting (2) > XGBoost (1)

**Example output:**
```
MODEL COMPARISON AND SELECTION
================================================================================

Model Comparison:
     model_name  accuracy  precision    recall        f1   roc_auc
random_forest__smote    0.9996     0.9412    0.8163    0.8743    0.9630
decision_tree__smote    0.9995     0.8941    0.7755    0.8306    0.9030
log_reg__smote          0.9991     0.8267    0.6327    0.7168    0.9605

--------------------------------------------------------------------------------
FINAL MODEL SELECTION
--------------------------------------------------------------------------------
Selected Model: random_forest__smote
Justification: Selected 'random_forest__smote' as it has the best f1 score of 0.8743
--------------------------------------------------------------------------------
```

### Viewing Results
```powershell
# View comparison reports
cat data\processed\model_comparison_creditcard.csv
cat data\processed\model_comparison_fraud.csv

# Launch MLflow UI to explore all experiments
mlflow ui --backend-store-uri .\mlruns
# Then open http://localhost:5000 in your browser
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

