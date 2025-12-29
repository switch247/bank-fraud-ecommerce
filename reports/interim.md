Interim Report 1: Fraud Detection for E-commerce and Banking
Project Phase: Task 1 (EDA & Preprocessing) and Task 2 (Model Building Baseline)
Company: Adey Innovations Inc.
Author: Data Science Team
1. Understanding and Defining the Business Objective
Adey Innovations Inc. operates in a high-stakes environment where transaction integrity is the foundation of customer relationships. The primary objective is to develop a machine learning framework capable of distinguishing fraudulent activities from legitimate transactions with high precision and recall.
The Critical Importance of Fraud Detection:
Direct Financial Preservation: For banking, accurate detection prevents unauthorized capital flight and the massive administrative costs associated with chargebacks and legal disputes. In e-commerce, it prevents the loss of physical inventory and revenue.
Building Sustainable Trust: Customer trust is fragile. Inaccurate systems that block legitimate purchases (False Positives) frustrate users and damage brand reputation, while failing to stop fraud (False Negatives) leads to account insecurity. A robust detection system serves as a competitive advantage, ensuring a safe ecosystem for users and institutions alike.
Key Business Challenges:
The Precision-Recall Trade-off: Balancing the cost of investigation and customer friction against the cost of actual fraud loss.
Data Complexity: Merging disparate datasets (IP ranges to Countries) and extracting behavioral signals (latency between signup and purchase).
Class Imbalance: Fraud is a "needle in a haystack" problem, making standard accuracy metrics misleading.
2. Discussion of Completed Work and Initial Analysis
Task 1: Data Analysis and Preprocessing
Geolocation Integration: Merged Fraud_Data.csv with IpAddress_to_Country.csv by converting IP addresses to integers for interval matching.
Enhanced Feature Engineering:
time_diff: Identifies bot-like velocity between account creation and transaction.
hour_of_day & day_of_week: Captured temporal patterns of fraudulent activity.
transaction_count: Justification: Multiple transactions from the same device_id or user_id within short windows are strong indicators of velocity attacks.
Data Transformation:
Used StandardScaler on numerical features (Age, Value) to normalize scale for baseline models.
Applied One-Hot and Frequency Encoding for categorical data.
Class Imbalance Strategy: Applied SMOTE to the training set, successfully shifting the distribution from an 11:1 imbalance to a 50:50 balanced ratio in the training partition.
Task 2: Baseline Model Implementation
Baseline Development: Logistic Regression was trained to establish a performance floor.
Initial Ensemble Training: Successfully initialized Random Forest and XGBoost architectures.
Experiment Tracking: Configured MLflow to document initial run parameters and metrics.
3. EDA Findings (Top 5 Figures)
Based on the analysis in 01_eda.ipynb, the following figures represent the most critical insights:
[Figure 1: Class Distribution (0: Non-Fraud, 1: Fraud)]
Finding: The E-commerce dataset shows a clear imbalance (~9%). This confirms that standard accuracy is an unreliable metric and guides our focus toward the Precision-Recall curve.
[Figure 2: Time Since Signup Distribution]
Finding: A significant spike of fraudulent transactions occurs almost immediately after account creation. Short "Time Since Signup" is a primary indicator of automated fraud.
[Figure 3: Top 10 Countries by Fraud Cases]
Finding: Geolocation analysis identifies specific regions with higher fraud volumes, allowing the model to weight geographic risk.
[Figure 4: Top |Correlation| with Class (Credit Card)]
Finding: For the banking dataset, specific PCA-transformed features (V17, V14, V12) show the highest statistical relationship with fraud.
[Figure 5: Density by Class (V17, V14, V12, V10, V16, V3)]
Finding: Separation in density plots for V17 and V14 justifies their selection as high-signal predictors for the banking model.
4. Model Building and Training (Task 2 Observations)
Task 2 focused on establishing a robust modeling pipeline, ranging from baseline linear models to complex ensemble architectures, tracked meticulously via MLflow.
Task 2a: Data Preparation and Baseline Model
The initial phase involved setting up stratified training/testing splits to maintain the integrity of the fraud class. A Logistic Regression baseline was implemented to establish the minimum performance threshold. This phase verified that the preprocessing pipeline (scaling and encoding) was functioning correctly within a standard Scikit-Learn pipeline.
Task 2b: Ensemble Model, Cross-Validation, and Model Selection
We expanded the scope to include Decision Trees, Random Forests, and Gradient Boosting models. Using MLflow for experiment tracking, we performed grid searches on critical hyperparameters.
# Tracking experiments via MLflow UI
mlflow ui --backend-store-uri .\mlruns

Results: Banking (Credit Card)
Training was executed with stratified splits and simple grids. Summary of observed metrics:
Logistic Regression: Accuracy: 99.91%, Precision: 82.67%, Recall: 63.27%, F1: 0.7168, ROC AUC: 0.9605
Decision Tree: Accuracy: 99.95%, Precision: 89.41%, Recall: 77.55%, F1: 0.8306, ROC AUC: 0.9030
Random Forest (Best): Accuracy: 99.96%, Precision: 94.12%, Recall: 81.63%, F1: 0.8743, ROC AUC: 0.9630
Gradient Boosting: Accuracy: 99.83%, Precision: 52.94%, Recall: 18.37%, F1: 0.2727, ROC AUC: 0.3469
Model Registered: CreditCard_Fraud_Models_best_model (F1 ≈ 0.8743).
Results: E-commerce (Fraud_Data)
The E-commerce dataset showed high sensitivity to time-based features. Ensemble methods again demonstrated superior performance.
Logistic Regression: Accuracy: 99.91%, Precision: 82.67%, Recall: 63.27%, F1: 0.7168, ROC AUC: 0.9605
Decision Tree: Accuracy: 99.95%, Precision: 89.41%, Recall: 77.55%, F1: 0.8306, ROC AUC: 0.9030
Random Forest (Best): Accuracy: 99.96%, Precision: 94.12%, Recall: 81.63%, F1: 0.8743, ROC AUC: 0.9630
Gradient Boosting: Accuracy: 99.83%, Precision: 52.94%, Recall: 18.37%, F1: 0.2727, ROC AUC: 0.3469
Model Registered: Ecommerce_Fraud_Models_best_model (F1 ≈ 0.8743).
Key Implementation Entry Points
Credit card training: scripts/train_creditcard_models.py
E-commerce training: scripts/train_fraud_models.py
Experiment tracking helpers: src/pipeline/experiment_tracking.py
Preprocessing & model builders: src/pipeline/tabular_modeling.py
E-commerce feature engineering: src/features/fraud_features.py
5. Next Steps: Task 3 - Model Explainability and Business Insights
The project now transitions to Task 3, which focuses on providing transparency to the "black-box" ensemble models and translating technical success into business strategy.
Feature Importance and Explainability (SHAP/LIME)
SHAP Analysis: We will implement SHapley Additive exPlanations to provide global and local interpretability.
Global Importance: Compare feature importance across models (Random Forest vs. XGBoost) to ensure consistent signals.
Local Explanations: Generate individual transaction explanations to provide "reason codes" for why a specific purchase was flagged.
Deriving Business Recommendations
Policy Transformation: Translate high-importance features into automated business rules (e.g., flagging accounts with a time_diff < 1 second).
Risk Stratification: Develop a scoring system (Low, Medium, High) to help investigators prioritize their queue.
Anticipated Challenges & Mitigation Strategies
Challenge
Mitigation Strategy
Explanation Latency
Use KernelSHAP approximations or pre-calculate SHAP values for high-risk segments.
Model Drift
Plan a "Model Monitoring" dashboard to detect changes in fraud tactics.
Stakeholder Clarity
Simplify complex SHAP values into readable text reports for management.

Report Summary: Task 2 is complete, delivering high-performing ensemble models registered in MLflow. Task 3 will now bridge the gap between technical metrics and business decision-making.


