# Telco Customer Churn Prediction & Model Benchmarking

## a. Problem Statement
In this assignment, the objective is to:
- Implement multiple classification models on a single dataset
- Build an interactive Streamlit web application to demonstrate model evaluations
- Deploy the app on Streamlit Community Cloud (FREE)
- Share clickable links for submission and evaluation
- I used Customer Churn dataset

Customer churn poses a significant financial challenge in the telecommunications industry. The objective of this project is to build, evaluate, and deploy machine learning classification models to predict whether a customer is likely to churn (cancel their service) based on demographic attributes, account information, and service usage patterns.

---

## b. Dataset Description
* **Dataset Name:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
* **Total Instances:** 7,043 instances (Meets minimum requirement of 500)
* **Total Features:** 20 predictor features + 1 target column (`Churn`) (Meets minimum requirement of 12 features)
* **Target Variable:** `Churn` (Binary: `Yes` / `No`)
* **Feature Types:**
  * **Numerical (3):** `tenure`, `MonthlyCharges`, `TotalCharges`
  * **Categorical (17):** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

---

## c. GitHub Repository & Live App Links
* **GitHub Repository:** [https://github.com/2025ac05070-web/ML_Assignment2_2025AC05070](https://github.com/2025ac05070-web/ML_Assignment2_2025AC05070)
* **Live Streamlit Web App:** [https://mlassignment22025ac05070-if4qpesuwzqbjfachkwdax.streamlit.app/](https://mlassignment22025ac05070-if4qpesuwzqbjfachkwdax.streamlit.app/)

---

## d. Models Used & Evaluation Metrics

### Model Comparison Table
The table below displays the evaluation metrics calculated across all trained classification models using the `test_data.csv` split:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8055** | **0.8421** | 0.6572 | 0.5588 | **0.6040** | **0.4790** |
| **Decision Tree** | 0.7942 | 0.8284 | 0.6296 | 0.5455 | 0.5845 | 0.4507 |
| **KNN** | 0.7630 | 0.7930 | 0.5538 | 0.5508 | 0.5523 | 0.3911 |
| **Naive Bayes** | 0.6558 | 0.8093 | 0.4269 | **0.8663** | 0.5719 | 0.3951 |
| **Random Forest (Ensemble)** | 0.8020 | 0.8407 | **0.6599** | 0.5241 | 0.5842 | 0.4617 |

---

### Model Performance Observations

| ML Model Name | Observation About Model Performance |
| :--- | :--- |
| **Logistic Regression** | Achieves the highest overall performance across primary evaluation metrics (Accuracy: 80.55%, AUC: 0.8421, F1: 0.6040, MCC: 0.4790). Its linear decision boundary effectively models relationships between scaled features and churn probability. |
| **Decision Tree** | Pruning the tree depth (`max_depth=5`) significantly improved performance (Accuracy: 79.42%, AUC: 0.8284), effectively controlling variance and preventing overfitting observed in unconstrained trees. |
| **KNN** | Achieved reasonable performance (Accuracy: 76.30%, AUC: 0.7930), but suffered slightly due to distance metric sensitivity in a higher-dimensional post-one-hot-encoded feature space. |
| **Naive Bayes** | Yields the highest Recall (86.63%) by catching the vast majority of potential churners, but suffers from low Precision (42.69%) due to feature independence assumptions. |
| **Random Forest (Ensemble)** | Second-best overall performer (Accuracy: 80.20%, AUC: 0.8407, Precision: 0.6599). Aggregating predictions across multiple decision trees stabilizes variance while maintaining competitive accuracy. |
| **Overall Winner** | **Logistic Regression** is the clear overall winner, delivering the optimal balance between Precision and Recall while maintaining top-tier AUC and Matthews Correlation Coefficient. |

---

## Interactive Streamlit Web Application Features
1. **Dataset Upload (CSV):** Allows uploading test dataset samples (`test_data.csv`).
2. **Model Selection Dropdown:** Dynamically select between Logistic Regression, Decision Tree, KNN, Naive Bayes, and Random Forest.
3. **Evaluation Metrics Display:** Real-time generation of Accuracy, AUC, Precision, Recall, F1, and MCC metrics.
4. **Visualizations:** Displays interactive confusion matrices and full classification reports.
